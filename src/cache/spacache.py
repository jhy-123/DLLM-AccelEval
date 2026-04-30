"""
SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models.

Drop-in cache implementation for the d2cache framework.
Instantiated via Hydra with only (model_config, **yaml_params).
proxy_projs and update_ratio are built automatically inside __init__.
"""

import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.frame import Frame, FrameDelta
from src.cache.base import dCache, AttentionContext, FFNContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported model configs
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    # key: last component of pretrained_model_name_or_path (or model_type string)
    "LLaDA-8B-Instruct": dict(
        n_layers=32, peak_pos=0.75,
        freq_start=0.03, freq_peak=0.25, freq_end=0.13,
    ),
    "LLaDA-1.5": dict(
        n_layers=32, peak_pos=0.8,
        freq_start=0.03, freq_peak=0.25, freq_end=0.13,
    ),
    "Dream-v0-Instruct-7B": dict(
        n_layers=28, peak_pos=0.5,
        freq_start=0.05, freq_peak=0.3, freq_end=0.25,
    ),
}

# ---------------------------------------------------------------------------
# Update-ratio helpers (from original spa-cache/cache.py)
# ---------------------------------------------------------------------------

def _split_gaussian_frequency(
    layer: np.ndarray,
    n_layers: int,
    peak_pos: float = 0.75,
    freq_peak: float = 0.25,
    freq_start: float = 0.01,
    freq_end: float = 0.15,
    freq_min: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    t = layer / n_layers
    freq_peak  = freq_peak  * scale
    freq_start = freq_start * scale
    freq_end   = freq_end   * scale
    mu = peak_pos

    if freq_start >= freq_peak or freq_end >= freq_peak:
        raise ValueError("freq_start and freq_end must be < freq_peak.")

    sigma_L = np.sqrt(-(0 - mu) ** 2 / (2 * np.log(max(freq_start, 1e-9) / freq_peak)))
    sigma_R = np.sqrt(-(1 - mu) ** 2 / (2 * np.log(max(freq_end,   1e-9) / freq_peak)))

    values     = np.zeros_like(t, dtype=float)
    left_mask  = t <  mu
    right_mask = t >= mu
    values[left_mask]  = (
        freq_peak * np.exp(-((t[left_mask] - mu) ** 2) / (2 * sigma_L ** 2))
    ).clip(freq_min, None)
    values[right_mask] = (
        freq_peak * np.exp(-((t[right_mask] - mu) ** 2) / (2 * sigma_R ** 2))
    )
    return values


def _get_update_ratio(
    model_key: str,
    freq_dist: Literal["gaussian", "uniform"],
    max_update_ratio: float | None = None,
    avg_update_ratio: float | None = None,
    min_update_ratio: float = 2 ** -5,
) -> np.ndarray:
    if model_key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_key}' not in SPA-Cache registry. "
            f"Supported: {list(_MODEL_REGISTRY.keys())}"
        )
    kwargs   = _MODEL_REGISTRY[model_key]
    n_layers = kwargs["n_layers"]
    layers   = np.arange(n_layers)

    if freq_dist == "uniform":
        ratio = max_update_ratio or avg_update_ratio
        return np.array([ratio] * n_layers)

    if freq_dist == "gaussian":
        if max_update_ratio is not None:
            scale = max_update_ratio / kwargs["freq_peak"]
        elif avg_update_ratio is not None:
            cur = _split_gaussian_frequency(layers, **kwargs).mean()
            scale = avg_update_ratio / cur
        else:
            raise ValueError("Provide max_update_ratio or avg_update_ratio.")
        return _split_gaussian_frequency(
            layers, scale=scale, freq_min=min_update_ratio, **kwargs
        )

    raise NotImplementedError(f"freq_dist must be 'gaussian' or 'uniform', got {freq_dist!r}")


# ---------------------------------------------------------------------------
# SVD proxy helpers (from original spa-cache/proxy.py)
# ---------------------------------------------------------------------------

def _resolve_model_key(model_config) -> str:
    """Best-effort: try common config attributes to find the registry key."""
    # Try architectures list first (LLaDA style)
    for attr in ("architectures", "_name_or_path", "model_type"):
        val = getattr(model_config, attr, None)
        if val is None:
            continue
        candidates = [val] if isinstance(val, str) else list(val)
        for c in candidates:
            # Match against last path component or full string
            for key in _MODEL_REGISTRY:
                if key.lower() in c.lower() or c.lower() in key.lower():
                    return key
    raise ValueError(
        f"Cannot resolve model key from config. "
        f"Set model_key explicitly in the YAML. "
        f"Supported keys: {list(_MODEL_REGISTRY.keys())}"
    )


def _get_model_dims(model_config):
    """Return (hidden_dim, kv_out_dim, n_layers) for supported architectures."""
    arch = getattr(model_config, "architectures", [""])[0]
    if arch == "LLaDAModelLM":
        head_dim   = model_config.d_model // model_config.n_heads
        hidden_dim = model_config.d_model
        kv_out_dim = model_config.n_kv_heads * head_dim
        n_layers   = model_config.n_layers
    elif arch == "DreamModel":
        head_dim   = model_config.hidden_size // model_config.num_attention_heads
        hidden_dim = model_config.hidden_size
        kv_out_dim = model_config.num_key_value_heads * head_dim
        n_layers   = model_config.num_hidden_layers
    else:
        raise NotImplementedError(f"Unsupported architecture: {arch!r}")
    return hidden_dim, kv_out_dim, n_layers


def save_projection_svd_from_model(
    model: nn.Module,
    transformer_blocks_name: str,
    feature_save_dir: Path,
    device: str = "cuda",
) -> None:
    """
    Pre-compute and save truncated SVD of each layer's Value projection matrix.

    Only needs to be run ONCE per model checkpoint.  The resulting files are
    loaded by SpaCache at inference time to build the low-rank proxy projections.

    Each file ``layer_XX.pt`` stores a dict with keys:
        "V" : right singular vectors, shape (kv_out_dim, kv_out_dim)
        "S" : singular values,        shape (kv_out_dim,)

    Parameters
    ----------
    model : nn.Module
        The loaded model (LLaDA or Dream).
    transformer_blocks_name : str
        Dotted path to the ModuleList of transformer blocks, e.g.:
            LLaDA → "model.transformer.blocks"
            Dream → "model.model.layers"
    feature_save_dir : Path
        Directory where layer_XX.pt files will be written (created if missing).
    device : str
        Device used for SVD computation ("cuda" recommended for speed).

    Example
    -------
    >>> from pathlib import Path
    >>> from src.cache.spacache import save_projection_svd_from_model
    >>> save_projection_svd_from_model(
    ...     model=model,
    ...     transformer_blocks_name="model.transformer.blocks",  # LLaDA
    ...     feature_save_dir=Path("./results/svd_cache"),
    ...     device="cuda",
    ... )
    """
    arch = getattr(model.config, "architectures", [""])[0]
    if arch not in ("LLaDAModelLM", "DreamModel"):
        raise NotImplementedError(
            f"save_projection_svd_from_model: unsupported architecture {arch!r}. "
            f"Supported: 'LLaDAModelLM', 'DreamModel'."
        )

    feature_save_dir = Path(feature_save_dir)
    feature_save_dir.mkdir(parents=True, exist_ok=True)

    # Locate the ModuleList by walking named modules.
    blocks: nn.ModuleList | None = None
    for name, module in model.named_modules():
        if name == transformer_blocks_name:
            blocks = module  # type: ignore[assignment]
            break

    if blocks is None:
        raise ValueError(
            f"Module '{transformer_blocks_name}' not found in model. "
            f"Available top-level names: "
            f"{[n for n, _ in model.named_modules() if '.' not in n]}"
        )

    progress = tqdm(blocks, desc="Computing SVD for proxy projections")
    for block in progress:
        # Resolve layer index and Value-projection weight for each architecture.
        if arch == "LLaDAModelLM":
            layer_id = block.layer_id
            Wt = block.v_proj.weight.data          # (kv_out_dim, hidden_dim)
        else:  # DreamModel
            layer_id = block.self_attn.layer_idx
            Wt = block.self_attn.v_proj.weight.data

        file_path = feature_save_dir / f"layer_{layer_id:02d}.pt"
        progress.set_postfix(layer=layer_id)

        Wt = Wt.to(device, dtype=torch.float32)
        # SVD of W_v^T  →  V contains right singular vectors of W_v
        V, S, _ = torch.linalg.svd(Wt.T, full_matrices=False)
        torch.save({"V": V.cpu(), "S": S.cpu()}, file_path)

    logger.info(f"SVD files saved to {feature_save_dir} ({len(blocks)} layers).")


def _build_proxy_projs_from_svd(
    model_config,
    proxy_rank: int,
    svd_cache_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.ModuleDict:
    """
    Load pre-computed SVD files from svd_cache_dir and build proxy projections.
    Each file: layer_XX.pt  containing {"V": ..., "S": ...}.
    Falls back to computing SVD on-the-fly from the live model weights if
    svd_cache_dir does not exist (requires model to be passed; not done here —
    caller must pre-compute and save the SVD files).
    """
    hidden_dim, kv_out_dim, n_layers = _get_model_dims(model_config)

    actual_rank = proxy_rank if proxy_rank > 1 else int(kv_out_dim * proxy_rank)
    actual_rank = max(1, actual_rank)

    logger.info(
        f"Loading SVD proxy projections (rank={actual_rank}) from {svd_cache_dir}"
    )

    proxy_dict: dict[str, nn.Linear] = {}
    for layer_id in tqdm(range(n_layers), desc="Loading SVD proxies"):
        path = Path(svd_cache_dir) / f"layer_{layer_id:02d}.pt"
        data = torch.load(path, map_location=str(device), weights_only=True)
        # V shape from save_projection_svd_from_model: (actual_hidden_dim, kv_out_dim)
        # Read hidden_dim directly from V to avoid model_config attribute mismatches
        # (e.g. Dream's hidden_size != v_proj input dim due to architecture quirks).
        actual_hidden_dim  = data["V"].shape[0]
        actual_rank_capped = min(actual_rank, data["S"].shape[0])
        # weights shape: (actual_rank_capped, actual_hidden_dim)
        weights = (data["V"][:, :actual_rank_capped] * data["S"][None, :actual_rank_capped]).T
        proj = nn.Linear(actual_hidden_dim, actual_rank_capped, bias=False)
        proj.weight.data.copy_(weights)
        proxy_dict[str(layer_id)] = proj

    return nn.ModuleDict(proxy_dict).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# SPA-Cache
# ---------------------------------------------------------------------------

class SPACache(dCache):
    """
    SPA-Cache for the d2cache framework.

    All heavy parameters (proxy_projs, update_ratio) are built inside __init__
    so that Hydra can instantiate this class with only:

        _target_: src.cache.SPACache
        proxy_rank: 128
        freq_dist: gaussian
        max_update_ratio: 0.25
        svd_cache_dir: ./results/svd_cache
        # optional:
        # model_key: LLaDA-8B-Instruct   (auto-detected if omitted)
        # min_update_ratio: 0.03125

    Parameters
    ----------
    model_config : PretrainedConfig
        Passed automatically by the framework (vanilla.py: cache_cls(model.config)).
    proxy_rank : int
        Dimension of the singular proxy subspace (paper default: 128 for LLaDA,
        32 for Dream which has smaller KV dim).
    freq_dist : "gaussian" | "uniform"
        Shape of the per-layer update-ratio curve.
    max_update_ratio : float, optional
        Peak update ratio (fraction of response tokens recomputed at the busiest
        layer).  Either this or avg_update_ratio must be provided.
    avg_update_ratio : float, optional
        Average update ratio across layers.
    min_update_ratio : float
        Floor for the Gaussian curve (default 1/32 ≈ 0.03125).
    svd_cache_dir : str | Path
        Directory containing pre-computed SVD files (layer_00.pt … layer_NN.pt).
        Generate them once with save_projection_svd_from_model() in proxy.py.
    model_key : str, optional
        Key into _MODEL_REGISTRY.  Auto-detected from model_config if omitted.
    """

    def __init__(
        self,
        model_config,
        proxy_rank: int = 128,
        freq_dist: Literal["gaussian", "uniform"] = "gaussian",
        max_update_ratio: float | None = 0.25,
        avg_update_ratio: float | None = None,
        min_update_ratio: float = 2 ** -5,
        svd_cache_dir: str = "./results/svd_cache",
        model_key: str | None = None,
    ):
        super().__init__(model_config)

        # ---- resolve model key & device ----
        if model_key is None:
            model_key = _resolve_model_key(model_config)
        logger.info(f"SpaCache: using model_key='{model_key}'")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype  = torch.bfloat16

        # ---- build update_ratio (numpy array, length = n_layers) ----
        self.update_ratio: np.ndarray = _get_update_ratio(
            model_key      = model_key,
            freq_dist      = freq_dist,
            max_update_ratio = max_update_ratio,
            avg_update_ratio = avg_update_ratio,
            min_update_ratio = min_update_ratio,
        )
        logger.info(
            f"SpaCache update_ratio (mean={self.update_ratio.mean():.3f}):\n"
            f"{self.update_ratio}"
        )

        # ---- build proxy projections from pre-computed SVD files ----
        self.proxy_projs: nn.ModuleDict = _build_proxy_projs_from_svd(
            model_config  = model_config,
            proxy_rank    = proxy_rank,
            svd_cache_dir = Path(svd_cache_dir),
            device        = device,
            dtype         = dtype,
        )

        # ---- cache tensors ----
        self.key_cache:       list[torch.Tensor]        = []
        self.value_cache:     list[torch.Tensor]        = []
        self.attn_cache:      list[torch.Tensor]        = []
        self.ffn_cache:       list[torch.Tensor]        = []
        # indicator_cache[l]: (B_total, L_resp, proxy_rank) or None for layer 0
        self.indicator_cache: list[torch.Tensor | None] = []

        # Absolute token indices selected for update this step, per layer.
        # None  →  full update (first pass, block start, layer 0).
        self._update_indices_layer: dict[int, torch.Tensor | None] = {}

        # Set in on_step_start each step; shape (B_total, T_full)
        self.response_mask:  torch.Tensor | None = None
        self._prompt_length: int = 0

        # True only during the first step of each block (cleared by on_step_end).
        self._block_first_step: bool = True

        # Converted attention mask (float additive), cached from layer 0 each
        # forward pass.  Raw mask from the model is int64; this converts it once.
        self._attention_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
        """
        Snapshot prompt_length and build response_mask each step.

        Dream shift compensation
        ------------------------
        Dream predicts token[i] from hidden_state[i-1].
        Response hidden states useful for the proxy are therefore at:
            LLaDA : positions [P,   T-1]  →  x[:, P:]
            Dream : positions [P-1, T-2]  →  x[:, P-1:T-1]
        We encode this as a boolean mask so _extract_response_features()
        never needs to branch on model type.
        """
        self._prompt_length = frame.prompts.shape[1]
        P = self._prompt_length
        B = frame.prompts.shape[0]
        T = P + frame.generated_tokens.shape[1]

        response_mask = torch.zeros((B, T), dtype=torch.bool,
                                    device=block_mask.device)
        if self.model_config.model_type.lower() == "dream":
            response_mask[:, P - 1 : T - 1] = True   # shifted
        else:
            response_mask[:, P:] = True               # standard

        self.response_mask = response_mask

    def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
        """
        Clear indicator_cache at the start of each new generation block.

        Rationale
        ---------
        Indicators store proxy-projected hidden states from the *previous step*.
        The proxy identifies drifted tokens by comparing current vs. cached
        indicators via cosine similarity.

        When a new block begins the generated-token distribution changes
        discontinuously (previous block's tokens are now fixed real tokens;
        new block is fresh [MASK] tokens).  Stale indicators from the last
        step of the previous block have no meaningful relationship to the
        new block's hidden states — using them would produce arbitrary
        similarity scores and corrupt the token-selection signal.

        Clearing them forces the first step of the new block to re-seed all
        indicators from a full forward pass, giving the proxy a valid
        reference point for subsequent sparse steps.

        KV / attn / FFN caches are NOT cleared because the new block still
        attends to the complete history (prompt + all decoded tokens so far).
        """
        for i in range(len(self.indicator_cache)):
            self.indicator_cache[i] = None  # type: ignore[list-item]
        self._block_first_step = True

    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        """Clear the block-first-step flag after all layers have finished."""
        self._block_first_step = False

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _extract_response_features(
        self,
        x: torch.Tensor,            # (B_active, T, C)
        active_rows: torch.Tensor,  # (B_active,) indices into B_total
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather hidden states at response positions defined by response_mask.

        Returns
        -------
        x_response   : (B_active, L_resp, C)
        resp_abs_idx : (B_active, L_resp)  — absolute positions in [0, T)
        """
        B_active         = x.shape[0]
        resp_mask_active = self.response_mask[active_rows]   # (B_active, T)
        resp_abs_idx     = resp_mask_active.nonzero()[:, 1].view(B_active, -1)
        x_response       = torch.gather(
            x, 1,
            resp_abs_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
        )
        return x_response, resp_abs_idx

    # ------------------------------------------------------------------
    # Attention context manager
    # ------------------------------------------------------------------

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        x: torch.Tensor,
        attn_norm: nn.Module,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        """
        FULL UPDATE PATH  (is_first_pass or _block_first_step)
        --------------------------------------------------------
        Run full Q/K/V projections for all tokens.
        Allocate caches on first pass; re-seed indicators.

        SPARSE UPDATE PATH  (all subsequent steps within a block)
        ----------------------------------------------------------
        layer 0  : always full (proxy disabled at the bottom layer).
        layer > 0: proxy cosine-similarity → top-k drifted tokens →
                   sparse Q/K/V → scatter-update KV & attn caches.
        """
        residual       = x
        x              = attn_norm(x)
        B_active, T, C = x.shape
        active_rows    = self.active_seq_mask.nonzero().squeeze(-1)  # (B_active,)

        # Convert attention_mask dtype exactly once per forward pass (layer 0),
        # mirroring base.py's behaviour.  Raw mask from the model is int64;
        # F.scaled_dot_product_attention requires bool or the same float dtype as q.
        if layer_idx == 0:
            # We need q's dtype — compute a throw-away q just for shape/dtype probe,
            # but that's wasteful.  Instead peek at q_proj weight dtype.
            _q_dtype = next(q_proj.parameters()).dtype
            self._attention_mask = AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=_q_dtype,
                query_length=T,
                key_value_length=T,
            )
        attention_mask = self._attention_mask

        is_first_pass  = len(self.key_cache) <= layer_idx
        do_full_update = is_first_pass or self._block_first_step

        # ══════════════════════════════════════════════════════════════
        # FULL UPDATE
        # ══════════════════════════════════════════════════════════════
        if do_full_update:
            q, k, v = q_proj(x), k_proj(x), v_proj(x)

            if is_first_pass:
                B_total = self.active_seq_mask.shape[0]
                self.key_cache.append(
                    torch.zeros((B_total, T, k.shape[-1]),
                                dtype=x.dtype, device=x.device))
                self.value_cache.append(
                    torch.zeros((B_total, T, v.shape[-1]),
                                dtype=x.dtype, device=x.device))
                self.attn_cache.append(
                    torch.zeros((B_total, T, C),
                                dtype=x.dtype, device=x.device))
                self.indicator_cache.append(None)   # placeholder

            # Seed indicator (layer > 0 only)
            if layer_idx > 0:
                s_proj        = self.proxy_projs[str(layer_idx)]
                x_resp, _     = self._extract_response_features(x, active_rows)
                new_indicator = s_proj(x_resp)   # (B_active, L_resp, r)

                if self.indicator_cache[layer_idx] is None:
                    B_total = self.active_seq_mask.shape[0]
                    self.indicator_cache[layer_idx] = torch.zeros(
                        (B_total, new_indicator.shape[1], new_indicator.shape[2]),
                        dtype=new_indicator.dtype, device=new_indicator.device)

                self.indicator_cache[layer_idx][active_rows] = new_indicator

            self.key_cache[layer_idx][active_rows]   = k
            self.value_cache[layer_idx][active_rows] = v
            self._update_indices_layer[layer_idx]    = None   # full → ffn()

            ctx = AttentionContext(
                q=q, k=k, v=v,
                residual=residual,
                attention_mask=attention_mask,
                q_position_ids=position_ids,
                kv_position_ids=position_ids,
            )
            yield ctx

            assert ctx.o is not None
            self.attn_cache[layer_idx][active_rows] = ctx.o

        # ══════════════════════════════════════════════════════════════
        # SPARSE UPDATE
        # ══════════════════════════════════════════════════════════════
        else:
            if layer_idx == 0:
                # Layer 0: full recompute, no proxy.
                q, k, v = q_proj(x), k_proj(x), v_proj(x)
                self.key_cache[layer_idx][active_rows]   = k
                self.value_cache[layer_idx][active_rows] = v
                self._update_indices_layer[layer_idx]    = None
                sliced_mask, sliced_pos                  = attention_mask, position_ids
                abs_update_indices                       = None

            else:
                # ---- compute proxy similarity ----
                s_proj               = self.proxy_projs[str(layer_idx)]
                x_resp, resp_abs_idx = self._extract_response_features(x, active_rows)
                new_indicator        = s_proj(x_resp)

                # Guard: indicator may still be None if on_block_start cleared it
                # but _block_first_step was reset before this layer ran (shouldn't
                # happen with correct on_step_end placement, but be safe).
                if self.indicator_cache[layer_idx] is None:
                    B_total = self.active_seq_mask.shape[0]
                    self.indicator_cache[layer_idx] = torch.zeros(
                        (B_total, new_indicator.shape[1], new_indicator.shape[2]),
                        dtype=new_indicator.dtype, device=new_indicator.device)
                    self.indicator_cache[layer_idx][active_rows] = new_indicator
                    # Fall back to full update for this layer this step.
                    q, k, v = q_proj(x), k_proj(x), v_proj(x)
                    self.key_cache[layer_idx][active_rows]   = k
                    self.value_cache[layer_idx][active_rows] = v
                    self._update_indices_layer[layer_idx]    = None
                    ctx = AttentionContext(
                        q=q, k=k, v=v, residual=residual,
                        attention_mask=attention_mask,
                        q_position_ids=position_ids,
                        kv_position_ids=position_ids,
                    )
                    yield ctx
                    assert ctx.o is not None
                    self.attn_cache[layer_idx][active_rows] = ctx.o
                    return

                old_indicator = self.indicator_cache[layer_idx][active_rows]
                L_resp        = x_resp.shape[1]
                update_length = max(1, int(L_resp * float(self.update_ratio[layer_idx])))

                # Low cosine similarity = most drifted = must recompute.
                cos_sim  = F.cosine_similarity(new_indicator, old_indicator, dim=-1)
                topk_rel = torch.topk(
                    cos_sim, k=update_length, dim=-1,
                    largest=False, sorted=False,
                ).indices                                    # (B_active, L_update)

                # Relative → absolute positions.
                abs_update_indices = torch.gather(resp_abs_idx, 1, topk_rel)

                # Update stored indicators.
                self.indicator_cache[layer_idx][active_rows] = new_indicator

                # Sparse Q/K/V.
                expand_idx = abs_update_indices.unsqueeze(-1).expand(-1, -1, C)
                selected_x = torch.gather(x, 1, expand_idx)
                q          = q_proj(selected_x)
                k_new      = k_proj(selected_x)
                v_new      = v_proj(selected_x)

                # Scatter K/V into cache.
                self.key_cache[layer_idx][
                    active_rows.unsqueeze(1), abs_update_indices] = k_new
                self.value_cache[layer_idx][
                    active_rows.unsqueeze(1), abs_update_indices] = v_new

                self._update_indices_layer[layer_idx] = abs_update_indices

                # Slice attention_mask to match sparse query length.
                if attention_mask is not None and attention_mask.dim() == 4:
                    idx_4d = abs_update_indices.view(B_active, 1, -1, 1).expand(
                        -1, attention_mask.shape[1], -1, attention_mask.shape[-1])
                    sliced_mask = torch.gather(attention_mask, 2, idx_4d)
                else:
                    sliced_mask = attention_mask

                sliced_pos = (
                    torch.gather(position_ids, 1, abs_update_indices)
                    if position_ids is not None else None
                )

            # ---- sparse Q attends to full cached K/V ----
            ctx = AttentionContext(
                q=q,
                k=self.key_cache[layer_idx][active_rows],
                v=self.value_cache[layer_idx][active_rows],
                residual=residual,
                attention_mask=sliced_mask,
                q_position_ids=sliced_pos,
                kv_position_ids=position_ids,
            )
            yield ctx

            assert ctx.o is not None
            upd_idx = self._update_indices_layer[layer_idx]
            if upd_idx is None:
                self.attn_cache[layer_idx][active_rows] = ctx.o
            else:
                self.attn_cache[layer_idx][
                    active_rows.unsqueeze(1), upd_idx] = ctx.o

            # Return complete cached sequence for correct downstream residuals.
            ctx.o = self.attn_cache[layer_idx][active_rows]

    # ------------------------------------------------------------------
    # FFN context manager
    # ------------------------------------------------------------------

    @contextmanager
    def ffn(self, layer_idx: int, x: torch.Tensor):
        """
        Mirrors the update-index decision from attention().
        Full update when _update_indices_layer[layer_idx] is None,
        sparse update otherwise.
        """
        B_active, T, C = x.shape
        residual       = x
        is_first_pass  = len(self.ffn_cache) <= layer_idx
        active_rows    = self.active_seq_mask.nonzero().squeeze(-1)
        abs_upd        = self._update_indices_layer.get(layer_idx)

        if is_first_pass:
            B_total = self.active_seq_mask.shape[0]
            self.ffn_cache.append(
                torch.zeros((B_total, T, C), dtype=x.dtype, device=x.device))

        if abs_upd is None:
            # Full FFN.
            ctx = FFNContext(x=x, residual=residual)
            yield ctx
            assert ctx.ffn_out is not None
            self.ffn_cache[layer_idx][active_rows] = ctx.ffn_out
        else:
            # Sparse FFN: only selected tokens.
            expand_idx = abs_upd.unsqueeze(-1).expand(-1, -1, C)
            selected_x = torch.gather(x, 1, expand_idx)
            ctx = FFNContext(x=selected_x, residual=residual)
            yield ctx
            assert ctx.ffn_out is not None
            self.ffn_cache[layer_idx][
                active_rows.unsqueeze(1), abs_upd] = ctx.ffn_out

        # Always return complete cached output for correct residual connections.
        ctx.ffn_out = self.ffn_cache[layer_idx][active_rows]
