import os
import torch
import torch.nn.functional as F
import torch.distributions as dists
from typing import Type

from src.cache import dCache
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import (
    check_can_generate,
    prepare_logits_for_generation,
    sample_tokens,
)
from src.utils import Timer, certainty_density, register


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.

    Args:
        mask_index: A boolean tensor of shape (B, L) indicating the positions of mask
        steps: The number of steps in a block to sample.

    Returns:
        A tensor of shape (B, steps) indicating the number of tokens to be transferred at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def init_step_metrics() -> dict[str, float]:
    return {
        "step_count": 0.0,
        "decode_step_count": 0.0,
        "refresh_step_count": 0.0,
        "prefill_step_count": 0.0,
        "generation_step_count": 0.0,
        "generated_token_count": 0.0,
        "prefill_generated_token_count": 0.0,
        "generation_generated_token_count": 0.0,
        "init_time_ms": 0.0,
        "step_time_ms_total": 0.0,
        "decode_step_time_ms_total": 0.0,
        "refresh_step_time_ms_total": 0.0,
        "prefill_step_time_ms_total": 0.0,
        "generation_step_time_ms_total": 0.0,
    }


def classify_cache_step(
    cache: dCache | None,
    *,
    pre_step_is_first_step: bool = False,
    pre_step_missing_q_mask: bool = False,
) -> str:
    if cache is None:
        return "refresh"

    cache_name = type(cache).__name__
    if cache_name == "PrefixCache":
        return "refresh" if pre_step_missing_q_mask else "decode"
    if cache_name == "dKVCache":
        return (
            "refresh"
            if pre_step_is_first_step or pre_step_missing_q_mask
            else "decode"
        )
    if cache_name == "dLLMCache":
        return (
            "refresh"
            if bool(
                getattr(cache, "refresh_prompt", False)
                or getattr(cache, "refresh_response", False)
            )
            else "decode"
        )
    return "decode"


def record_step_metrics(
    metrics: dict[str, float],
    step_type: str,
    elapsed_time_ms: float,
    transferred_token_count: float = 0.0,
) -> None:
    is_prefill_step = metrics["step_count"] == 0.0
    metrics["step_count"] += 1.0
    metrics["step_time_ms_total"] += elapsed_time_ms
    metrics["generated_token_count"] += transferred_token_count
    bucket = "refresh" if step_type == "refresh" else "decode"
    metrics[f"{bucket}_step_count"] += 1.0
    metrics[f"{bucket}_step_time_ms_total"] += elapsed_time_ms
    metrics[f"{bucket}_generated_token_count"] = (
        metrics.get(f"{bucket}_generated_token_count", 0.0) + transferred_token_count
    )

    if is_prefill_step:
        metrics["init_time_ms"] = elapsed_time_ms
        metrics["prefill_step_count"] += 1.0
        metrics["prefill_step_time_ms_total"] += elapsed_time_ms
        metrics["prefill_generated_token_count"] += transferred_token_count
    else:
        metrics["generation_step_count"] += 1.0
        metrics["generation_step_time_ms_total"] += elapsed_time_ms
        metrics["generation_generated_token_count"] += transferred_token_count


def count_transferred_tokens(delta: FrameDelta) -> float:
    transferred_tokens = delta.transferred_tokens
    if isinstance(transferred_tokens, tuple):
        return float(sum(tokens.numel() for tokens in transferred_tokens))
    return float(transferred_tokens.numel())


def finalize_step_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    step_count = metrics["step_count"]
    decode_step_count = metrics["decode_step_count"]
    refresh_step_count = metrics["refresh_step_count"]
    prefill_step_count = metrics["prefill_step_count"]
    generation_step_count = metrics["generation_step_count"]

    return {
        "avg_step_count": step_count,
        "avg_q_part_step_count": decode_step_count,
        "avg_q_full_step_count": refresh_step_count,
        "avg_generation_step_count": generation_step_count,
        "avg_generated_token_count": metrics["generated_token_count"],
        "avg_q_full_generated_token_count": metrics.get(
            "refresh_generated_token_count", 0.0
        ),
        "avg_q_part_generated_token_count": metrics.get(
            "decode_generated_token_count", 0.0
        ),
        "avg_generation_generated_token_count": metrics[
            "generation_generated_token_count"
        ],
        "generation_tps": (
            metrics["generation_generated_token_count"] / generation_step_count
            if generation_step_count > 0
            else None
        ),
        "avg_step_time_ms": (
            metrics["step_time_ms_total"] / step_count if step_count > 0 else None
        ),
        "avg_q_part_step_time_ms": (
            metrics["decode_step_time_ms_total"] / decode_step_count
            if decode_step_count > 0
            else None
        ),
        "avg_q_full_step_time_ms": (
            metrics["refresh_step_time_ms_total"] / refresh_step_count
            if refresh_step_count > 0
            else None
        ),
        "avg_generation_step_time_ms": (
            metrics["generation_step_time_ms_total"] / generation_step_count
            if generation_step_count > 0
            else None
        ),
        "avg_q_full_step_fraction": (
            refresh_step_count / step_count if step_count > 0 else None
        ),
        "init_time_ms": metrics["init_time_ms"] if prefill_step_count > 0 else None,
    }


def confidence_unmasking(
    scores: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    min_transfer_tokens: torch.Tensor,
    max_transfer_tokens: torch.Tensor | None = None,
    # parallel decoding
    threshold: float | torch.Tensor | None = None,
    factor: float | None = None,
    # EB sampler
    gamma: float | None = None,
    p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Select tokens to be fixed based on token probs, i.e., confidence.
    It consists of parallel decoding and low-confidence remasking.
    Args:
        token_probs: A tensor of shape [B, gen_length] containing the probabilities of each token.
        transfer_index_mask: A boolean tensor of shape [B, gen_length] indicating which tokens can be transferred.
        min_transfer_tokens: A tensor of shape [B,] indicating the minimum number of tokens to be transferred at each step.
        max_transfer_tokens: Optional cap of shape [B,] for tokens transferred when threshold/factor select too many.
        threshold: A threshold for remasking. If provided, all tokens whose confidence is above this threshold will be kept.
        factor: factor-based parallel decoding factor, see https://arxiv.org/pdf/2505.22618.
        gamma: threshold of upper bound of joint dependence error, see https://arxiv.org/pdf/2505.24857.
        p: A tensor of shape [B, gen_length, vocab_size] containing the probabilities of each token. Must be provided if using EB sampler.
    """

    if (threshold is not None) + (factor is not None) + (gamma is not None) > 1:
        raise ValueError(
            "Only one of `threshold`, `factor`, or `gamma` should be provided."
        )

    batch_size, _ = scores.shape

    if min_transfer_tokens.numel() != batch_size:
        raise ValueError(
            "`min_transfer_tokens` must have shape (batch_size,) to match scores."
        )

    if max_transfer_tokens is not None:
        if max_transfer_tokens.numel() != batch_size:
            raise ValueError(
                "`max_transfer_tokens` must have shape (batch_size,) to match scores."
            )
    else:
        max_transfer_tokens = torch.sum(transfer_index_mask, dim=-1)
    num_transfer_tokens = torch.minimum(min_transfer_tokens, max_transfer_tokens)

    confidence = torch.where(transfer_index_mask, scores, -torch.inf)
    transfer_index = [torch.tensor([]) for _ in range(batch_size)]
    if threshold is not None or factor is not None:
        if threshold is not None:
            col_indices = torch.nonzero(confidence >= threshold, as_tuple=False)[:, 1]
            counts = torch.sum(confidence >= threshold, dim=-1).cpu().tolist()
            transfer_index = list(torch.split(col_indices, counts))
            for i, t in enumerate(transfer_index):
                if t.numel() > max_transfer_tokens[i]:
                    transfer_index[i] = torch.tensor([])
                    num_transfer_tokens[i] = max_transfer_tokens[i]
        elif factor is not None:
            num_unmasked_tokens = torch.sum(transfer_index_mask, dim=-1, keepdim=True)
            for i in range(batch_size):
                sorted_conf, _ = torch.sort(
                    confidence[i][transfer_index_mask[i]],
                    dim=-1,
                    descending=True,
                )
                for n in range(1, num_unmasked_tokens[i] + 1):
                    if (n + 1) * (1 - sorted_conf[n - 1]) >= factor:
                        break
                transfer_index[i] = torch.topk(confidence[i], min(n - 1, int(max_transfer_tokens[i].item())), dim=-1).indices  # type: ignore
    elif gamma is not None:
        if p is None:
            raise ValueError(
                "Probabilities of all tokens `p` must be provided for EB sampler."
            )
        _, ids = torch.sort(confidence, dim=-1, descending=True)
        entropy = torch.gather(
            dists.Categorical(probs=p.float()).entropy(), dim=-1, index=ids
        )
        acc_entropy = torch.cumsum(entropy, dim=1)
        cummax_entropy = torch.cummax(entropy, dim=0).values
        num_transfer_tokens = (acc_entropy - cummax_entropy <= gamma).sum(dim=1)

    num_transfer_tokens = torch.clamp(
        num_transfer_tokens,
        min=min_transfer_tokens,
        max=max_transfer_tokens,
    )

    if fallback_indices := [
        i for i, t in enumerate(transfer_index) if t.numel() < num_transfer_tokens[i]
    ]:
        confidence_subset = confidence[fallback_indices]
        topk_transfer_index = [
            torch.topk(
                confidence_subset[i],
                int(
                    torch.min(
                        transfer_index_mask[fallback_indices[i]].sum(),
                        num_transfer_tokens[fallback_indices[i]],
                    )
                ),
                dim=-1,
            ).indices
            for i in range(confidence_subset.size(0))
        ]
        source_iter = iter(topk_transfer_index)
        transfer_index = [
            next(source_iter) if i in fallback_indices else t
            for i, t in enumerate(transfer_index)
        ]

    return tuple(transfer_index)


@torch.no_grad()
def generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    num_transfer_tokens: torch.Tensor | int,
    attention_mask: torch.Tensor | None = None,
    past_key_values: dCache | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    mask_token_id: int = None,  # type: ignore
    eot_token_id: int | None = None,
    sigma: float | None = None,
    stop_until_eot: bool = False,
    gamma: float | None = None,
    debias: bool = False,
    clip_alpha: float | None = None,
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape
    device = block_mask.device

    if isinstance(num_transfer_tokens, torch.Tensor):
        if num_transfer_tokens.numel() != batch_size or num_transfer_tokens.dim() != 1:
            raise ValueError(
                f"`num_transfer_tokens` must be a tensor of shape ({batch_size},) or a single integer, "
                f"but got shape of {num_transfer_tokens.shape}."
            )
    else:
        num_transfer_tokens = torch.full(
            (batch_size,), num_transfer_tokens, device=device, dtype=torch.long
        )

    can_generate = check_can_generate(
        frame,
        eligible_mask=block_mask,
        num_transfer_tokens=num_transfer_tokens,
        stop_until_eot=stop_until_eot,
        mask_token_id=mask_token_id,
        eot_token_id=eot_token_id,
    )
    if not torch.any(can_generate):
        return None

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()

    if past_key_values is not None:
        past_key_values.active_seq_mask = can_generate

    x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[can_generate]
    attention_mask = (
        attention_mask[can_generate] if attention_mask is not None else None
    )
    num_transfer_tokens = num_transfer_tokens[can_generate]
    outputs = model(
        x,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        past_key_values=past_key_values,
        use_cache=past_key_values is not None,
    )

    logits = prepare_logits_for_generation(model, outputs.logits)
    if past_key_values is not None and past_key_values.active_q_mask is not None:
        if model.config.model_type.lower() == "dream":
            valid_mask = past_key_values.active_q_mask[:, prompt_length - 1 : -1]
        else:
            valid_mask = past_key_values.active_q_mask[:, prompt_length:]
        transfer_index_mask[can_generate].logical_and_(valid_mask)
    logits = logits[:, prompt_length:]

    hidden_states = (
        tuple((i, hs) for i, hs in enumerate(outputs.hidden_states))
        if output_hidden_states
        else None
    )

    confidence, x0, p = sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        debias=debias,
        clip_alpha=clip_alpha,
        alg=alg,
    )
    scores = confidence = torch.where(
        transfer_index_mask[can_generate], confidence, -torch.inf
    )
    if sigma is not None and sigma > 0:
        scores = confidence * certainty_density(
            ~remaining_mask[can_generate], sigma=sigma
        )

    transfer_index = confidence_unmasking(
        scores=scores,
        transfer_index_mask=(transfer_index_mask & block_mask)[can_generate],
        min_transfer_tokens=num_transfer_tokens,
        threshold=threshold,
        factor=factor,
        gamma=gamma,
        p=p,
    )
    transfer_index_iter = iter(transfer_index)
    transfer_index = tuple(
        (
            next(transfer_index_iter)
            if is_not_finished
            else torch.tensor([], dtype=torch.long, device=device)
        )
        for is_not_finished in can_generate
    )

    return FrameDelta(
        transfer_index=transfer_index,
        decoded_tokens=torch.where(
            transfer_index_mask[can_generate], x0, INVALID_TOKEN_ID
        ),
        confidence=confidence,
        probs=(
            torch.where(transfer_index_mask[can_generate].unsqueeze(-1), p, -torch.inf)
            if output_probs
            else None
        ),
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
    )


@register.gen_strategy("vanilla")
def vanilla_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    steps: int = 128,
    block_length: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    sigma: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eot_token_id: int | None = None,
    stop_until_eot: bool = False,
    gamma: float | None = None,
    debias: bool = False,
    clip_alpha: float | None = None,
    threshold: float | None = None,
    factor: float | None = None,
    sparsed: bool = False,
    sparsed_select: float = 0.5,
    sparsed_skip: float = 0.2,
    sparsed_block_size: int = 32,
    sparsed_param: dict | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
    cache_reloading_step: int | None = None,
) -> DecodeRecord:
    """
    Vanilla generation for diffusion large language models.
    """

    use_sparsed = sparsed
    if use_sparsed:
        if cache_cls is not None:
            raise ValueError(
                "SparseD under generation=vanilla does not support cache. "
                "Please remove cache=prefix/dkvcache when generation.sparsed=true."
            )
        from src.generation.sparsed_vanilla import sparsed_vanilla_generate

        merged_sparsed_param = dict(sparsed_param) if sparsed_param is not None else {}
        merged_sparsed_param.setdefault("select", sparsed_select)
        merged_sparsed_param.setdefault("skip", sparsed_skip)
        merged_sparsed_param.setdefault("block_size", sparsed_block_size)

        return sparsed_vanilla_generate(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            alg=alg,
            steps=steps,
            block_length=block_length,
            gen_length=gen_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sigma=sigma,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            eot_token_id=eot_token_id,
            stop_until_eot=stop_until_eot,
            gamma=gamma,
            debias=debias,
            clip_alpha=clip_alpha,
            threshold=threshold,
            factor=factor,
            sparsed_param=merged_sparsed_param,
            output_hidden_states=output_hidden_states,
            output_probs=output_probs,
        )

    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore
    if stop_until_eot:
        if eot_token_id is None and os.environ.get("EOT_TOKEN_ID", None) is None:
            raise ValueError(
                "eot_token_id must be provided either as an argument or an environment variable if stop_until_eot is set to True."
            )
        eot_token_id = eot_token_id or int(os.environ.get("EOT_TOKEN_ID"))  # type: ignore

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()

    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(
            model.device
        )

    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame

    deltas = []
    step_metrics = init_step_metrics()

    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (input_ids.size(0), gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[
            :,
            block_idx * block_length : (block_idx + 1) * block_length,
        ] = True

        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps)
        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)
        block_deltas = []
        for i in range(steps):
            pre_step_is_first_step = False
            pre_step_missing_q_mask = False
            if cache is not None:
                if (
                    cache_reloading_step is not None
                    and i > 0
                    and i % cache_reloading_step == 0
                ):
                    cache.active_q_mask = None
                pre_step_is_first_step = bool(getattr(cache, "_is_first_step", False))
                pre_step_missing_q_mask = getattr(cache, "active_q_mask", None) is None
                cache.on_step_start(block_mask, frame)
            step_type = classify_cache_step(
                cache,
                pre_step_is_first_step=pre_step_is_first_step,
                pre_step_missing_q_mask=pre_step_missing_q_mask,
            )
            with Timer() as step_timer:
                delta = generate_step(
                    model=model,
                    frame=frame,
                    block_mask=block_mask,
                    num_transfer_tokens=num_transfer_tokens[:, i],
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    alg=alg,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    sigma=sigma,
                    gamma=gamma,
                    mask_token_id=mask_token_id,
                    eot_token_id=eot_token_id,
                    stop_until_eot=stop_until_eot,
                    debias=debias,
                    clip_alpha=clip_alpha,
                    threshold=threshold,
                    factor=factor,
                    output_hidden_states=output_hidden_states,
                    output_probs=output_probs,
                )
            if delta is None:
                break
            record_step_metrics(
                step_metrics,
                step_type,
                step_timer.elapsed_time_ms,
                count_transferred_tokens(delta),
            )
            if cache is not None:
                cache.on_step_end(block_mask, frame, delta)

            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

        if cache is not None:
            cache.on_block_end(block_mask, start_frame, block_deltas)

        deltas.extend(block_deltas)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        metrics=finalize_step_metrics(step_metrics),
        block_length=block_length,
    )
