import os
import torch
import torch.nn.functional as F

from loguru import logger

from src.frame import Frame, DecodeRecord, FrameDelta, INVALID_TOKEN_ID
from src.generation.utils import check_can_generate, prepare_logits_for_generation
from src.generation.vanilla import get_num_transfer_tokens
from src.utils import register


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _build_uniform_suffix_indices(
    start_idx: int, seq_len: int, keep_ratio: float, device: torch.device
) -> torch.Tensor:
    """
    Build a uniform sparse index for suffix tokens [start_idx, seq_len).
    This keeps the full prefix and sparsifies only the suffix for lower compute.
    """
    if start_idx >= seq_len:
        return torch.empty((0,), dtype=torch.long, device=device)

    suffix_len = seq_len - start_idx
    keep_num = max(1, int(suffix_len * keep_ratio))
    keep_num = min(keep_num, suffix_len)
    if keep_num == suffix_len:
        return torch.arange(start_idx, seq_len, device=device, dtype=torch.long)

    idx = torch.linspace(
        0, suffix_len - 1, steps=keep_num, device=device, dtype=torch.float32
    ).long()
    return idx + start_idx


def _get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_indices: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        confidence = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        confidence = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_indices, x0, x)
    confidence = torch.where(mask_indices, confidence, -torch.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    valid_counts = mask_indices.sum(dim=1)
    for i in range(confidence.shape[0]):
        k = int(num_transfer_tokens[i].item())
        k = min(k, int(valid_counts[i].item()))
        if k <= 0:
            continue
        idx = torch.topk(confidence[i], k=k).indices
        transfer_index[i, idx] = True
    transfer_index &= mask_indices
    return x0, transfer_index, confidence


@torch.no_grad()
@register.gen_strategy("sparse")
def sparse_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    steps: int = 32,
    block_length: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    keep_ratio: float = 0.5,
    kernel_size: int = 3,
    early_termination: bool = True,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eot_token_id: int | None = None,
    output_hidden_states: bool = False,
    cache_cls=None,
) -> DecodeRecord:
    if cache_cls is not None:
        logger.warning(
            "Sparse generation does not support external cache_cls; ignoring cache_cls."
        )
    # if model.config.model_type.lower() != "llada":
    #     raise NotImplementedError("Sparse generation is only implemented for LLaDA.")
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1].")

    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID") is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ["MASK_TOKEN_ID"])

    if early_termination:
        if eot_token_id is None and os.environ.get("EOT_TOKEN_ID") is None:
            raise ValueError(
                "eot_token_id must be provided either as an argument or environment variable if early_termination is True."
            )
        eot_token_id = eot_token_id or int(os.environ["EOT_TOKEN_ID"])

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by number of blocks")
    steps = steps // num_blocks

    initial_frame = Frame.create_initial_frame(
        input_ids, gen_length=gen_length, mask_token_id=mask_token_id
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()
    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(model.device)

    frame = initial_frame
    deltas = []
    batch_size = input_ids.size(0)
    prompt_length = input_ids.size(1)
    seq_len = prompt_length + gen_length
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=model.device)

    for block_idx in range(num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        block_start_abs = prompt_length + block_start
        block_end_abs = prompt_length + block_end

        block_mask = torch.zeros(
            (batch_size, gen_length), dtype=torch.bool, device=model.device
        )
        block_mask[:, block_start:block_end] = True
        block_mask[finished] = False

        block_mask_indices = frame.generated_tokens[:, block_start:block_end].eq(mask_token_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_indices, steps)
        num_transfer_tokens[finished] = 0

        suffix_len = max(seq_len - block_end_abs, 0)
        suffix_indices = (
            _build_uniform_suffix_indices(
                start_idx=block_end_abs,
                seq_len=seq_len,
                keep_ratio=keep_ratio,
                device=model.device,
            )
            if suffix_len > 0
            else torch.empty((0,), dtype=torch.long, device=model.device)
        )

        for step_idx in range(steps):
            eligible_mask = block_mask.clone()
            eligible_mask[finished] = False
            can_generate = check_can_generate(
                frame,
                eligible_mask=eligible_mask,
                num_transfer_tokens=num_transfer_tokens[:, step_idx],
                mask_token_id=mask_token_id,
            )
            if not torch.any(can_generate):
                break

            x_full = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[can_generate]
            active_batch = x_full.size(0)

            # Keep full prefix and sparse suffix to reduce compute.
            if suffix_len > 0 and suffix_indices.numel() > 0:
                prefix_indices = (
                    torch.arange(block_end_abs, device=model.device)
                    .unsqueeze(0)
                    .expand(active_batch, -1)
                )
                q_idx = torch.cat(
                    [prefix_indices, suffix_indices.unsqueeze(0).expand(active_batch, -1)],
                    dim=-1,
                )
            else:
                q_idx = torch.arange(seq_len, device=model.device).unsqueeze(0).expand(active_batch, -1)

            x_pruned = x_full.gather(1, q_idx)

            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask[can_generate].gather(1, q_idx)

            outputs = model(
                x_pruned,
                attention_mask=attn_mask,
                position_ids=q_idx,
                output_hidden_states=output_hidden_states,
            )
            logits = prepare_logits_for_generation(model, outputs.logits)

            mask_indices = x_pruned.eq(mask_token_id)
            mask_indices[:, :prompt_length] = False
            mask_indices[:, block_end_abs:] = False

            x0, transfer_mask, confidence = _get_transfer_index(
                logits=logits,
                temperature=temperature,
                remasking=remasking,
                mask_indices=mask_indices,
                x=x_pruned,
                num_transfer_tokens=num_transfer_tokens[can_generate, step_idx],
            )

            x_updated = torch.where(transfer_mask, x0, x_pruned)
            x_full_updated = x_full.scatter(1, q_idx, x_updated)

            conf_dtype = (
                frame.confidence.dtype
                if frame.confidence is not None
                else torch.float32
            )
            # No suffix score bookkeeping in this minimal sparse version.

            decoded_tokens = torch.full(
                (active_batch, gen_length),
                INVALID_TOKEN_ID,
                dtype=torch.long,
                device=model.device,
            )
            conf_tensor = torch.full(
                (active_batch, gen_length),
                -torch.inf,
                dtype=conf_dtype,
                device=model.device,
            )
            block_slice = slice(block_start, block_end)
            decoded_tokens[:, block_slice] = x_full_updated[:, block_start_abs:block_end_abs]
            # Fill only block confidence to avoid full-seq scatter.
            conf_block = torch.full(
                (active_batch, block_length),
                -torch.inf,
                dtype=conf_dtype,
                device=model.device,
            )
            for i in range(active_batch):
                q = q_idx[i]
                m = (q >= block_start_abs) & (q < block_end_abs)
                if torch.any(m):
                    rel = (q[m] - block_start_abs).long()
                    conf_block[i, rel] = confidence[i, m].to(conf_dtype)
            conf_tensor[:, block_slice] = conf_block

            transfer_index_active = []
            for i in range(active_batch):
                local_idx = torch.nonzero(
                    transfer_mask[i, block_start_abs:block_end_abs], as_tuple=False
                ).squeeze(1)
                transfer_index_active.append(local_idx + block_start)

            transfer_iter = iter(transfer_index_active)
            transfer_index = tuple(
                (
                    next(transfer_iter)
                    if active
                    else torch.tensor([], dtype=torch.long, device=model.device)
                )
                for active in can_generate
            )

            delta = FrameDelta(
                transfer_index=transfer_index,
                decoded_tokens=decoded_tokens,
                confidence=conf_tensor,
            )
            deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

            if early_termination:
                block_done = frame.generated_tokens[:, block_slice].ne(mask_token_id).all(dim=-1)
                eos_in_block = frame.generated_tokens[:, block_slice].eq(eot_token_id).any(dim=-1)
                finished |= block_done & eos_in_block
                if finished.all():
                    break

        if finished.all():
            break

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
