import os
import torch
import torch.nn.functional as F

from loguru import logger

from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord
from src.generation.utils import (
    check_can_generate,
    prepare_logits_for_generation,
    sample_tokens,
)
from src.utils import register


def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    attention_mask = attention_mask.long()
    return (torch.cumsum(attention_mask, dim=-1) - 1).clamp_min(0) * attention_mask


def _suppress_mask_token(logits: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    if 0 <= mask_token_id < logits.size(-1):
        logits = logits.clone()
        logits[..., mask_token_id] = -torch.inf
    return logits


def _build_lopa_attention_mask(
    base_attention_mask: torch.Tensor,
    *,
    base_length: int,
    block_start_abs: int,
    block_end_abs: int,
    block_length: int,
    num_branches: int,
) -> torch.Tensor:
    full_length = base_length + num_branches * block_length
    attention_mask = torch.zeros(
        (1, 1, full_length, full_length),
        dtype=torch.bool,
        device=base_attention_mask.device,
    )
    attention_mask[:, :, :, :base_length] = base_attention_mask.bool().view(
        1, 1, 1, base_length
    )

    for branch_idx in range(num_branches):
        start = base_length + branch_idx * block_length
        end = start + block_length
        attention_mask[:, :, start:end, start:end] = True
        attention_mask[:, :, start:end, block_start_abs:block_end_abs] = False

    return attention_mask


@torch.no_grad()
def _lopa_generate_step(
    model,
    *,
    prompt: torch.Tensor,
    generated_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    block_start: int,
    block_length: int,
    alg: str,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    threshold: float | None,
    k: int,
    mask_token_id: int,
    output_probs: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    prompt_length = prompt.numel()
    gen_length = generated_tokens.numel()
    device = generated_tokens.device
    block_end = block_start + block_length
    block_start_abs = prompt_length + block_start
    block_end_abs = prompt_length + block_end

    x = torch.cat([prompt, generated_tokens], dim=0).unsqueeze(0)
    base_attention_mask = attention_mask.unsqueeze(0)

    outputs = model(
        x,
        attention_mask=base_attention_mask,
        use_cache=False,
    )
    logits = _suppress_mask_token(
        prepare_logits_for_generation(model, outputs.logits),
        mask_token_id,
    )
    confidence_all, x0_all, probs_all = sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
    )

    confidence = confidence_all[:, prompt_length:].squeeze(0)
    x0 = x0_all[:, prompt_length:].squeeze(0)
    probs = probs_all[:, prompt_length:].squeeze(0)

    current_block = generated_tokens[block_start:block_end].clone()
    remaining_mask_block = current_block.eq(mask_token_id)
    confidence_block = torch.where(
        remaining_mask_block, confidence[block_start:block_end], -torch.inf
    )

    transfer_mask_block = torch.zeros_like(remaining_mask_block)
    if threshold is not None:
        transfer_mask_block = remaining_mask_block & confidence_block.ge(threshold)
    if not transfer_mask_block.any():
        transfer_mask_block[torch.argmax(confidence_block)] = True

    anchor_block = current_block.clone()
    anchor_block[transfer_mask_block] = x0[block_start:block_end][transfer_mask_block]

    final_block = anchor_block
    remaining_after_anchor = final_block.eq(mask_token_id)
    if remaining_after_anchor.any():
        actual_k = min(k, int(remaining_after_anchor.sum().item()))
        branch_scores = torch.where(remaining_after_anchor, confidence_block, -torch.inf)
        branch_rel_indices = torch.topk(branch_scores, k=actual_k).indices

        candidate_blocks = final_block.unsqueeze(0).repeat(actual_k, 1)
        candidate_blocks[
            torch.arange(actual_k, device=device),
            branch_rel_indices,
        ] = x0[block_start:block_end][branch_rel_indices]

        x_after_anchor = x.clone()
        x_after_anchor[:, block_start_abs:block_end_abs] = anchor_block.unsqueeze(0)
        x_ahead = torch.cat([x_after_anchor, candidate_blocks.reshape(1, -1)], dim=1)

        base_position_ids = _build_position_ids(attention_mask)
        branch_position_ids = base_position_ids[block_start_abs:block_end_abs]
        position_ids = torch.cat(
            [base_position_ids, branch_position_ids.repeat(actual_k)],
            dim=0,
        ).unsqueeze(0)
        attention_mask_ahead = _build_lopa_attention_mask(
            attention_mask,
            base_length=x.size(1),
            block_start_abs=block_start_abs,
            block_end_abs=block_end_abs,
            block_length=block_length,
            num_branches=actual_k,
        )

        outputs_ahead = model(
            x_ahead,
            attention_mask=attention_mask_ahead,
            position_ids=position_ids,
            use_cache=False,
        )
        logits_ahead = _suppress_mask_token(
            prepare_logits_for_generation(model, outputs_ahead.logits),
            mask_token_id,
        )
        confidence_ahead, _, _ = sample_tokens(
            logits_ahead,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
        )
        confidence_ahead = confidence_ahead.squeeze(0)

        mask_index_ahead = x_ahead.squeeze(0).eq(mask_token_id)
        mask_index_ahead[block_end_abs : x.size(1)] = False
        confidence_ahead = torch.where(
            mask_index_ahead,
            confidence_ahead,
            torch.full_like(confidence_ahead, -torch.inf),
        )

        anchor_mask = mask_index_ahead[block_start_abs:block_end_abs]
        best_score = (
            confidence_ahead[block_start_abs:block_end_abs][anchor_mask].mean().item()
            if anchor_mask.any()
            else 1.0
        )
        winner = None

        for branch_idx in range(actual_k):
            start = x.size(1) + branch_idx * block_length
            end = start + block_length
            branch_mask = mask_index_ahead[start:end]
            branch_score = (
                confidence_ahead[start:end][branch_mask].mean().item()
                if branch_mask.any()
                else 1.0
            )
            if branch_score >= best_score:
                best_score = branch_score
                winner = branch_idx

        if winner is not None:
            final_block = candidate_blocks[winner]

    newly_decoded_mask = generated_tokens[block_start:block_end].eq(mask_token_id) & (
        final_block != mask_token_id
    )
    transfer_index = torch.arange(
        block_start,
        block_end,
        device=device,
        dtype=torch.long,
    )[newly_decoded_mask]

    decoded_tokens = torch.full(
        (gen_length,),
        INVALID_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    step_confidence = torch.full(
        (gen_length,),
        -torch.inf,
        dtype=confidence.dtype,
        device=device,
    )
    step_probs = None
    if output_probs:
        step_probs = torch.full(
            (gen_length, probs.size(-1)),
            -torch.inf,
            dtype=probs.dtype,
            device=device,
        )

    decoded_tokens[transfer_index] = final_block[newly_decoded_mask]
    step_confidence[transfer_index] = confidence[transfer_index]
    if step_probs is not None:
        step_probs[transfer_index] = probs[transfer_index]

    return transfer_index, decoded_tokens, step_confidence, step_probs


@register.gen_strategy("lopa")
@torch.no_grad()
def lopa_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    steps: int | None = None,
    block_length: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    sigma: float | None = None,
    threshold: float | None = 0.9,
    factor: float | None = None,
    k: int = 5,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eot_token_id: int | None = None,
    stop_until_eot: bool = False,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls=None,
) -> DecodeRecord:
    """
    LoPA decoding strategy.
    Each step first commits anchor tokens whose confidence exceeds `threshold`,
    then evaluates up to `k` lookahead branches and keeps the best branch if it
    improves the masked-token confidence inside the current block.
    """
    _ = steps

    if cache_cls is not None:
        logger.warning(
            "LoPA generation does not support cache; ignoring cache_cls.",
            once=True,
            rank_zero_only=True,
        )
    if sigma is not None:
        logger.warning(
            "LoPA generation does not use sigma-based certainty prior; ignoring sigma.",
            once=True,
            rank_zero_only=True,
        )
    if factor is not None:
        logger.warning(
            "LoPA generation does not use factor-based parallel decoding; ignoring factor.",
            once=True,
            rank_zero_only=True,
        )
    if output_hidden_states:
        logger.warning(
            "LoPA generation does not return hidden states yet; ignoring output_hidden_states.",
            once=True,
            rank_zero_only=True,
        )

    if mask_token_id is None:
        mask_token_id = int(os.environ.get("MASK_TOKEN_ID", -1))
    if pad_token_id is None:
        pad_token_id = int(os.environ.get("PAD_TOKEN_ID", -1))
    if -1 in [mask_token_id, pad_token_id]:
        raise ValueError(
            "mask_token_id and pad_token_id must be provided either as arguments or environment variables."
        )
    if stop_until_eot:
        if eot_token_id is None and os.environ.get("EOT_TOKEN_ID", None) is None:
            raise ValueError(
                "eot_token_id must be provided either as an argument or an environment variable if stop_until_eot is set to True."
            )
        if eot_token_id is None:
            eot_token_id = int(os.environ.get("EOT_TOKEN_ID"))  # type: ignore[arg-type]

    os.environ["MASK_TOKEN_ID"] = str(mask_token_id)
    os.environ["PAD_TOKEN_ID"] = str(pad_token_id)
    if eot_token_id is not None:
        os.environ["EOT_TOKEN_ID"] = str(eot_token_id)

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None:
        attention_mask = (input_ids != pad_token_id).long()
    if attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1)
    attention_mask = attention_mask.to(model.device)

    batch_size = input_ids.size(0)
    num_blocks = gen_length // block_length
    frame = initial_frame
    deltas = []

    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (batch_size, gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[
            :,
            block_idx * block_length : (block_idx + 1) * block_length,
        ] = True

        while True:
            can_generate = check_can_generate(
                frame,
                eligible_mask=block_mask,
                num_transfer_tokens=1,
                stop_until_eot=stop_until_eot,
                mask_token_id=mask_token_id,
                eot_token_id=eot_token_id,
            )
            if not torch.any(can_generate):
                break

            transfer_index = []
            decoded_rows = []
            confidence_rows = []
            probs_rows = []

            for row_idx in range(batch_size):
                if not can_generate[row_idx]:
                    transfer_index.append(
                        torch.tensor([], dtype=torch.long, device=model.device)
                    )
                    continue

                row_transfer_index, row_decoded, row_confidence, row_probs = (
                    _lopa_generate_step(
                        model,
                        prompt=frame.prompts[row_idx],
                        generated_tokens=frame.generated_tokens[row_idx],
                        attention_mask=attention_mask[row_idx],
                        block_start=block_idx * block_length,
                        block_length=block_length,
                        alg=alg,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        threshold=threshold,
                        k=k,
                        mask_token_id=mask_token_id,
                        output_probs=output_probs,
                    )
                )
                transfer_index.append(row_transfer_index)
                decoded_rows.append(row_decoded)
                confidence_rows.append(row_confidence)
                if row_probs is not None:
                    probs_rows.append(row_probs)

            if len(decoded_rows) == 0:
                break
            if not any(index.numel() > 0 for index in transfer_index):
                raise RuntimeError(
                    "LoPA failed to decode any token in a generation step."
                )

            delta = FrameDelta(
                transfer_index=tuple(transfer_index),
                decoded_tokens=torch.stack(decoded_rows, dim=0),
                confidence=torch.stack(confidence_rows, dim=0),
                probs=(
                    torch.stack(probs_rows, dim=0)
                    if output_probs and len(probs_rows) > 0
                    else None
                ),
            )
            frame = frame.apply_delta(delta)
            deltas.append(delta.to("cpu"))

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
