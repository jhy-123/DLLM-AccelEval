import os
import torch
import torch.nn.functional as F
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import (
    check_can_generate,
    prepare_logits_for_generation,
    sample_tokens,
)
from src.generation.vanilla import get_num_transfer_tokens
from src.utils import register


def _forward_generated_logits(
    model,
    prompt: torch.Tensor,
    generated: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    output_hidden_states: bool = False,
):
    x = torch.cat([prompt.unsqueeze(0), generated.unsqueeze(0)], dim=-1)
    outputs = model(
        x,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
    )
    prompt_length = prompt.size(0)
    logits = prepare_logits_for_generation(model, outputs.logits)[:, prompt_length:]
    return logits[0], outputs


def _predict_masked_tokens(
    logits: torch.Tensor,
    generated: torch.Tensor,
    temperature: float,
    top_p: float | None,
    top_k: float | None,
    alg: str,
    mask_token_id: int,
):
    masked = generated == mask_token_id
    if not masked.any():
        return None

    masked_logits = logits[masked]
    confidence, x0, probs = sample_tokens(
        masked_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
    )

    pred_tokens = torch.full_like(generated, mask_token_id)
    pred_conf = torch.full(
        generated.shape,
        -torch.inf,
        device=generated.device,
        dtype=confidence.dtype,
    )
    pred_tokens[masked] = x0
    pred_conf[masked] = confidence

    return pred_tokens, pred_conf, probs, masked


def _select_transfer_indices(
    confidence: torch.Tensor,
    generated: torch.Tensor,
    block_start: int,
    block_end: int,
    mask_token_id: int,
    num_transfer_tokens: int,
    confidence_threshold: float | None,
):
    block_conf = confidence[block_start:block_end]
    block_mask = generated[block_start:block_end] == mask_token_id
    if not block_mask.any():
        return torch.tensor([], device=generated.device, dtype=torch.long)

    if confidence_threshold is not None:
        selected = torch.nonzero(
            block_mask & (block_conf >= confidence_threshold), as_tuple=False
        ).squeeze(-1)
        if selected.numel() > 0:
            return selected + block_start

    available = int(block_mask.sum().item())
    k = min(max(num_transfer_tokens, 1), available)
    _, idx = torch.topk(
        torch.where(block_mask, block_conf, -torch.inf),
        k=k,
        dim=-1,
    )
    return idx + block_start


def _apply_indices(
    generated: torch.Tensor,
    pred_tokens: torch.Tensor,
    indices: torch.Tensor,
):
    new_generated = generated.clone()
    if indices.numel() > 0:
        new_generated[indices] = pred_tokens[indices]
    return new_generated


def _build_stale_drafts(
    generated: torch.Tensor,
    pred_tokens: torch.Tensor,
    pred_conf: torch.Tensor,
    schedule: torch.Tensor,
    start_step: int,
    draft_steps: int,
    block_start: int,
    block_end: int,
    mask_token_id: int,
    confidence_threshold: float | None,
):
    drafts: list[torch.Tensor] = []
    current = generated
    stale_conf = pred_conf.clone()

    for draft_offset in range(draft_steps):
        step_idx = start_step + draft_offset
        if step_idx >= schedule.numel():
            break

        indices = _select_transfer_indices(
            confidence=stale_conf,
            generated=current,
            block_start=block_start,
            block_end=block_end,
            mask_token_id=mask_token_id,
            num_transfer_tokens=int(schedule[step_idx].item()),
            confidence_threshold=confidence_threshold,
        )
        if indices.numel() == 0:
            break

        current = _apply_indices(current, pred_tokens, indices)
        stale_conf[indices] = -torch.inf
        drafts.append(current.clone())

        if (current[block_start:block_end] == mask_token_id).sum() == 0:
            break

    return drafts


def _verify_drafts(
    model,
    prompt: torch.Tensor,
    drafts: list[torch.Tensor],
    attention_mask: torch.Tensor | None,
    schedule: torch.Tensor,
    start_step: int,
    block_start: int,
    block_end: int,
    temperature: float,
    top_p: float | None,
    top_k: float | None,
    alg: str,
    mask_token_id: int,
):
    verified_next: list[torch.Tensor] = []
    verified_conf: list[torch.Tensor] = []

    for draft_idx, draft in enumerate(drafts[:-1]):
        step_idx = start_step + draft_idx + 1
        logits, _ = _forward_generated_logits(
            model=model,
            prompt=prompt,
            generated=draft,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        pred = _predict_masked_tokens(
            logits=logits,
            generated=draft,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            mask_token_id=mask_token_id,
        )
        if pred is None:
            verified_next.append(draft.clone())
            verified_conf.append(
                torch.full(
                    draft.shape,
                    -torch.inf,
                    device=draft.device,
                    dtype=torch.float32,
                )
            )
            continue

        pred_tokens, pred_conf, _, _ = pred
        indices = _select_transfer_indices(
            confidence=pred_conf,
            generated=draft,
            block_start=block_start,
            block_end=block_end,
            mask_token_id=mask_token_id,
            num_transfer_tokens=int(schedule[step_idx].item()),
            confidence_threshold=None,
        )
        verified_next.append(_apply_indices(draft, pred_tokens, indices))
        verified_conf.append(pred_conf)

    return verified_next, verified_conf


def _run_single_sample(
    model,
    prompt: torch.Tensor,
    generated: torch.Tensor,
    attention_mask: torch.Tensor | None,
    schedule: torch.Tensor,
    step_idx: int,
    block_start: int,
    block_end: int,
    draft_steps: int,
    temperature: float,
    top_p: float | None,
    top_k: float | None,
    alg: str,
    mask_token_id: int,
    confidence_threshold: float | None,
    output_hidden_states: bool,
):
    logits, outputs = _forward_generated_logits(
        model=model,
        prompt=prompt,
        generated=generated,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
    )
    pred = _predict_masked_tokens(
        logits=logits,
        generated=generated,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
        mask_token_id=mask_token_id,
    )
    if pred is None:
        return None

    pred_tokens, pred_conf, _, _ = pred
    drafts = _build_stale_drafts(
        generated=generated,
        pred_tokens=pred_tokens,
        pred_conf=pred_conf,
        schedule=schedule,
        start_step=step_idx,
        draft_steps=draft_steps,
        block_start=block_start,
        block_end=block_end,
        mask_token_id=mask_token_id,
        confidence_threshold=confidence_threshold,
    )
    if not drafts:
        return None

    accept_idx = 0
    accepted_conf = pred_conf
    if len(drafts) > 1:
        verified_next, verified_conf = _verify_drafts(
            model=model,
            prompt=prompt,
            drafts=drafts,
            attention_mask=attention_mask,
            schedule=schedule,
            start_step=step_idx,
            block_start=block_start,
            block_end=block_end,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            mask_token_id=mask_token_id,
        )
        while accept_idx < len(verified_next) and torch.equal(
            verified_next[accept_idx], drafts[accept_idx + 1]
        ):
            accept_idx += 1
            accepted_conf = verified_conf[accept_idx - 1]

    accepted = drafts[accept_idx]
    changed = torch.nonzero(accepted != generated, as_tuple=False).squeeze(-1)
    if changed.numel() == 0:
        return None

    decoded_tokens = torch.full_like(generated, INVALID_TOKEN_ID)
    decoded_tokens[changed] = accepted[changed]

    confidence = torch.full(
        generated.shape,
        -torch.inf,
        device=generated.device,
        dtype=accepted_conf.dtype,
    )
    confidence[changed] = accepted_conf[changed]

    hidden_states = (
        tuple((i, hs[0]) for i, hs in enumerate(outputs.hidden_states))
        if output_hidden_states
        else tuple()
    )

    return decoded_tokens, confidence, changed, hidden_states


@torch.no_grad()
def freedave_generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    step_idx: int,
    num_transfer_tokens: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    draft_steps: int = 4,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    mask_token_id: int | None = None,
    eot_token_id: int | None = None,
    stop_until_eot: bool = False,
    confidence_threshold: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    if output_probs:
        raise ValueError("generation.output_probs is not supported by Freedave yet.")

    frame = frame.as_batch()
    batch_size = frame.prompts.size(0)
    device = frame.prompts.device
    can_generate = check_can_generate(
        frame,
        eligible_mask=block_mask,
        num_transfer_tokens=num_transfer_tokens[:, step_idx],
        stop_until_eot=stop_until_eot,
        mask_token_id=mask_token_id,
        eot_token_id=eot_token_id,
    )
    if not torch.any(can_generate):
        return None

    block_indices = torch.nonzero(block_mask[0], as_tuple=False).squeeze(-1)
    block_start = int(block_indices[0].item())
    block_end = int(block_indices[-1].item()) + 1

    transfer_index: list[torch.Tensor] = []
    decoded_rows: list[torch.Tensor] = []
    confidence_rows: list[torch.Tensor] = []
    hidden_state_rows: list[tuple[tuple[int, torch.Tensor], ...]] = []

    for batch_idx in range(batch_size):
        if not can_generate[batch_idx]:
            transfer_index.append(torch.tensor([], device=device, dtype=torch.long))
            continue

        row_attention_mask = (
            attention_mask[batch_idx : batch_idx + 1] if attention_mask is not None else None
        )
        result = _run_single_sample(
            model=model,
            prompt=frame.prompts[batch_idx],
            generated=frame.generated_tokens[batch_idx],
            attention_mask=row_attention_mask,
            schedule=num_transfer_tokens[batch_idx],
            step_idx=step_idx,
            block_start=block_start,
            block_end=block_end,
            draft_steps=draft_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            mask_token_id=mask_token_id,
            confidence_threshold=confidence_threshold,
            output_hidden_states=output_hidden_states,
        )

        if result is None:
            transfer_index.append(torch.tensor([], device=device, dtype=torch.long))
            continue

        decoded_tokens, confidence, changed, hidden_states = result
        transfer_index.append(changed)
        decoded_rows.append(decoded_tokens)
        confidence_rows.append(confidence)
        hidden_state_rows.append(hidden_states)

    if not decoded_rows:
        return None

    intermediate = Intermediate()
    if output_hidden_states and hidden_state_rows:
        intermediate = Intermediate(
            hidden_states=tuple(
                (
                    layer_idx,
                    torch.stack(
                        [row[layer_pos][1] for row in hidden_state_rows], dim=0
                    ),
                )
                for layer_pos, (layer_idx, _) in enumerate(hidden_state_rows[0])
            )
        )

    return FrameDelta(
        transfer_index=tuple(transfer_index),
        decoded_tokens=torch.stack(decoded_rows, dim=0),
        confidence=torch.stack(confidence_rows, dim=0),
        intermediate=intermediate,
    )


@register.gen_strategy("freedave")
def freedave_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    steps: int = 128,
    block_length: int = 32,
    gen_length: int = 128,
    draft_steps: int = 4,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eot_token_id: int | None = None,
    stop_until_eot: bool = False,
    confidence_threshold: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> DecodeRecord:
    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))
    pad_token_id = pad_token_id or int(os.environ.get("PAD_TOKEN_ID", -1))

    if stop_until_eot:
        if eot_token_id is None and os.environ.get("EOT_TOKEN_ID", None) is None:
            raise ValueError(
                "eot_token_id must be provided either as an argument or an environment variable if stop_until_eot is set to True."
            )
        eot_token_id = eot_token_id or int(os.environ.get("EOT_TOKEN_ID"))

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length.")

    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by the number of blocks.")
    steps_per_block = steps // num_blocks

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id >= 0:
        attention_mask = (input_ids != pad_token_id).long()
    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(model.device)

    frame = initial_frame
    deltas = []

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

        schedule = get_num_transfer_tokens(block_mask, steps_per_block)
        for step_idx in range(steps_per_block):
            delta = freedave_generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                step_idx=step_idx,
                num_transfer_tokens=schedule,
                attention_mask=attention_mask,
                draft_steps=draft_steps,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                mask_token_id=mask_token_id,
                eot_token_id=eot_token_id,
                stop_until_eot=stop_until_eot,
                confidence_threshold=confidence_threshold,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )
            if delta is None:
                break

            deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta, mask_token_id=mask_token_id)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
