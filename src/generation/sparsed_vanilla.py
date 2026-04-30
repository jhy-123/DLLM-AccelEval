import os
import torch
import torch.nn.functional as F

from src.frame import INVALID_TOKEN_ID, DecodeRecord, Frame, FrameDelta, Intermediate
from src.generation.utils import (
    check_can_generate,
    prepare_logits_for_generation,
    sample_tokens,
)
from src.generation.vanilla import (
    confidence_unmasking,
    finalize_step_metrics,
    get_num_transfer_tokens,
    init_step_metrics,
    count_transferred_tokens,
    record_step_metrics,
)
from src.utils import Timer, certainty_density, register


@torch.no_grad()
def generate_step_sparsed(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    num_transfer_tokens: torch.Tensor | int,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    mask_token_id: int | None = None,
    eot_token_id: int | None = None,
    sigma: float | None = None,
    stop_until_eot: bool = False,
    gamma: float | None = None,
    debias: bool = False,
    clip_alpha: float | None = None,
    threshold: float | None = None,
    factor: float | None = None,
    sparsed_param: dict | None = None,
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
    transfer_index_mask = remaining_mask & block_mask

    x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[can_generate]
    attention_mask = attention_mask[can_generate] if attention_mask is not None else None
    num_transfer_tokens = num_transfer_tokens[can_generate]

    outputs = model(
        x,
        attention_mask=attention_mask,
        SparseD_param=sparsed_param if sparsed_param is not None else {},
        output_hidden_states=output_hidden_states,
    )

    logits = prepare_logits_for_generation(model, outputs.logits)
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
        transfer_index_mask=transfer_index_mask[can_generate],
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


def _sparsed_vanilla_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    steps: int = 256,
    block_length: int = 256,
    gen_length: int = 256,
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
    sparsed_param: dict | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> DecodeRecord:
    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore[arg-type]
    if stop_until_eot:
        if eot_token_id is None and os.environ.get("EOT_TOKEN_ID", None) is None:
            raise ValueError(
                "eot_token_id must be provided either as an argument or an environment variable if stop_until_eot is set to True."
            )
        eot_token_id = eot_token_id or int(os.environ.get("EOT_TOKEN_ID"))  # type: ignore[arg-type]

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
        if torch.all(attention_mask == 1):
            attention_mask = None

    frame = initial_frame
    deltas = []
    total_steps = steps * num_blocks
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
        block_deltas = []
        for i in range(steps):
            step_sparsed_param = dict(sparsed_param) if sparsed_param is not None else {}
            step_sparsed_param["now_step"] = block_idx * steps + i
            step_sparsed_param.setdefault("whole_steps", total_steps)
            step_sparsed_param.setdefault("new_generation", gen_length)
            step_sparsed_param.setdefault("select", 0.5)
            step_sparsed_param.setdefault("skip", 0.2)
            step_sparsed_param.setdefault("block_size", 32)

            with Timer() as step_timer:
                delta = generate_step_sparsed(
                    model=model,
                    frame=frame,
                    block_mask=block_mask,
                    num_transfer_tokens=num_transfer_tokens[:, i],
                    attention_mask=attention_mask,
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
                    sparsed_param=step_sparsed_param,
                    output_hidden_states=output_hidden_states,
                    output_probs=output_probs,
                )
            if delta is None:
                break
            record_step_metrics(
                step_metrics,
                "refresh",
                step_timer.elapsed_time_ms,
                count_transferred_tokens(delta),
            )

            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

        deltas.extend(block_deltas)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        metrics=finalize_step_metrics(step_metrics),
        block_length=block_length,
    )


def _sparsed_vanilla_generate_dream(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    steps: int = 256,
    block_length: int = 256,
    gen_length: int = 256,
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
    sparsed_param: dict | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> DecodeRecord:
    return _sparsed_vanilla_generate(
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
        sparsed_param=sparsed_param,
        output_hidden_states=output_hidden_states,
        output_probs=output_probs,
    )


@torch.no_grad()
@register.gen_strategy("sparsed_vanilla")
def sparsed_vanilla_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    steps: int = 256,
    block_length: int = 256,
    gen_length: int = 256,
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
    sparsed_param: dict | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> DecodeRecord:
    model_type = getattr(model.config, "model_type", "").lower()

    if model_type == "dream":
        return _sparsed_vanilla_generate_dream(
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
            sparsed_param=sparsed_param,
            output_hidden_states=output_hidden_states,
            output_probs=output_probs,
        )
    return _sparsed_vanilla_generate(
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
        sparsed_param=sparsed_param,
        output_hidden_states=output_hidden_states,
        output_probs=output_probs,
    )
