import os
import torch
import torch.nn.functional as F

from typing import Type

from src.cache import dCache
from src.frame import INVALID_TOKEN_ID, Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import (
    prepare_logits_for_generation,
    sample_tokens,
    check_can_generate,
)
from src.utils import certainty_density, register
from src.generation.vanilla import (
    classify_cache_step,
    confidence_unmasking,
    finalize_step_metrics,
    get_num_transfer_tokens,
    init_step_metrics,
    count_transferred_tokens,
    record_step_metrics,
)
from src.utils import Timer


@torch.no_grad()
def generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    num_transfer_tokens: torch.Tensor | int,
    prev_probs: torch.Tensor,
    kl_history: torch.Tensor,
    kl_threshold: float = 0.02,
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
    threshold: float | None = None,
    factor: float | None = None,
    sparsed_param: dict | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape
    gen_length = frame.generated_tokens.size(1)
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
        SparseD_param=sparsed_param if sparsed_param is not None else {},
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
    logits = logits[:, prompt_length:].to(torch.float64)

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
        alg=alg,
    )
    scores = confidence = torch.where(
        transfer_index_mask[can_generate], confidence, -torch.inf
    )
    if sigma is not None and sigma > 0:
        scores = confidence * certainty_density(
            ~remaining_mask[can_generate], sigma=sigma
        )

    eps = 1e-12
    kl_current_prev = (
        p * (torch.log(p + eps) - torch.log(prev_probs[can_generate] + eps))
    ).sum(dim=-1)
    active_index = torch.nonzero(can_generate, as_tuple=True)[0]
    kl_history[active_index] = kl_history[active_index].roll(shifts=-1, dims=-1)
    kl_history[active_index, ..., -1] = kl_current_prev

    stable_mask = torch.zeros_like(transfer_index_mask)
    stable_mask[active_index] = torch.all(
        kl_history[active_index] < kl_threshold, dim=-1
    )
    failback_transfer_mask = transfer_index_mask & block_mask
    stable_transfer_mask = failback_transfer_mask & stable_mask

    ta = confidence_unmasking(
        scores=scores,
        transfer_index_mask=stable_transfer_mask[can_generate],
        min_transfer_tokens=torch.zeros_like(num_transfer_tokens),
        threshold=threshold,
        factor=factor,
    )

    tb = confidence_unmasking(
        scores=scores,
        transfer_index_mask=failback_transfer_mask[can_generate],
        min_transfer_tokens=num_transfer_tokens,
        threshold=None,
        factor=None,
    )

    transfer_index = tuple(
        ta_idx if ta_idx.numel() > 0 else tb_idx for ta_idx, tb_idx in zip(ta, tb)
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
        extra=dict(curr_probs=p, active_index=active_index),
    )


@register.gen_strategy("klass")
def klass_generate(
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
    kl_threshold: float = 0.01,
    kl_history_length: int = 2,
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
) -> DecodeRecord:
    """
    KLASS generation strategy: KL-Adaptive Stability Sampling.
    """

    if sparsed and cache_cls is not None:
        raise ValueError(
            "SparseD under generation=klass does not support cache. "
            "Please remove cache=prefix/dkvcache when generation.sparsed=true."
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
    batch_size, prompt_length = input_ids.shape

    deltas = []
    kl_history = torch.zeros(
        (batch_size, gen_length, kl_history_length),
        dtype=torch.float64,
        device=model.device,
    )
    prev_probs = torch.zeros(
        (batch_size, gen_length, model.config.vocab_size),
        dtype=torch.float64,
        device=model.device,
    )
    step_metrics = init_step_metrics()

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

        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps)
        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)
        block_deltas = []
        for i in range(steps):
            pre_step_is_first_step = False
            pre_step_missing_q_mask = False
            if cache is not None:
                pre_step_is_first_step = bool(getattr(cache, "_is_first_step", False))
                pre_step_missing_q_mask = getattr(cache, "active_q_mask", None) is None
                cache.on_step_start(block_mask, frame)

            step_sparsed_param = None
            if sparsed:
                step_sparsed_param = dict(sparsed_param) if sparsed_param is not None else {}
                step_sparsed_param["now_step"] = block_idx * steps + i
                step_sparsed_param.setdefault("whole_steps", steps * num_blocks)
                step_sparsed_param.setdefault("new_generation", gen_length)
                step_sparsed_param.setdefault("select", sparsed_select)
                step_sparsed_param.setdefault("skip", sparsed_skip)
                step_sparsed_param.setdefault("block_size", sparsed_block_size)

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
                    prev_probs=prev_probs,
                    kl_history=kl_history,
                    kl_threshold=kl_threshold,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    alg=alg,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    sigma=sigma,
                    mask_token_id=mask_token_id,
                    eot_token_id=eot_token_id,
                    stop_until_eot=stop_until_eot,
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
                step_type,
                step_timer.elapsed_time_ms,
                count_transferred_tokens(delta),
            )

            prev_probs[delta.extra.pop("active_index")] = delta.extra.pop("curr_probs")
            delta = delta.to(dtype=model.dtype)
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
