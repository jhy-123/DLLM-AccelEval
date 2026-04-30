import os
import types
import torch

from transformers import AutoTokenizer

from src.frame import DecodeRecord, Frame
from src.utils import register
from src.models.fast_dllm_v2.generation_utils import Fast_dLLM_QwenForCausalLM as FastDLLMGenMixin

_TOKENIZER_CACHE = {}


def _get_tokenizer(model):
    tokenizer_path = getattr(model.config, "_name_or_path", None)
    if tokenizer_path is None:
        raise ValueError("Fast dLLM v2 tokenizer path is missing.")
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_path)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.mask_token is None:
            tokenizer.mask_token = "|<MASK>|"
        tokenizer.eot_token = tokenizer.eos_token
        tokenizer.eot_token_id = tokenizer.eos_token_id
        _TOKENIZER_CACHE[tokenizer_path] = tokenizer
    return tokenizer


@register.gen_strategy("fast_dllm_v2")
@torch.no_grad()
def fast_dllm_v2_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    gen_length: int = 256,
    block_length: int = 32,
    steps: int = 256,
    temperature: float = 0.0,
    top_p: float | None = None,
    mask_token_id: int | None = None,
    eot_token_id: int | None = None,
    pad_token_id: int | None = None,
    threshold: float | None = None,
    use_block_cache: bool = False,
    bd_size: int | None = None,
    small_block_size: int | None = None,
):
    del steps
    device = model.device
    batch_size = input_ids.size(0)
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID", "151665"))
    tokenizer = _get_tokenizer(model)
    eot_token_id = eot_token_id or tokenizer.eos_token_id
    pad_token_id = pad_token_id or tokenizer.pad_token_id
    # Canonicalize Fast-dLLM v2 block size to the shared generation.block_length.
    # Keep bd_size as a backward-compatible alias that can still override block_length.
    block_size = bd_size or block_length
    small_block_size = small_block_size or max(1, block_size // 4)
    threshold = 0.9 if threshold is None else threshold
    top_p = 0.95 if top_p is None else top_p

    if not hasattr(model, "mdm_sample"):
        model.mdm_sample = types.MethodType(FastDLLMGenMixin.batch_sample, model)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    seq_lens = attention_mask.sum(dim=-1).to(torch.long)
    max_len = int(seq_lens.max().item())
    trimmed = []
    for i in range(batch_size):
        seq_len = int(seq_lens[i].item())
        trimmed_ids = input_ids[i, -seq_len:].to(device)
        if seq_len < max_len:
            right_pad = torch.full((max_len - seq_len,), mask_token_id, dtype=torch.long, device=device)
            trimmed_ids = torch.cat([trimmed_ids, right_pad], dim=0)
        trimmed.append(trimmed_ids)
    batched_input_ids = torch.stack(trimmed, dim=0)

    generated, full_step, timing_metrics = model.mdm_sample(
        batched_input_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        max_new_tokens=gen_length,
        small_block_size=small_block_size,
        min_len=int(seq_lens.min().item()),
        seq_len=seq_lens.to(device),
        mask_id=mask_token_id,
        threshold=threshold,
        stop_token=eot_token_id,
        use_block_cache=use_block_cache,
        top_p=top_p,
        temperature=temperature,
    )
    refresh_step_count = int(timing_metrics.get("avg_q_full_step_count", 0.0) or 0.0)
    step_count = int(full_step) + refresh_step_count

    ordered = [generated[i].to(device) for i in range(batch_size)]
    generated_tokens = torch.full((batch_size, gen_length), eot_token_id, dtype=torch.long, device=device)
    confidence = torch.zeros((batch_size, gen_length), dtype=torch.float32, device=device)
    steps_tensor = torch.zeros((batch_size, gen_length), dtype=torch.long, device=device)

    for i, full_seq in enumerate(ordered):
        seq_len = int(seq_lens[i].item())
        continuation = full_seq[seq_len:]
        continuation = torch.where(continuation == mask_token_id, torch.full_like(continuation, eot_token_id), continuation)
        num_tokens = min(int(continuation.numel()), gen_length)
        if num_tokens == 0:
            continue
        generated_tokens[i, :num_tokens] = continuation[:num_tokens]
        if full_step <= 1:
            step_ids = torch.zeros(num_tokens, dtype=torch.long, device=device)
        else:
            step_ids = torch.floor(torch.arange(num_tokens, device=device, dtype=torch.float32) * full_step / max(num_tokens, 1)).long()
            step_ids.clamp_(max=full_step - 1)
        steps_tensor[i, :num_tokens] = step_ids
        if num_tokens < gen_length:
            steps_tensor[i, num_tokens:] = step_ids[-1] if num_tokens > 0 else 0

    final_frame = Frame(
        prompts=input_ids.to(device),
        generated_tokens=generated_tokens,
        confidence=confidence,
        steps=steps_tensor,
    )

    return DecodeRecord(
        initial_frame=final_frame,
        deltas=[],
        metrics={
            "avg_step_count": float(step_count),
            "avg_q_part_step_count": float(full_step),
            "avg_q_full_step_count": float(refresh_step_count),
            "avg_q_part_step_time_ms": timing_metrics.get("avg_q_part_step_time_ms"),
            "avg_q_full_step_time_ms": timing_metrics.get(
                "avg_q_full_step_time_ms"
            ),
            "init_time_ms": timing_metrics.get("init_time_ms")
            or timing_metrics.get("avg_q_full_step_time_ms"),
            "avg_q_full_step_fraction": (
                float(refresh_step_count) / float(step_count)
                if step_count > 0
                else 0.0
            ),
        },
        block_length=block_size,
    )
