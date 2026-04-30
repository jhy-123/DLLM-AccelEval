import os
import math
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


class Sampler:
    def __init__(self, length: int, window: int | None = None):
        self.length = length
        self.window = window if window is not None else length
        if self.window > self.length:
            raise ValueError("window must be <= length")

    def pdf(self, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, src: torch.Tensor) -> torch.Tensor:
        pdf = self.pdf(src.device)
        if src.numel() == 0:
            return src
        uniform = torch.rand(src.shape[0], device=src.device)
        return src[uniform < pdf[: src.shape[0]]]


class GaussianSampler(Sampler):
    def __init__(self, length: int, window: int | None = None, sigma: float = 1.0, scale: float = 1.0):
        super().__init__(length, window)
        self.sigma = sigma
        self.scale = scale

    def pdf(self, device: torch.device) -> torch.Tensor:
        mean = 0.0
        std_dev = 1.0
        x = torch.linspace(mean, mean + self.sigma * std_dev, self.window, device=device)
        pdf = self.scale * torch.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (
            std_dev * math.sqrt(2 * math.pi)
        )
        if self.window < self.length:
            pdf = torch.cat([pdf, torch.zeros(self.length - self.window, device=device)])
        return pdf


class UniformSampler(Sampler):
    def __init__(self, length: int, window: int | None = None, number: int = 0):
        super().__init__(length, window)
        self.number = number
        if self.number > self.length:
            raise ValueError("number must be <= length")

    def sample(self, src: torch.Tensor) -> torch.Tensor:
        if src.numel() == 0 or self.number <= 0:
            return src[:0]
        if self.number >= src.shape[0]:
            return src
        max_window = min(src.shape[0], self.window)
        indices = torch.sort(torch.randperm(max_window, device=src.device)[: self.number]).values
        return src[indices]


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_indices: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor | None,
    threshold: float | None = None,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_indices, x0, x)
    confidence = torch.where(mask_indices, x0_p, -torch.inf)

    if threshold is not None:
        num_transfer_tokens = mask_indices.sum(dim=1)
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be provided if threshold is None")

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item())
        if k <= 0:
            continue
        _, select_index = torch.topk(confidence[j], k=k)
        transfer_index[j, select_index] = True
        if threshold is not None:
            for n in range(1, k):
                if confidence[j, select_index[n]] < threshold:
                    transfer_index[j, select_index[n]] = False

    return x0, transfer_index, x0_p


def build_q_indices(
    *,
    block_end: int,
    seq_len: int,
    sampler: Sampler | None,
    device: torch.device,
    batch_size: int,
):
    q_indices = torch.arange(block_end, device=device)
    if sampler is not None and block_end < seq_len:
        suffix_indices = sampler.sample(torch.arange(block_end, seq_len, device=device))
        q_indices = torch.cat([q_indices, suffix_indices], dim=0)
    return q_indices.unsqueeze(0).expand(batch_size, -1)


@register.gen_strategy("dpad")
@torch.no_grad()
def dpad_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    steps: int = 128,
    block_length: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    threshold: float | None = None,
    dropout: str = "gaussian",
    sigma: float | None = None,
    scale: float | None = None,
    preserved_tokens: int = 0,
    window: int | None = None,
    early_termination: bool = True,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eot_token_id: int | None = None,
    output_hidden_states: bool = False,
    cache_cls=None,
) -> DecodeRecord:
    if cache_cls is not None:
        logger.warning("DPad generation does not support cache; ignoring cache_cls.")

    # if model.config.model_type.lower() != "llada":
    #     raise NotImplementedError("DPad generation is only implemented for LLaDA.")

    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore
    if early_termination:
        if eot_token_id is None and os.environ.get("EOT_TOKEN_ID", None) is None:
            raise ValueError(
                "eot_token_id must be provided either as an argument or an environment variable if early_termination is True."
            )
        eot_token_id = eot_token_id or int(os.environ.get("EOT_TOKEN_ID"))  # type: ignore

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by number of blocks")
    steps = steps // num_blocks

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()
    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(model.device)

    if window is None:
        window = gen_length

    if dropout in {"none", "null", None}:
        sampler = None
    elif dropout == "gaussian":
        if sigma is None:
            sigma = 4.0
        if scale is None:
            scale = 2.0
        sampler = GaussianSampler(length=gen_length, window=window, sigma=sigma, scale=scale)
    elif dropout == "uniform":
        sampler = UniformSampler(length=gen_length, window=window, number=preserved_tokens)
    else:
        raise ValueError(f"dropout {dropout} not recognized")

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

        block_mask_indices = (
            frame.generated_tokens[:, block_start:block_end] == mask_token_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_indices, steps)
        num_transfer_tokens[finished] = 0

        q_indices = build_q_indices(
            block_end=block_end_abs,
            seq_len=seq_len,
            sampler=sampler,
            device=model.device,
            batch_size=batch_size,
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

            x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[can_generate]
            q_idx = q_indices[: x.size(0)]
            q_len = q_idx.shape[1]
            x_pruned = x[:, :q_len]

            attn_mask = None
            if attention_mask is not None:
                full_mask = attention_mask[can_generate]
                attn_mask = full_mask.gather(1, q_idx)

            outputs = model(
                x_pruned,
                attention_mask=attn_mask,
                position_ids=q_idx,
                output_hidden_states=output_hidden_states,
            )
            logits = prepare_logits_for_generation(model, outputs.logits)

            mask_indices = x_pruned == mask_token_id
            mask_indices[:, :prompt_length] = False
            mask_indices[:, block_end_abs:] = False

            x0, transfer_index_mask, x0_p = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_indices,
                x_pruned,
                num_transfer_tokens=num_transfer_tokens[can_generate, step_idx],
                threshold=threshold,
            )

            x_pruned[transfer_index_mask] = x0[transfer_index_mask]

            decoded_tokens = torch.full(
                (x.size(0), gen_length),
                INVALID_TOKEN_ID,
                dtype=torch.long,
                device=model.device,
            )
            conf_dtype = (
                frame.confidence.dtype
                if frame.confidence is not None
                else torch.float32
            )
            confidence = torch.full(
                (x.size(0), gen_length),
                -torch.inf,
                dtype=conf_dtype,
                device=model.device,
            )

            block_slice = slice(block_start, block_end)
            x0_block = x0[:, block_start_abs:block_end_abs]
            mask_block = mask_indices[:, block_start_abs:block_end_abs]
            decoded_tokens[:, block_slice] = torch.where(
                mask_block, x0_block, INVALID_TOKEN_ID
            )
            confidence[:, block_slice] = torch.where(
                mask_block,
                x0_p[:, block_start_abs:block_end_abs].to(conf_dtype),
                torch.tensor(-torch.inf, device=model.device, dtype=conf_dtype),
            )

            transfer_index_active = []
            for i in range(x.size(0)):
                local_idx = torch.nonzero(
                    transfer_index_mask[i, block_start_abs:block_end_abs], as_tuple=False
                ).squeeze(1)
                transfer_index_active.append(local_idx + block_start)

            transfer_index_iter = iter(transfer_index_active)
            transfer_index = tuple(
                (
                    next(transfer_index_iter)
                    if is_not_finished
                    else torch.tensor([], dtype=torch.long, device=model.device)
                )
                for is_not_finished in can_generate
            )

            delta = FrameDelta(
                transfer_index=transfer_index,
                decoded_tokens=decoded_tokens,
                confidence=confidence,
            )
            deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

            if early_termination:
                block_done = (
                    frame.generated_tokens[:, block_slice] != mask_token_id
                ).all(dim=-1)
                eos_in_block = (
                    frame.generated_tokens[:, block_slice] == eot_token_id
                ).any(dim=-1)
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
