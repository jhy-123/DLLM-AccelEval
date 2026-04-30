import re
import os
import accelerate
import torch

from datetime import timedelta
from typing import Iterable
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm

from src.utils import Timer

try:
    from lm_eval.models.huggingface import HFLM
except ImportError:
    HFLM = None  # type: ignore[assignment]

try:
    from lm_eval.models.utils import Collator, handle_stop_sequences
except ImportError:
    Collator = None  # type: ignore[assignment]
    handle_stop_sequences = None  # type: ignore[assignment]

try:
    from lm_eval.models.utils import normalize_gen_kwargs
except ImportError:

    def normalize_gen_kwargs(gen_kwargs: dict, default_max_gen_toks: int | None) -> dict:
        kwargs = dict(gen_kwargs)
        if kwargs.get("max_gen_toks") is None and default_max_gen_toks is not None:
            kwargs["max_gen_toks"] = default_max_gen_toks
        return kwargs

try:
    from lm_eval.models.utils import postprocess_generated_text
except ImportError:

    def postprocess_generated_text(
        generation: str,
        stop: list[str] | None = None,
        think_end_token: str | None = None,
    ) -> str:
        text = generation
        if think_end_token and think_end_token in text:
            text = text.split(think_end_token)[-1]
        if stop:
            stop_positions = [text.find(item) for item in stop if item and item in text]
            if stop_positions:
                text = text[: min(stop_positions)]
        return text

HAS_HFLM_UTILS = all(
    item is not None for item in (HFLM, Collator, handle_stop_sequences)
)

STEP_METRIC_NAMES = [
    "avg_step_count",
    "avg_q_part_step_count",
    "avg_q_full_step_count",
    "init_time_ms",
    "generation_speed",
    "avg_step_time_ms",
    "avg_q_part_step_time_ms",
    "avg_q_full_step_time_ms",
    "avg_q_full_step_fraction",
]


def resolve_ar_max_new_tokens(
    cfg: DictConfig,
    gen_kwargs: dict,
    task_name: str | None = None,
    default_max_gen_toks: int | None = None,
) -> int:
    dataset_name = str(cfg.get("dataset", {}).get("name", ""))
    is_longbench = dataset_name.startswith("longbench") or (
        isinstance(task_name, str) and task_name.startswith("longbench")
    )

    max_new_tokens = (
        gen_kwargs.get("max_gen_toks")
        if is_longbench and not cfg.generation.get("_cli_gen_length_provided", False)
        else cfg.generation.get("gen_length")
    )
    if max_new_tokens is None:
        max_new_tokens = gen_kwargs.get("max_gen_toks", default_max_gen_toks)
    if max_new_tokens is None:
        raise ValueError("Unable to determine max_new_tokens for AR generation.")
    return int(max_new_tokens)


def assign_metric_values(target, metric_values: dict[str, float | None]) -> None:
    metrics = getattr(target, "metrics", None)
    for name, value in metric_values.items():
        setattr(target, name, value)
        if isinstance(metrics, dict):
            metrics[name] = value


def _sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if temperature is not None and temperature > 0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -torch.inf)

    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -torch.inf)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def timed_causal_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int | None,
    tokenizer=None,
    stop_sequences: list[str] | None = None,
    do_sample: bool = False,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    generated_tokens: list[torch.Tensor] = []
    active = torch.ones(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    decode_time_ms = 0.0
    decode_step_count = 0

    with Timer() as prefill_timer:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values

    next_token = _sample_next_token(
        outputs.logits[:, -1, :],
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    for step_idx in range(max_new_tokens):
        if step_idx > 0:
            with Timer() as decode_timer:
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            decode_time_ms += decode_timer.elapsed_time_ms
            decode_step_count += 1
            past_key_values = outputs.past_key_values
            next_token = _sample_next_token(
                outputs.logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        if pad_token_id is not None:
            next_token = torch.where(
                active[:, None],
                next_token,
                torch.full_like(next_token, pad_token_id),
            )
        generated_tokens.append(next_token)
        attention_mask = torch.cat([attention_mask, active[:, None].long()], dim=-1)

        if eos_token_id is not None:
            active = active & (next_token.squeeze(-1) != eos_token_id)

        if tokenizer is not None and stop_sequences:
            current_ids = torch.cat(generated_tokens, dim=-1)
            for batch_idx in range(current_ids.size(0)):
                if not active[batch_idx]:
                    continue
                text = tokenizer.decode(
                    current_ids[batch_idx],
                    spaces_between_special_tokens=False,
                    skip_special_tokens=False,
                )
                if any(stop and stop in text for stop in stop_sequences):
                    active[batch_idx] = False

        if not bool(active.any()):
            break

    if generated_tokens:
        generated_ids = torch.cat(generated_tokens, dim=-1)
    else:
        generated_ids = torch.empty(
            (input_ids.size(0), 0), dtype=input_ids.dtype, device=input_ids.device
        )

    return torch.cat([input_ids, generated_ids], dim=-1), {
        "prefill_time_ms": prefill_timer.elapsed_time_ms,
        "decode_time_ms": decode_time_ms,
        "decode_step_count": float(decode_step_count),
    }


def has_bos_prefix(sequence: str, bos_str: str | Iterable[str] | None = None):
    if bos_str is None:
        return False
    if isinstance(bos_str, str):
        return sequence.startswith(bos_str)
    return any(sequence.startswith(x) for x in bos_str)


def parse_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    try:
        return getattr(torch, dtype_name)
    except AttributeError as e:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from e


def build_ar_model_kwargs(
    cfg: DictConfig, local_process_index: int, num_processes: int
) -> tuple[dict, object | None, str | None]:
    torch_dtype = parse_torch_dtype(cfg.model.get("dtype"))
    model_kwargs = {
        "trust_remote_code": cfg.model.get("trust_remote_code", True),
        "low_cpu_mem_usage": cfg.model.get("low_cpu_mem_usage", True),
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    attn_implementation = cfg.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    device_map = cfg.model.get("device_map")
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    elif attn_implementation == "flash_attention_2":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "flash_attention_2 requires CUDA, but no GPU is available."
            )
        local_rank = local_process_index if num_processes > 1 else 0
        model_kwargs["device_map"] = {"": local_rank}

    return model_kwargs, device_map, attn_implementation


def load_ar_model_and_tokenizer(
    cfg: DictConfig,
    device: torch.device,
    local_process_index: int | None = None,
    num_processes: int | None = None,
) -> tuple[
    AutoModelForCausalLM,
    PreTrainedTokenizer | PreTrainedTokenizerFast,
]:
    if local_process_index is None:
        local_process_index = int(os.environ.get("LOCAL_RANK", "0"))
    if num_processes is None:
        num_processes = int(os.environ.get("WORLD_SIZE", "1"))

    model_kwargs, device_map, attn_implementation = build_ar_model_kwargs(
        cfg, local_process_index, num_processes
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        **model_kwargs,
    ).eval()
    if device_map is None and attn_implementation != "flash_attention_2":
        model = model.to(device)

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
        AutoTokenizer.from_pretrained(
            cfg.model.path,
            trust_remote_code=cfg.model.get("trust_remote_code", True),
            use_fast=cfg.model.get("use_fast", True),
        )
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.model.get("padding_side"):
        tokenizer.padding_side = cfg.model.padding_side

    return model, tokenizer


class AutoRegressiveEval(TemplateLM):
    def __init__(self, cfg: DictConfig, **kwargs):
        accelerator_kwargs = accelerate.InitProcessGroupKwargs(
            timeout=timedelta(weeks=52)
        )
        self.accelerator = accelerate.Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.cfg = cfg
        self._metrics: dict[str, float | None] = {}
        self._device = self.accelerator.device

        self.model, self.tokenizer = load_ar_model_and_tokenizer(
            cfg,
            self._device,
            self.accelerator.local_process_index,
            self.accelerator.num_processes,
        )

        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

    @property
    def metrics(self) -> dict[str, float | None]:
        return self._metrics

    @property
    def device(self):
        return self._device

    @property
    def eot_token_id(self) -> int:  # type: ignore
        return self.tokenizer.eos_token_id  # type: ignore

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        add_special_tokens = add_special_tokens or self.cfg.get("add_bos_token")
        if add_special_tokens is None and has_bos_prefix(
            string, self.tokenizer.decode(self.prefix_token_id)
        ):
            add_special_tokens = False
        if add_special_tokens is not None:
            kwargs["add_special_tokens"] = add_special_tokens
        return self.tokenizer.encode(string, **kwargs)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "AutoRegressiveEval currently supports generate_until only."
        )

    def _loglikelihood_tokens(self, requests, **kwargs):
        raise NotImplementedError(
            "AutoRegressiveEval currently supports generate_until only."
        )

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        assert isinstance(chat_templated, str)
        return chat_templated

    def _trim_stop_words(
        self,
        output_ids: torch.Tensor,
        stop_words: list[str],
    ) -> tuple[torch.Tensor, str]:
        text = self.tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
            skip_special_tokens=False,
        )
        effective_text = text
        if stop_words:
            pattern = r"|".join(re.escape(sw) for sw in stop_words if sw)
            if pattern:
                match = re.search(pattern, effective_text)
                if match:
                    effective_text = effective_text[: match.start()]

        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    effective_text = effective_text.replace(special_tok, "")
            else:
                effective_text = effective_text.replace(special_token, "")

        effective_text = effective_text.rstrip()
        effective_ids = self.tokenizer(
            effective_text, add_special_tokens=False
        ).input_ids
        return torch.tensor(effective_ids, dtype=torch.long), effective_text

    def _trim_to_eos(self, output_ids: torch.Tensor) -> torch.Tensor:
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return output_ids
        eos_positions = (output_ids == eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            return output_ids[: int(eos_positions[0].item())]
        return output_ids

    @torch.no_grad()
    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False):
        out, throughput, tps = [], [], []
        full_throughput, full_tps = [], []
        generation_speed = []
        latency = []
        input_length = []
        avg_step_count = []
        avg_generation_step_count = []
        avg_generated_token_count = []
        avg_decode_step_count = []
        avg_refresh_step_count = []
        avg_prefill_time_ms = []
        avg_step_time_ms = []
        avg_decode_step_time_ms = []
        avg_refresh_step_time_ms = []
        avg_refresh_step_fraction = []

        for instance in tqdm(
            requests,
            total=len(requests),
            desc="Generating...",
            disable=disable_tqdm or not self.accelerator.is_main_process,
        ):
            context, gen_kwargs = instance.args
            task_name = getattr(self, "current_task_name", None)
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_new_tokens = resolve_ar_max_new_tokens(
                self.cfg,
                gen_kwargs,
                task_name=task_name,
                default_max_gen_toks=self.cfg.generation.get("gen_length"),
            )

            if self.cfg.get("add_bos_token", False) and self.tokenizer.bos_token:
                context = (
                    self.tokenizer.bos_token + context
                    if not has_bos_prefix(context, self.tokenizer.bos_token)
                    else context
                )

            tokenized = self.tokenizer(
                [context], return_tensors="pt", padding=True, padding_side="left"
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            temperature = self.cfg.generation.get("temperature")
            top_k = self.cfg.generation.get("top_k")
            top_p = self.cfg.generation.get("top_p")
            do_sample = bool(
                (temperature is not None and temperature > 0)
                or (top_k is not None and top_k > 0)
                or (top_p is not None and top_p < 1)
            )

            with Timer("eval") as timer:
                output_ids, timing_metrics = timed_causal_generate(
                    self.model,
                    tokenized["input_ids"],
                    tokenized["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    tokenizer=self.tokenizer,
                    stop_sequences=until,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            generated_ids = output_ids[0][len(tokenized["input_ids"][0]) :]
            eos_trimmed_ids = self._trim_to_eos(generated_ids)
            _effective_ids, generated_text = self._trim_stop_words(
                eos_trimmed_ids, until
            )

            throughput_tokens = int(eos_trimmed_ids.numel())
            full_new_tokens = (
                throughput_tokens
                if self.cfg.generation.get("stop_until_eos", False)
                else int(generated_ids.numel())
            )

            out.append(generated_text)
            input_length.append(torch.sum(tokenized["attention_mask"]).item())

            if timer.elapsed_time_s > 0:
                throughput.append(throughput_tokens / timer.elapsed_time_s)
                full_throughput.append(full_new_tokens / timer.elapsed_time_s)
                latency.append(timer.elapsed_time_s)
            prefill_time_ms = timing_metrics["prefill_time_ms"]
            decode_time_ms = timing_metrics["decode_time_ms"]
            decode_step_count = timing_metrics["decode_step_count"]
            generation_time_s = (timer.elapsed_time_ms - prefill_time_ms) / 1000.0
            if generation_time_s > 0:
                generation_speed.append(throughput_tokens / generation_time_s)
            avg_generation_step_count.append(float(decode_step_count))
            avg_generated_token_count.append(float(throughput_tokens))
            refresh_steps = 1.0
            if throughput_tokens > 0:
                total_step_count = decode_step_count + refresh_steps
                step_time_ms = timer.elapsed_time_ms / total_step_count
                avg_step_count.append(total_step_count)
                avg_decode_step_count.append(decode_step_count)
                avg_refresh_step_count.append(refresh_steps)
                avg_prefill_time_ms.append(prefill_time_ms)
                avg_step_time_ms.append(step_time_ms)
                avg_decode_step_time_ms.append(
                    decode_time_ms / decode_step_count if decode_step_count > 0 else 0.0
                )
                avg_refresh_step_time_ms.append(prefill_time_ms)
                avg_refresh_step_fraction.append(refresh_steps / total_step_count)
            else:
                avg_step_count.append(1.0)
                avg_decode_step_count.append(0.0)
                avg_refresh_step_count.append(refresh_steps)
                avg_prefill_time_ms.append(prefill_time_ms)
                avg_step_time_ms.append(prefill_time_ms)
                avg_decode_step_time_ms.append(0.0)
                avg_refresh_step_time_ms.append(prefill_time_ms)
                avg_refresh_step_fraction.append(1.0)

            tps.append(1.0 if throughput_tokens > 0 else 0.0)
            full_tps.append(1.0 if full_new_tokens > 0 else 0.0)

        gathered_metrics = {
            "throughput": self.accelerator.gather_for_metrics(throughput),
            "tps": self.accelerator.gather_for_metrics(tps),
            "full_throughput": self.accelerator.gather_for_metrics(full_throughput),
            "full_tps": self.accelerator.gather_for_metrics(full_tps),
            "generation_speed": self.accelerator.gather_for_metrics(generation_speed),
            "latency": self.accelerator.gather_for_metrics(latency),
            "input_length": self.accelerator.gather_for_metrics(input_length),
            "avg_step_count": self.accelerator.gather_for_metrics(avg_step_count),
            "avg_generation_step_count": self.accelerator.gather_for_metrics(
                avg_generation_step_count
            ),
            "avg_generated_token_count": self.accelerator.gather_for_metrics(
                avg_generated_token_count
            ),
            "avg_q_part_step_count": self.accelerator.gather_for_metrics(
                avg_decode_step_count
            ),
            "avg_q_full_step_count": self.accelerator.gather_for_metrics(
                avg_refresh_step_count
            ),
            "init_time_ms": self.accelerator.gather_for_metrics(
                avg_prefill_time_ms
            ),
            "avg_step_time_ms": self.accelerator.gather_for_metrics(avg_step_time_ms),
            "avg_q_part_step_time_ms": self.accelerator.gather_for_metrics(
                avg_decode_step_time_ms
            ),
            "avg_q_full_step_time_ms": self.accelerator.gather_for_metrics(
                avg_refresh_step_time_ms
            ),
            "avg_q_full_step_fraction": self.accelerator.gather_for_metrics(
                avg_refresh_step_fraction
            ),
        }

        if self.accelerator.is_main_process:
            metric_values = {
                name: (sum(values) / max(len(values), 1) if len(values) > 0 else None)
                for name, values in gathered_metrics.items()
            }
            self.metrics.update(metric_values)
            assign_metric_values(
                self,
                {
                    **metric_values,
                    "total_time": Timer.get_cumulative_s("eval"),
                },
            )

        return out


if HAS_HFLM_UTILS:

    class HFAutoRegressiveEval(HFLM):
        def __init__(self, *args, **kwargs):
            self._metrics: dict[str, float | None] = {}
            super().__init__(*args, **kwargs)
            accelerator_kwargs = accelerate.InitProcessGroupKwargs(
                timeout=timedelta(weeks=52)
            )
            self.accelerator = accelerate.Accelerator(
                kwargs_handlers=[accelerator_kwargs]
            )
            self._device = self.accelerator.device
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes

        @property
        def metrics(self) -> dict[str, float | None]:
            return self._metrics

        def _trim_padding(self, output_ids: list[int]) -> list[int]:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                return output_ids
            end = len(output_ids)
            while end > 0 and output_ids[end - 1] == pad_token_id:
                end -= 1
            return output_ids[:end]

        def _trim_to_eos(self, output_ids: list[int]) -> list[int]:
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                return output_ids
            try:
                return output_ids[: output_ids.index(eos_token_id)]
            except ValueError:
                return output_ids

        def _gather_for_metrics(self, values: list[float]) -> list[float]:
            if hasattr(self, "accelerator"):
                gathered = self.accelerator.gather_for_metrics(values)
                return list(gathered)
            return values

        @torch.no_grad()
        def generate_until(
            self, requests: list[Instance], disable_tqdm: bool = False
        ) -> list[str]:
            res = []
            throughput, tps = [], []
            full_throughput, full_tps = [], []
            generation_speed = []
            latency = []
            input_length = []
            avg_step_count = []
            avg_generation_step_count = []
            avg_generated_token_count = []
            avg_decode_step_count = []
            avg_refresh_step_count = []
            avg_prefill_time_ms = []
            avg_step_time_ms = []
            avg_decode_step_time_ms = []
            avg_refresh_step_time_ms = []
            avg_refresh_step_fraction = []
            think_end_token = getattr(self, "think_end_token", None)

            def _collate(req: tuple[str, dict]):
                toks = self.tok_encode(req[0])
                return -len(toks), req[0]

            pbar = tqdm(
                total=len(requests),
                disable=(disable_tqdm or (self.rank != 0)),
                desc="Running generate_until requests",
            )
            adaptive_batch_size = None
            if self.batch_size == "auto":
                print("Passed argument batch_size = auto. Detecting largest batch size")
                batch_size = self._detect_batch_size()
                print(f"Determined Largest batch size: {batch_size}")
                adaptive_batch_size = batch_size
            batch_size = (
                self.batch_size
                if self.batch_size != "auto"
                else adaptive_batch_size
                if adaptive_batch_size is not None
                else 0
            )
            batch_fn = (
                self._batch_scheduler
                if self.batch_size == "auto" and not adaptive_batch_size
                else None
            )

            re_ords = Collator(
                [reg.args for reg in requests],
                sort_fn=_collate,
                group_by="gen_kwargs",
                group_fn=lambda x: x[1],
            )
            chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk, strict=True)
                gen_kwargs = all_gen_kwargs[0]
                assert isinstance(gen_kwargs, dict), (
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
                kwargs = normalize_gen_kwargs(gen_kwargs, self.max_gen_toks)
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
                max_gen_toks = resolve_ar_max_new_tokens(
                    self.cfg,
                    kwargs,
                    task_name=getattr(self, "current_task_name", None),
                    default_max_gen_toks=self.max_gen_toks,
                )
                kwargs.pop("max_gen_toks", None)

                if self.backend not in {"causal", "seq2seq"}:
                    raise ValueError(f"Unsupported backend: {self.backend}")

                context_enc, attn_masks = self.tok_batch_encode(
                    contexts,
                    truncation=False,
                )
                context_enc = context_enc.to(self.device)
                attn_masks = attn_masks.to(self.device)

                kwargs.pop("max_length", None)
                temperature = kwargs.get(
                    "temperature", self.cfg.generation.get("temperature")
                )
                top_k = kwargs.get("top_k", self.cfg.generation.get("top_k"))
                top_p = kwargs.get("top_p", self.cfg.generation.get("top_p"))
                do_sample = bool(
                    kwargs.get("do_sample", False)
                    or (temperature is not None and temperature > 0)
                    or (top_k is not None and top_k > 0)
                    or (top_p is not None and top_p < 1)
                )

                if self.backend != "causal":
                    raise ValueError(
                        "Timed AR decode metrics currently require causal backend."
                    )

                with Timer("eval") as timer:
                    cont, timing_metrics = timed_causal_generate(
                        self.model,
                        context_enc,
                        attn_masks,
                        max_new_tokens=max_gen_toks,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        tokenizer=self.tokenizer,
                        stop_sequences=until,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                cont_toks_list = cont.tolist()
                for cont_toks, context, attn_mask in zip(
                    cont_toks_list, contexts, attn_masks, strict=True
                ):
                    if self.backend == "causal":
                        cont_toks = cont_toks[context_enc.shape[1] :]

                    if isinstance(think_end_token, int):
                        think_token_indices = [
                            i
                            for i, token in enumerate(cont_toks)
                            if token == think_end_token
                        ]
                        if think_token_indices:
                            cont_toks = cont_toks[think_token_indices[-1] + 1 :]

                    full_cont_toks = self._trim_padding(list(cont_toks))
                    eos_trimmed_toks = self._trim_to_eos(full_cont_toks)
                    throughput_tokens = len(eos_trimmed_toks)
                    full_new_tokens = len(full_cont_toks)

                    s = self.tok_decode(cont_toks)
                    if isinstance(think_end_token, int):
                        s = s.lstrip()
                    s = postprocess_generated_text(
                        generation=s,
                        stop=until,
                        think_end_token=(
                            think_end_token if isinstance(think_end_token, str) else None
                        ),
                    )
                    res.append(s)

                    input_length.append(int(attn_mask.sum().item()))
                    if timer.elapsed_time_s > 0:
                        throughput.append(throughput_tokens / timer.elapsed_time_s)
                        full_throughput.append(full_new_tokens / timer.elapsed_time_s)
                        latency.append(timer.elapsed_time_s)
                    effective_prefill_ms = timing_metrics["prefill_time_ms"]
                    decode_time_ms = timing_metrics["decode_time_ms"]
                    decode_step_count = timing_metrics["decode_step_count"]
                    aggregate_generation_time_s = (
                        timer.elapsed_time_ms - effective_prefill_ms
                    ) / 1000.0
                    if aggregate_generation_time_s > 0:
                        generation_speed.append(throughput_tokens / aggregate_generation_time_s)
                    avg_generation_step_count.append(float(decode_step_count))
                    avg_generated_token_count.append(float(throughput_tokens))
                    refresh_steps = 1.0
                    if throughput_tokens > 0:
                        total_step_count = decode_step_count + refresh_steps
                        avg_step_count.append(total_step_count)
                        avg_decode_step_count.append(decode_step_count)
                        avg_refresh_step_count.append(refresh_steps)
                        avg_prefill_time_ms.append(effective_prefill_ms)
                        avg_step_time_ms.append(
                            timer.elapsed_time_ms / total_step_count
                            if total_step_count > 0
                            else 0.0
                        )
                        avg_decode_step_time_ms.append(
                            decode_time_ms / decode_step_count
                            if decode_step_count > 0
                            else 0.0
                        )
                        avg_refresh_step_time_ms.append(
                            effective_prefill_ms if refresh_steps > 0 else 0.0
                        )
                        avg_refresh_step_fraction.append(
                            refresh_steps / total_step_count
                            if total_step_count > 0
                            else 0.0
                        )
                    else:
                        refresh_steps = 1.0 if self.backend == "causal" else 0.0
                        avg_step_count.append(refresh_steps)
                        avg_decode_step_count.append(0.0)
                        avg_refresh_step_count.append(refresh_steps)
                        avg_prefill_time_ms.append(
                            effective_prefill_ms if self.backend == "causal" else 0.0
                        )
                        avg_step_time_ms.append(
                            effective_prefill_ms if self.backend == "causal" else 0.0
                        )
                        avg_decode_step_time_ms.append(0.0)
                        avg_refresh_step_time_ms.append(
                            effective_prefill_ms if self.backend == "causal" else 0.0
                        )
                        avg_refresh_step_fraction.append(
                            1.0 if self.backend == "causal" else 0.0
                        )
                    tps.append(1.0 if throughput_tokens > 0 else 0.0)
                    full_tps.append(1.0 if full_new_tokens > 0 else 0.0)

                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                    pbar.update(1)

            res = re_ords.get_original(res)
            pbar.close()

            gathered_metrics = {
                "throughput": self._gather_for_metrics(throughput),
                "tps": self._gather_for_metrics(tps),
                "full_throughput": self._gather_for_metrics(full_throughput),
                "full_tps": self._gather_for_metrics(full_tps),
                "generation_speed": self._gather_for_metrics(generation_speed),
                "latency": self._gather_for_metrics(latency),
                "input_length": self._gather_for_metrics(input_length),
                "avg_step_count": self._gather_for_metrics(avg_step_count),
                "avg_generation_step_count": self._gather_for_metrics(
                    avg_generation_step_count
                ),
                "avg_generated_token_count": self._gather_for_metrics(
                    avg_generated_token_count
                ),
                "avg_q_part_step_count": self._gather_for_metrics(avg_decode_step_count),
                "avg_q_full_step_count": self._gather_for_metrics(
                    avg_refresh_step_count
                ),
                "init_time_ms": self._gather_for_metrics(avg_prefill_time_ms),
                "avg_step_time_ms": self._gather_for_metrics(avg_step_time_ms),
                "avg_q_part_step_time_ms": self._gather_for_metrics(
                    avg_decode_step_time_ms
                ),
                "avg_q_full_step_time_ms": self._gather_for_metrics(
                    avg_refresh_step_time_ms
                ),
                "avg_q_full_step_fraction": self._gather_for_metrics(
                    avg_refresh_step_fraction
                ),
            }

            if self.rank == 0:
                metric_values = {
                    name: (sum(values) / max(len(values), 1) if len(values) > 0 else None)
                    for name, values in gathered_metrics.items()
                }
                self.metrics.update(metric_values)
                assign_metric_values(
                    self,
                    {
                        **metric_values,
                        "total_time": Timer.get_cumulative_s("eval"),
                    },
                )

            return res

else:

    class HFAutoRegressiveEval:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "HFAutoRegressiveEval requires a newer lm_eval with HFLM helper utilities."
            )
