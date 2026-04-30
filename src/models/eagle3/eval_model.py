import re
import accelerate
import torch

from datetime import timedelta
from typing import Iterable
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from omegaconf import DictConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm

from .ea_model import EaModel
from src.utils import Timer

STEP_METRIC_NAMES = [
    "avg_step_count",
    "avg_q_part_step_count",
    "avg_q_full_step_count",
    "init_time_ms",
    "avg_step_time_ms",
    "avg_q_part_step_time_ms",
    "avg_q_full_step_time_ms",
    "avg_q_full_step_fraction",
]


def assign_metric_values(target, metric_values: dict[str, float | None]) -> None:
    metrics = getattr(target, "metrics", None)
    for name, value in metric_values.items():
        setattr(target, name, value)
        if isinstance(metrics, dict):
            metrics[name] = value


try:
    from lm_eval.models.utils import handle_stop_sequences
except ImportError:

    def handle_stop_sequences(until, eos=None):  # type: ignore[no-redef]
        if until is None:
            return [eos] if eos else []
        if isinstance(until, str):
            return [until]
        items = list(until)
        if eos and eos not in items:
            items.append(eos)
        return items

try:
    from lm_eval.models.utils import normalize_gen_kwargs
except ImportError:

    def normalize_gen_kwargs(
        gen_kwargs: dict, default_max_gen_toks: int | None
    ) -> dict:
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


def has_bos_prefix(sequence: str, bos_str: str | Iterable[str] | None = None):
    if bos_str is None:
        return False
    if isinstance(bos_str, str):
        return sequence.startswith(bos_str)
    return any(sequence.startswith(x) for x in bos_str)


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_name)
    except AttributeError as e:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from e


def resolve_device_map(cfg: DictConfig, local_process_index: int):
    configured_device_map = cfg.model.get("device_map", None)
    if configured_device_map is not None:
        return configured_device_map
    if cfg.get("attn_implementation", None) == "flash_attention_2":
        if not torch.cuda.is_available():
            raise RuntimeError("flash_attention_2 requires CUDA, but no GPU is available.")
        return {"": local_process_index}
    return "auto"


class Eagle3Eval(TemplateLM):
    def __init__(self, cfg: DictConfig, **kwargs):
        accelerator_kwargs = accelerate.InitProcessGroupKwargs(
            timeout=timedelta(weeks=52)
        )
        self.accelerator = accelerate.Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.cfg = cfg
        self._metrics: dict[str, float | None] = {}
        self._device = self.accelerator.device
        device_map = resolve_device_map(cfg, self.accelerator.local_process_index)

        self.model = EaModel.from_pretrained(
            base_model_path=cfg.model.base_model_path,
            ea_model_path=cfg.model.ea_model_path,
            total_token=cfg.model.total_token,
            depth=cfg.model.depth,
            top_k=cfg.model.top_k,
            attn_implementation=cfg.get("attn_implementation", None),
            draft_sliding_window=cfg.model.get("draft_sliding_window", None),
            torch_dtype=parse_torch_dtype(cfg.model.dtype),
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        self.model.eval()
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.model.get_tokenizer()
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if cfg.model.get("padding_side"):
            self.tokenizer.padding_side = cfg.model.padding_side

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
        raise NotImplementedError("EAGLE3 evaluation currently supports generate_until only.")

    def _loglikelihood_tokens(self, requests, **kwargs):
        raise NotImplementedError("EAGLE3 evaluation currently supports generate_until only.")

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

    def _tokenize_context(
        self,
        context: str,
    ) -> dict[str, torch.Tensor]:
        add_special_tokens = {}
        bos_token = getattr(self.tokenizer, "bos_token", None)
        if has_bos_prefix(context, bos_token):
            add_special_tokens = {"add_special_tokens": False}
        elif self.cfg.get("add_bos_token") is not None:
            add_special_tokens = {"add_special_tokens": self.cfg.get("add_bos_token")}

        tokenized = self.tokenizer(
            [context],
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        return tokenized

    def _encode_stop_sequences(self, stop_sequences: list[str]) -> list[list[int]]:
        encoded: list[list[int]] = []
        for stop in stop_sequences:
            if not stop:
                continue
            token_ids = self.tokenizer.encode(stop, add_special_tokens=False)
            if token_ids:
                encoded.append(token_ids)
        return encoded

    @torch.no_grad()
    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False):
        out, throughput, tps = [], [], []
        full_throughput, full_tps = [], []
        generation_speed = []
        latency = []
        input_length = []
        avg_generation_step_count = []
        avg_generated_token_count = []
        step_metrics = {name: [] for name in STEP_METRIC_NAMES}

        for instance in tqdm(
            requests,
            total=len(requests),
            desc="Generating...",
            disable=disable_tqdm or not self.accelerator.is_main_process,
        ):
            context = instance.args[0]
            gen_kwargs = instance.args[1]

            default_max_gen_toks = (
                self.cfg.generation.get("gen_length")
                if self.cfg.generation.get("_cli_gen_length_provided", False)
                else 2048
            )
            is_longbench = str(self.cfg.dataset.name).startswith("longbench") or str(
                getattr(self, "current_task_name", "")
            ).startswith("longbench")
            kwargs = dict(gen_kwargs)
            if not is_longbench:
                kwargs["max_gen_toks"] = default_max_gen_toks
            kwargs = normalize_gen_kwargs(kwargs, default_max_gen_toks)
            eos = self.tokenizer.decode(self.eot_token_id, skip_special_tokens=False)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_new_tokens = kwargs.pop("max_gen_toks")

            temperature = kwargs.pop(
                "temperature", self.cfg.generation.get("temperature", 0.0)
            )
            top_k = kwargs.pop("top_k", self.cfg.generation.get("top_k"))
            top_p = kwargs.pop("top_p", self.cfg.generation.get("top_p"))

            tokenized = self._tokenize_context(context)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            stop_token_ids = self._encode_stop_sequences(until)
            runtime_max_length = max(
                int(self.cfg.model.max_length),
                int(tokenized["input_ids"].shape[1]) + int(max_new_tokens),
            )

            with Timer("eval") as timer:
                (
                    output_ids,
                    _raw_new_tokens,
                    step,
                    _accept_lengths,
                    generation_metrics,
                ) = self.model.eagenerate(
                    torch.as_tensor(tokenized["input_ids"]).to(self.device),
                    temperature=temperature,
                    top_p=top_p or 0.0,
                    top_k=top_k or 0.0,
                    max_new_tokens=max_new_tokens,
                    max_length=runtime_max_length,
                    stop_token_ids=stop_token_ids,
                    log=True,
                )

            generated_ids = output_ids[0][len(tokenized["input_ids"][0]) :]
            eos_trimmed_ids = self._trim_to_eos(generated_ids)
            throughput_tokens = int(eos_trimmed_ids.numel())
            full_new_tokens = (
                throughput_tokens
                if self.cfg.generation.get("stop_until_eos", False)
                else int(generated_ids.numel())
            )
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            generated_text = postprocess_generated_text(
                generation=generated_text,
                stop=until,
                think_end_token=None,
            )

            out.append(generated_text)
            input_length.append(torch.sum(tokenized["attention_mask"]).item())

            if timer.elapsed_time_s > 0:
                throughput.append(throughput_tokens / timer.elapsed_time_s)
                full_throughput.append(full_new_tokens / timer.elapsed_time_s)
                latency.append(timer.elapsed_time_s)
            generation_time_s = None
            prefill_ms = generation_metrics.get("init_time_ms")
            if prefill_ms is not None:
                generation_time_s = (timer.elapsed_time_ms - float(prefill_ms)) / 1000.0
                if generation_time_s > 0:
                    generation_speed.append(throughput_tokens / generation_time_s)
            avg_generation_step_count.append(float(step or 0))
            avg_generated_token_count.append(float(throughput_tokens))

            for name, value in generation_metrics.items():
                if name in step_metrics and value is not None:
                    step_metrics[name].append(value)

            if step and int(step) > 0:
                tps.append(throughput_tokens / int(step))
                full_tps.append(full_new_tokens / int(step))

        gathered_metrics = {
            "throughput": self.accelerator.gather_for_metrics(throughput),
            "tps": self.accelerator.gather_for_metrics(tps),
            "full_throughput": self.accelerator.gather_for_metrics(full_throughput),
            "full_tps": self.accelerator.gather_for_metrics(full_tps),
            "generation_speed": self.accelerator.gather_for_metrics(generation_speed),
            "latency": self.accelerator.gather_for_metrics(latency),
            "input_length": self.accelerator.gather_for_metrics(input_length),
            "avg_generation_step_count": self.accelerator.gather_for_metrics(
                avg_generation_step_count
            ),
            "avg_generated_token_count": self.accelerator.gather_for_metrics(
                avg_generated_token_count
            ),
        }
        gathered_step_metrics = {
            name: self.accelerator.gather_for_metrics(values)
            for name, values in step_metrics.items()
        }

        if self.accelerator.is_main_process:
            metric_values = {
                name: (sum(values) / max(len(values), 1) if len(values) > 0 else None)
                for name, values in gathered_metrics.items()
            }
            metric_values["total_time"] = Timer.get_cumulative_s("eval")
            metric_values.update(
                {
                    name: (sum(values) / max(len(values), 1) if len(values) > 0 else None)
                    for name, values in gathered_step_metrics.items()
                }
            )
            assign_metric_values(self, metric_values)

        return out
