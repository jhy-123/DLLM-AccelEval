import gc
import os
import accelerate
import torch
import itertools

from typing import Iterable, cast
from datetime import timedelta
from omegaconf import DictConfig
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm
from loguru import logger

from src.frame import Frame
from src.utils import Timer, load_pretrained_model, load_tokenizer

STEP_METRIC_NAMES = [
    "avg_step_count",
    "avg_q_part_step_count",
    "avg_q_full_step_count",
    "avg_generation_step_count",
    "avg_generated_token_count",
    "avg_q_full_generated_token_count",
    "avg_q_part_generated_token_count",
    "avg_generation_generated_token_count",
    "generation_tps",
    "avg_step_time_ms",
    "avg_q_part_step_time_ms",
    "avg_q_full_step_time_ms",
    "avg_generation_step_time_ms",
    "avg_q_full_step_fraction",
    "init_time_ms",
]


def extract_step_metrics(decode_record, elapsed_time_ms: float) -> dict[str, float | None]:
    metrics = dict(getattr(decode_record, "metrics", {}) or {})

    step_count = metrics.get("avg_step_count")
    if step_count is None:
        final_frame = decode_record[-1]
        max_step = (
            int(final_frame.steps.max().item()) if final_frame.steps.numel() > 0 else -1
        )
        step_count = float(max_step + 1) if max_step >= 0 else 0.0
        metrics["avg_step_count"] = step_count

    if metrics.get("avg_step_time_ms") is None and step_count > 0:
        metrics["avg_step_time_ms"] = elapsed_time_ms / step_count

    if metrics.get("avg_generation_step_count") is None:
        metrics["avg_generation_step_count"] = max(step_count - 1.0, 0.0)
    if metrics.get("avg_generated_token_count") is None:
        final_token_seqs = decode_record[-1].generated_tokens
        metrics["avg_generated_token_count"] = float(final_token_seqs.size(-1))
    if metrics.get("avg_generation_generated_token_count") is None:
        metrics["avg_generation_generated_token_count"] = metrics[
            "avg_generated_token_count"
        ]
    generation_step_count = metrics.get("avg_generation_step_count")
    if metrics.get("generation_tps") is None:
        metrics["generation_tps"] = (
            metrics["avg_generation_generated_token_count"] / generation_step_count
            if generation_step_count and generation_step_count > 0
            else None
        )

    if metrics.get("avg_q_part_step_count") is None:
        metrics["avg_q_part_step_count"] = step_count
    if metrics.get("avg_q_full_step_count") is None:
        metrics["avg_q_full_step_count"] = 0.0
    q_part_step_count = metrics.get("avg_q_part_step_count")
    if (
        metrics.get("avg_q_part_step_time_ms") is None
        and q_part_step_count is not None
        and q_part_step_count > 0
    ):
        metrics["avg_q_part_step_time_ms"] = metrics["avg_step_time_ms"]
    if metrics.get("avg_q_full_step_fraction") is None:
        metrics["avg_q_full_step_fraction"] = (
            metrics["avg_q_full_step_count"] / step_count if step_count > 0 else 0.0
        )
    if metrics.get("avg_q_full_step_time_ms") is None:
        metrics["avg_q_full_step_time_ms"] = None
    if metrics.get("init_time_ms") is None:
        metrics["init_time_ms"] = metrics.get("avg_q_full_step_time_ms")
    return {name: metrics.get(name) for name in STEP_METRIC_NAMES}


def generated_tokens_per_sample(decode_record, until_eot: bool = True) -> list[float]:
    final_token_seqs = decode_record[-1].generated_tokens
    if final_token_seqs.numel() == 0:
        return [0.0 for _ in range(final_token_seqs.size(0))]

    if until_eot:
        eot_token_id = int(os.environ.get("EOT_TOKEN_ID", "126081"))
        eot_positions = (final_token_seqs == eot_token_id).int().argmax(dim=-1)
        token_counts = torch.where(
            eot_positions > 0,
            eot_positions,
            torch.full_like(eot_positions, final_token_seqs.size(-1)),
        )
        return [float(value) for value in token_counts.detach().cpu().tolist()]

    metrics = getattr(decode_record, "metrics", {}) or {}
    generated_count = metrics.get("avg_generated_token_count")
    if generated_count is not None and final_token_seqs.size(0) > 0:
        per_sample = float(generated_count) / float(final_token_seqs.size(0))
        return [per_sample for _ in range(final_token_seqs.size(0))]

    return [float(final_token_seqs.size(-1)) for _ in range(final_token_seqs.size(0))]


class EvalMDLM(TemplateLM):
    """
    Base class for evaluating masked denoising language models (MDLMs) using the LM Evaluation Harness.
    """

    def __init__(self, cfg: DictConfig, **kwargs):

        # setup facilities...
        accelerator_kwargs = accelerate.InitProcessGroupKwargs(
            timeout=timedelta(weeks=52)
        )
        self.accelerator = accelerate.Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.model = (
            load_pretrained_model(
                cfg,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
                flash_attention=cfg.get("flash_attention", False),
            )
            .eval()
            .to(self.accelerator.device)
        )
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = load_tokenizer(
            cfg, trust_remote_code=True
        )

        # setup properties from LM
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        # setup custom properties
        self.cfg = cfg
        self.throughput = None
        self.tps = None
        self.full_throughput = None
        self.full_tps = None
        self.latency = None
        self.total_time = None
        self.input_length = None
        self.avg_step_count = None
        self.avg_q_part_step_count = None
        self.avg_q_full_step_count = None
        self.avg_step_time_ms = None
        self.avg_q_part_step_time_ms = None
        self.avg_q_full_step_time_ms = None
        self.avg_q_full_step_fraction = None
        self.init_time_ms = None
        self._device = self.accelerator.device
        self.device = self.accelerator.device
        self.extra_gen_kwargs = kwargs.get("extra_gen_kwargs", {})

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        """
        Tokenize a string using the model's tokenizer and return a list of token IDs.
        NOTE: This method is expected to handle strings which already contain the BOS token (when add_special_tokens=None).
        Otherwise, will use add_special_tokens if specified.
        """
        add_special_tokens = add_special_tokens or self.cfg.get("add_bos_token")
        # set add_special_tokens=False if the string already starts with BOS token.
        if add_special_tokens is None and has_bos_prefix(
            string, self.tokenizer.decode(self.prefix_token_id)
        ):
            add_special_tokens = False
        if add_special_tokens is not None:
            kwargs["add_special_tokens"] = add_special_tokens
        return self.tokenizer.encode(string, **kwargs)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError

    def _loglikelihood_tokens(self, requests, **kwargs):
        raise NotImplementedError

    @property
    def eot_token_id(self) -> int:  # type: ignore
        try:
            return int(os.environ.get("EOT_TOKEN_ID"))  # type: ignore
        except TypeError:
            return self.tokenizer.eos_token_id  # type: ignore

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False):
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, gen_kwargs).
            context: str
                Context string
            gen_kwargs: dict
                A dictionary of keyword arguments to pass to the generation function e.g. top_k, until, etc.
        :return: list[str]
            A list of model generated continuations.
            continuation: str
                The generated continuation.
        """
        from src.generation import generate, decode_final_frame

        throughput, tps = [], []
        full_throughput, full_tps = [], []
        generation_speed = []
        latency = []
        input_length = []
        step_metrics = {name: [] for name in STEP_METRIC_NAMES}

        batch_size = self.cfg.get("batch_size", 1)

        def batched(iterable):
            """
            A quick implementation of itertools.batched for Python versions < 3.12.
            """
            it = iter(iterable)
            while batch := tuple(itertools.islice(it, batch_size)):
                yield batch

        def should_override_gen_length() -> bool:
            dataset_name = self.cfg.dataset.name
            task_name = getattr(self, "current_task_name", None)
            return (
                dataset_name.startswith("longbench")
                or (isinstance(task_name, str) and task_name.startswith("longbench"))
            )

        def get_instance_gen_length(instance: Instance) -> int:
            if self.cfg.generation.get("_cli_gen_length_provided", False):
                return int(self.cfg.generation.gen_length)
            if not should_override_gen_length():
                return int(self.cfg.generation.gen_length)
            _, gen_kwargs = instance.args
            return int(gen_kwargs.get("max_gen_toks", self.cfg.generation.gen_length))

        grouped_requests: list[list[tuple[int, Instance]]] = []
        current_group: list[tuple[int, Instance]] = []
        current_gen_length: int | None = None
        for idx, instance in enumerate(requests):
            gen_length = get_instance_gen_length(instance)
            if current_group and gen_length != current_gen_length:
                grouped_requests.append(current_group)
                current_group = []
            current_group.append((idx, instance))
            current_gen_length = gen_length
        if current_group:
            grouped_requests.append(current_group)

        ordered_outputs = [""] * len(requests)
        total_batches = sum(
            (len(group) + batch_size - 1) // batch_size for group in grouped_requests
        )
        task_name = getattr(self, "current_task_name", None)
        progress_desc = f"Generating {task_name}..." if task_name else "Generating..."
        progress = tqdm(
            total=total_batches,
            desc=progress_desc,
            disable=disable_tqdm or not self.accelerator.is_main_process,
        )

        for group in grouped_requests:
            for indexed_instances in batched(group):
                instance_indices, instances = zip(*indexed_instances)
                context, gen_kwargs = map(
                    list, zip(*(instance.args for instance in instances))
                )
                gen_length = get_instance_gen_length(instances[0])
                if self.cfg.model.name.startswith("fast_dllm_v2"):
                    if self.cfg.dataset.name.startswith("minerva_math") or self.cfg.dataset.name == "math-500":
                        context = [ctx.replace("Solution:", "Please reason step by step, and put your final answer within \\boxed{}.") for ctx in context]
                    elif self.cfg.dataset.name.startswith("gsm8k"):
                        context = [ctx.replace("Answer:", "Please reason step by step, and put your final answer after \n#### .") for ctx in context]
                if self.cfg.get("add_bos_token", False):
                    context = [
                        (
                            self.tokenizer.bos_token + ctx
                            if not has_bos_prefix(ctx, self.tokenizer.bos_token)
                            else ctx
                        )
                        for ctx in context
                    ]
                inputs = self.tokenizer(
                    context, return_tensors="pt", padding=True, padding_side="left"
                )
                until = [u["until"] for u in gen_kwargs]
                generation_kwargs = {
                    **dict(self.cfg.generation),
                    "gen_length": gen_length,
                    "steps": gen_length,
                }
                decode_record = None
                final_frame = None
                generated_answer = None
                try:
                    with Timer("eval") as timer:
                        decode_record = generate(
                            self.model,
                            **inputs,
                            **generation_kwargs,
                            **self.extra_gen_kwargs,
                            ignore_unknown_args="ignore",
                        )

                    current_batch_size = len(instances)
                    input_length.append(
                        torch.sum(inputs["attention_mask"]).item() / current_batch_size  # type: ignore
                    )
                    current_latency = timer.elapsed_time_s / current_batch_size
                    throughput_value = timer.token_per_second(decode_record)
                    full_throughput_value = timer.token_per_second(
                        decode_record, self.cfg.generation.stop_until_eot
                    )
                    throughput.append(throughput_value)
                    full_throughput.append(full_throughput_value)
                    tps.append(timer.token_per_step(decode_record))
                    full_tps.append(
                        timer.token_per_step(
                            decode_record, self.cfg.generation.stop_until_eot
                        )
                    )
                    latency.append(current_latency)
                    extracted_step_metrics = extract_step_metrics(
                        decode_record, timer.elapsed_time_ms
                    )
                    prefill_ms = (
                        extracted_step_metrics.get("init_time_ms")
                        or extracted_step_metrics.get("avg_q_full_step_time_ms")
                    )
                    if prefill_ms is not None:
                        generation_time_s = current_latency - (
                            float(prefill_ms) / 1000.0 / current_batch_size
                        )
                        if generation_time_s > 0:
                            for generated_tokens in generated_tokens_per_sample(
                                decode_record
                            ):
                                generation_speed.append(
                                    generated_tokens / generation_time_s
                                )
                    for name, value in extracted_step_metrics.items():
                        if value is not None:
                            step_metrics[name].append(value)
                    final_frame: Frame = decode_record[-1]
                    generated_answer = [
                        cast(
                            str,
                            decode_final_frame(
                                self.tokenizer,
                                final_frame[i],
                                stop_words=(
                                    u if not self.cfg.dataset.name == "humaneval" else None
                                ),
                                skip_special_tokens=True,
                            ),
                        )
                        for i, u in enumerate(until)
                    ]

                    for output_idx, answer in zip(instance_indices, generated_answer):
                        ordered_outputs[output_idx] = answer
                except torch.cuda.OutOfMemoryError:
                    task_name = getattr(self, "current_task_name", None) or "unknown"
                    if os.environ.get("D2CACHE_RERAISE_OOM", "0") == "1":
                        logger.warning(
                            "CUDA OOM on rank {} while evaluating task {} for instances {}. Re-raising due to D2CACHE_RERAISE_OOM=1.",
                            self.accelerator.local_process_index,
                            task_name,
                            list(instance_indices),
                        )
                        raise
                    logger.warning(
                        "CUDA OOM on rank {} while evaluating task {} for instances {}. Marking outputs as [out-of-memory] and clearing CUDA cache.",
                        self.accelerator.local_process_index,
                        task_name,
                        list(instance_indices),
                    )
                    for output_idx in instance_indices:
                        ordered_outputs[output_idx] = "[out-of-memory]"
                finally:
                    decode_record = None
                    final_frame = None
                    generated_answer = None
                    inputs = None
                    context = None
                    gen_kwargs = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.ipc_collect()
                        except RuntimeError:
                            pass
                    progress.update(1)

            # if you got a watchdog timeout error, you can uncomment this line to avoid it.
            # it will slow down the evaluation though.
            # self.accelerator.wait_for_everyone()

        progress.close()

        throughput = self.accelerator.gather_for_metrics(throughput)
        tps = self.accelerator.gather_for_metrics(tps)
        full_throughput = self.accelerator.gather_for_metrics(full_throughput)
        full_tps = self.accelerator.gather_for_metrics(full_tps)
        generation_speed = self.accelerator.gather_for_metrics(generation_speed)
        latency = self.accelerator.gather_for_metrics(latency)
        input_length = self.accelerator.gather_for_metrics(input_length)
        gathered_step_metrics = {
            name: self.accelerator.gather_for_metrics(values)
            for name, values in step_metrics.items()
        }

        if self.accelerator.is_main_process:
            self.tps = sum(tps) / len(tps) if len(tps) > 0 else None
            self.throughput = (
                sum(throughput) / len(throughput) if len(throughput) > 0 else None
            )
            self.full_tps = sum(full_tps) / len(full_tps) if len(full_tps) > 0 else None
            self.full_throughput = (
                sum(full_throughput) / len(full_throughput)
                if len(full_throughput) > 0
                else None
            )
            self.latency = sum(latency) / len(latency) if len(latency) > 0 else None
            self.generation_speed = (
                sum(generation_speed) / len(generation_speed)
                if len(generation_speed) > 0
                else None
            )
            self.total_time = Timer.get_cumulative_s("eval")
            self.input_length = (
                sum(input_length) / len(input_length) if len(input_length) > 0 else None
            )
            for name, values in gathered_step_metrics.items():
                setattr(
                    self,
                    name,
                    sum(values) / len(values) if len(values) > 0 else None,
                )

        return ordered_outputs

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        assert isinstance(chat_templated, str)
        return chat_templated


def has_bos_prefix(sequence: str, bos_str: str | Iterable[str] | None = None):
    if bos_str is None:
        return False
    elif isinstance(bos_str, str):
        return sequence.startswith(bos_str)
    else:
        return any(sequence.startswith(x) for x in bos_str)
