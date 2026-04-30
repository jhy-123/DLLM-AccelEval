import gc
import inspect
import os
import hydra
import json
import re
import traceback
import torch

from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import cast
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager

from src.utils import pre_initialize, Timer, sympy_antlr_patcher, load_eval_model

MODEL_METRIC_NAMES = [
    "tps",
    "throughput",
    "total_time",
    "full_tps",
    "full_throughput",
    "latency",
    "input_length",
    "avg_step_count",
    "avg_q_part_step_count",
    "avg_q_full_step_count",
    "avg_generation_step_count",
    "avg_generated_token_count",
    "avg_q_full_generated_token_count",
    "avg_q_part_generated_token_count",
    "avg_generation_generated_token_count",
    "generation_tps",
    "generation_speed",
    "init_time_ms",
    "avg_step_time_ms",
    "avg_q_part_step_time_ms",
    "avg_q_full_step_time_ms",
    "avg_generation_step_time_ms",
    "avg_q_full_step_fraction",
]


def serializer(o):
    if inspect.isfunction(o):
        try:
            source_code = inspect.getsource(o)
            return source_code
        except (TypeError, OSError):
            return f"<uninspectable function: {o.__name__}>"

    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return o.item()
        else:
            return o.tolist()

    return f"<unserializable object of type {o.__class__.__name__}>"


def overwrite_eval_task(cfg: DictConfig):
    eval_args = cast(dict, OmegaConf.to_container(cfg.eval_args, resolve=True))
    eval_args["task_manager"] = build_task_manager()
    return eval_args


def build_task_manager() -> TaskManager:
    return TaskManager(
        include_path=[
            str(path)
            for dirname in os.listdir(Path(__file__).parent / "tasks")
            if os.path.isdir(path := Path(__file__).parent / "tasks" / dirname)
        ]
    )


def find_tagged_subtasks(task_root: Path, tag_name: str) -> list[str]:
    task_names: list[str] = []
    for yaml_path in sorted(task_root.rglob("*.yaml")):
        lines = yaml_path.read_text(encoding="utf-8").splitlines()
        task_name = None
        tags: list[str] = []
        in_tag_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("task:"):
                task_name = stripped.split(":", 1)[1].strip()
                in_tag_block = False
            elif stripped == "tag:":
                in_tag_block = True
            elif in_tag_block and re.match(r"^-\s+", stripped):
                tags.append(re.sub(r"^-\s+", "", stripped))
            elif in_tag_block and stripped:
                in_tag_block = False

        if task_name and tag_name in tags:
            task_names.append(task_name)

    return task_names


def resolve_eval_tasks(cfg: DictConfig) -> list[str]:
    task_name = cfg.dataset.name
    tagged_subtasks = find_tagged_subtasks(Path(__file__).parent / "tasks", task_name)
    return tagged_subtasks or [task_name]


def find_task_yaml_path(task_root: Path, task_name: str) -> Path | None:
    for yaml_path in sorted(task_root.rglob("*.yaml")):
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("task:"):
                current_task_name = stripped.split(":", 1)[1].strip()
                if current_task_name == task_name:
                    return yaml_path
                break
    return None


def parse_yaml_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def read_task_metadata_bool(yaml_path: Path, key: str) -> bool | None:
    metadata_indent: int | None = None
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        if stripped == "metadata:":
            metadata_indent = indent
            continue

        if metadata_indent is None:
            continue

        if indent <= metadata_indent:
            metadata_indent = None
            continue

        if stripped.startswith(f"{key}:"):
            raw_value = stripped.split(":", 1)[1].strip()
            return parse_yaml_bool(raw_value)

    return None


def get_cli_apply_chat_template_override() -> bool | None:
    hydra_cfg = HydraConfig.get()
    override_prefixes = (
        "model.apply_chat_template=",
        "+model.apply_chat_template=",
        "++model.apply_chat_template=",
    )
    for override in hydra_cfg.overrides.task:
        if override.startswith(override_prefixes):
            raw_value = override.split("=", 1)[1].strip()
            return parse_yaml_bool(raw_value)
    return None


def _read_yaml_bool_key(yaml_path: Path, key: str) -> bool | None:
    cfg = OmegaConf.load(yaml_path)
    if not OmegaConf.is_dict(cfg) or key not in cfg:
        return None
    value = cfg.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return parse_yaml_bool(value)
    return None


@lru_cache(maxsize=None)
def _resolve_model_config_apply_chat_template(
    task_root_str: str, model_choice: str
) -> bool | None:
    model_root = Path(task_root_str)
    visited: set[Path] = set()

    def resolve_from_config(config_name: str) -> bool | None:
        yaml_path = model_root / f"{config_name}.yaml"
        if not yaml_path.exists() or yaml_path in visited:
            return None

        visited.add(yaml_path)
        cfg = OmegaConf.load(yaml_path)
        defaults = cfg.get("defaults", [])
        resolved_value: bool | None = None
        self_applied = False

        for default in defaults:
            if isinstance(default, str):
                if default == "_self_":
                    self_applied = True
                    current_value = _read_yaml_bool_key(yaml_path, "apply_chat_template")
                    if current_value is not None:
                        resolved_value = current_value
                    continue
                inherited_value = resolve_from_config(default)
                if inherited_value is not None:
                    resolved_value = inherited_value

        if not self_applied:
            current_value = _read_yaml_bool_key(yaml_path, "apply_chat_template")
            if current_value is not None:
                resolved_value = current_value

        return resolved_value

    return resolve_from_config(model_choice)


def get_model_config_apply_chat_template() -> bool | None:
    hydra_cfg = HydraConfig.get()
    model_choice = hydra_cfg.runtime.choices.get("model")
    if not model_choice:
        return None
    model_root = Path(__file__).parent / "configs" / "model"
    return _resolve_model_config_apply_chat_template(str(model_root), model_choice)


def resolve_apply_chat_template(cfg: DictConfig, task_name: str | None = None) -> bool:
    cli_override = get_cli_apply_chat_template_override()
    if cli_override is not None:
        return cli_override

    model_config_value = get_model_config_apply_chat_template()
    if model_config_value is not None:
        return model_config_value

    if task_name is not None:
        task_yaml_path = find_task_yaml_path(Path(__file__).parent / "tasks", task_name)
        if task_yaml_path is not None:
            task_apply_chat_template = read_task_metadata_bool(
                task_yaml_path, "apply_chat_template"
            )
            if task_apply_chat_template is not None:
                return task_apply_chat_template

    return True


def get_model_metric(model, name: str):
    if hasattr(model, name):
        return getattr(model, name)
    metrics = getattr(model, "metrics", None)
    if isinstance(metrics, dict):
        return metrics.get(name)
    return None


def format_metric(value, fmt: str = ".2f", suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:{fmt}}{suffix}"


def reset_model_metrics(model) -> None:
    model_metrics = getattr(model, "metrics", None)
    for name in MODEL_METRIC_NAMES:
        setattr(model, name, None)
        if isinstance(model_metrics, dict):
            model_metrics[name] = None


def append_model_metrics(results: dict, model, peak_memory_allocated: float) -> dict:
    results = results or {}
    model_metric_values = {name: get_model_metric(model, name) for name in MODEL_METRIC_NAMES}
    generation_tps = model_metric_values.get("generation_tps")
    generated_token_count = model_metric_values.get("avg_generated_token_count")
    generation_step_count = model_metric_values.get("avg_generation_step_count")
    if (
        generation_tps is None
        and generated_token_count is not None
        and generation_step_count is not None
        and generation_step_count > 0
    ):
        model_metric_values["generation_tps"] = (
            generated_token_count / generation_step_count
        )
    if model_metric_values.get("avg_generation_step_time_ms") is None:
        model_metric_values["avg_generation_step_time_ms"] = model_metric_values.get(
            "avg_q_part_step_time_ms"
        )

    has_model_metrics = False
    for name in MODEL_METRIC_NAMES:
        value = model_metric_values.get(name)
        if value is not None:
            has_model_metrics = True
            break
    if has_model_metrics:
        results["model_metrics"] = dict(model_metric_values)
        for name, value in model_metric_values.items():
            results[name] = value
    results["peak_memory_allocated_GB"] = peak_memory_allocated

    tps = model_metric_values.get("tps")
    throughput = model_metric_values.get("throughput")
    if tps is not None and throughput is not None:
        full_throughput = model_metric_values.get("full_throughput")
        full_tps = model_metric_values.get("full_tps")
        latency = model_metric_values.get("latency")
        total_time = model_metric_values.get("total_time")
        input_length = model_metric_values.get("input_length")
        init_time_ms = model_metric_values.get("init_time_ms")
        avg_step_time_ms = model_metric_values.get("avg_step_time_ms")
        avg_q_part_step_time_ms = model_metric_values.get("avg_q_part_step_time_ms")
        avg_q_full_step_time_ms = model_metric_values.get("avg_q_full_step_time_ms")
        avg_q_full_step_count = model_metric_values.get("avg_q_full_step_count")
        generation_tps = model_metric_values.get("generation_tps")
        generation_speed = model_metric_values.get("generation_speed")
        avg_generation_step_time_ms = model_metric_values.get(
            "avg_generation_step_time_ms"
        )
        logger.info(
            f"Throughput: {format_metric(throughput)} tokens/sec, "
            f"Tokens per step: {format_metric(tps)} tokens/step "
            f"(generation-only: {format_metric(generation_tps)} tokens/step, "
            f"{format_metric(avg_generation_step_time_ms)} ms/step, "
            f"{format_metric(generation_speed)} tokens/sec), "
            f"(full: {format_metric(full_throughput)} tokens/sec, "
            f"{format_metric(full_tps)} tokens/step), "
            f"Latency: {format_metric(latency)} s, "
            f"Total time: {format_metric(total_time)} s, "
            f"Avg input length: {format_metric(input_length)} tokens, "
            f"Init: {format_metric(init_time_ms)} ms, "
            f"Avg step: {format_metric(avg_step_time_ms)} ms, "
            f"Avg q_part step: {format_metric(avg_q_part_step_time_ms)} ms, "
            f"Avg q_full step: {format_metric(avg_q_full_step_time_ms)} ms "
            f"({format_metric(avg_q_full_step_count)} q_full steps), "
            f"Peak memory allocated: {peak_memory_allocated:.2f} GB"
        )
    return results


def write_results(path: str, results: dict) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=serializer)
    logger.info(f"Results saved to {path}")


def is_main_process(model) -> bool:
    accelerator = getattr(model, "accelerator", None)
    if accelerator is not None:
        return bool(accelerator.is_main_process)
    return int(getattr(model, "rank", 0)) == 0


def wait_for_everyone(model) -> None:
    accelerator = getattr(model, "accelerator", None)
    if accelerator is not None:
        accelerator.wait_for_everyone()


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    extra_cfg = pre_initialize(cfg)
    model = load_eval_model(cfg, extra_gen_kwargs=extra_cfg.get("extra_gen_kwargs"))
    output_dir = HydraConfig.get().runtime.output_dir

    patcher_ctx = sympy_antlr_patcher if cfg.dataset.name == "math-500" else nullcontext
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    use_cache = os.path.join(output_dir, "response") if cfg.use_eval_cache else None
    eval_args = overwrite_eval_task(cfg)
    with patcher_ctx():
        if cfg.dataset.get("per_task_results", False):
            combined_results = {}
            task_names = resolve_eval_tasks(cfg)
            logger.info(f"Running per-task evaluation for {len(task_names)} task(s)")
            for task_name in task_names:
                reset_model_metrics(model)
                model.current_task_name = task_name
                task_eval_args = dict(eval_args)
                task_eval_args["tasks"] = task_name
                task_use_cache = f"{use_cache}_{task_name}" if use_cache is not None else None
                logger.info(f"Evaluating task: {task_name}")
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                try:
                    task_results = simple_evaluate(
                        model=model,
                        use_cache=task_use_cache,
                        apply_chat_template=resolve_apply_chat_template(cfg, task_name),
                        **task_eval_args,
                    )
                except Exception as exc:
                    logger.exception(f"Task failed, continuing to next task: {task_name}")
                    task_results = {
                        "error": {
                            "task": task_name,
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    }
                    combined_results[task_name] = task_results
                    if is_main_process(model):
                        task_results_path = os.path.join(output_dir, f"results_{task_name}.json")
                        write_results(task_results_path, task_results)
                    wait_for_everyone(model)
                    model.current_task_name = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.ipc_collect()
                        except RuntimeError:
                            pass
                    continue

                peak_memory_allocated = (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                task_results = append_model_metrics(task_results, model, peak_memory_allocated)
                combined_results[task_name] = task_results

                if is_main_process(model):
                    task_results_path = os.path.join(output_dir, f"results_{task_name}.json")
                    write_results(task_results_path, task_results)

                wait_for_everyone(model)
                model.current_task_name = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except RuntimeError:
                        pass
                logger.info(f"Finished task: {task_name}")

            results = {
                "per_task_results": combined_results,
                "tasks": task_names,
            }
        else:
            task_name = eval_args.get("tasks") if isinstance(eval_args.get("tasks"), str) else None
            results = simple_evaluate(
                model=model,
                use_cache=use_cache,
                apply_chat_template=resolve_apply_chat_template(cfg, task_name),
                **eval_args,
            )

    peak_memory_allocated = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    results_path = os.path.join(output_dir, "results.json")

    if is_main_process(model):
        if not cfg.dataset.get("per_task_results", False):
            results = append_model_metrics(results, model, peak_memory_allocated)
        write_results(results_path, results or {})

        for timer in Timer.cumulative:
            logger.info(f"{timer} time: {Timer(timer).cumulative_s:.2f} seconds")


if __name__ == "__main__":
    main()
