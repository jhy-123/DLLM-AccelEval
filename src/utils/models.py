import torch
import transformers

from omegaconf import DictConfig
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel


def get_model_family(cfg: DictConfig) -> str:
    family = cfg.model.get("family")
    if family is not None:
        return str(family).lower()
    return str(cfg.model.name).split("-")[0].lower()


def parse_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    try:
        return getattr(torch, dtype_name)
    except AttributeError as e:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from e


def _load_dream_model(cfg: DictConfig, **model_kwargs) -> PreTrainedModel:
    from ..models import DreamConfig, DreamModel

    flash_attention = model_kwargs.pop("flash_attention", False)
    config = DreamConfig.from_pretrained(cfg.model.path)
    config.flash_attention = flash_attention
    return DreamModel.from_pretrained(cfg.model.path, config=config, **model_kwargs)


def load_pretrained_model(cfg: DictConfig, **model_kwargs) -> PreTrainedModel:
    """
    Load a pretrained model based on the configuration.
    """
    from ..models import LLaDAModelLM, DreamModel, Fast_dLLM_QwenForCausalLM

    model_family = get_model_family(cfg)
    if model_family == "llada":
        return LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "ultrallada":
        return LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dream":
        return _load_dream_model(cfg, **model_kwargs)
    elif model_family == "dparallel_llada":
        return LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dparallel_dream":
        return _load_dream_model(cfg, **model_kwargs)
    elif model_family == "fast_dllm_v2_7b" or model_family == "fast_dllm_v2_1.5b":
        flash_attention = model_kwargs.pop("flash_attention", False)
        model = Fast_dLLM_QwenForCausalLM.from_pretrained(cfg.model.path, **model_kwargs)
        resolved_attn_impl = "flash_attention_2" if flash_attention else cfg.get("attn_implementation", None)
        model.config.flash_attention = flash_attention
        model.config.attn_implementation = resolved_attn_impl
        model.config._attn_implementation = resolved_attn_impl
        return model
    elif model_family == "d2f_llada":
        print(f"Loading Base Model from: {cfg.model.path}")
        base_model = LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
        if hasattr(cfg.model, "lora_path") and cfg.model.lora_path:
            print(f"Loading D2F LoRA Adapter from: {cfg.model.lora_path}")
            model = PeftModel.from_pretrained(base_model, cfg.model.lora_path)
            model = model.merge_and_unload()
            return model
        else:
            raise ValueError("Config for 'd2f_llada' requires a valid path pointing to the LoRA weights.")
    elif model_family == "d2f_dream":
        print(f"Loading Base Model from: {cfg.model.path}")
        base_model = _load_dream_model(cfg, **model_kwargs)
        if hasattr(cfg.model, "lora_path") and cfg.model.lora_path:
            print(f"Loading D2F LoRA Adapter from: {cfg.model.lora_path}")
            model = PeftModel.from_pretrained(base_model, cfg.model.lora_path)
            model = model.merge_and_unload()
            return model
        else:
            raise ValueError("Config for 'd2f_dream' requires a valid path pointing to the LoRA weights.")

    raise ValueError(f"Unsupported pretrained model: {cfg.model.name}")


def load_eval_model(cfg: DictConfig, **model_kwargs):
    from ..models import (
        LLaDAEval,
        DreamEval,
        Eagle3Eval,
        AutoRegressiveEval,
        HFAutoRegressiveEval,
    )
    from ..models.eval_mdlm import EvalMDLM
    from ..models.ar.eval_model import HAS_HFLM_UTILS, load_ar_model_and_tokenizer
    import os

    model_family = get_model_family(cfg)
    if model_family == "llada":
        eval_model = LLaDAEval(cfg, **model_kwargs)
    elif model_family == "ultrallada":
        eval_model = LLaDAEval(cfg, **model_kwargs)
    elif model_family == "dream":
        eval_model = DreamEval(cfg, **model_kwargs)
    elif model_family == "dparallel_llada":
        eval_model = LLaDAEval(cfg, **model_kwargs)
    elif model_family == "dparallel_dream":
        eval_model = DreamEval(cfg, **model_kwargs)
    elif model_family == 'd2f_llada':
        eval_model = LLaDAEval(cfg, **model_kwargs)
    elif model_family == 'd2f_dream':
        eval_model = DreamEval(cfg, **model_kwargs)
    elif model_family == "fast_dllm_v2_7b" or model_family == "fast_dllm_v2_1.5b":
        eval_model = EvalMDLM(cfg, **model_kwargs)
    elif model_family == "eagle3":
        eval_model = Eagle3Eval(cfg, **model_kwargs)
    elif model_family == "ar":
        if not HAS_HFLM_UTILS:
            raise ImportError(
                "HFAutoRegressiveEval requires lm_eval HFLM helper utilities in the current environment."
            )
        local_process_index = int(os.environ.get("LOCAL_RANK", "0"))
        device = (
            torch.device(f"cuda:{local_process_index}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        pretrained_model, tokenizer = load_ar_model_and_tokenizer(
            cfg, device, local_process_index
        )
        eval_model = HFAutoRegressiveEval(
            pretrained=pretrained_model,
            tokenizer=tokenizer,
            revision=cfg.model.get("revision", "main"),
            batch_size=cfg.batch_size,
            max_length=cfg.get("max_length"),
            trust_remote_code=cfg.model.get("trust_remote_code", True),
            use_fast_tokenizer=cfg.model.get("use_fast", True),
            add_bos_token=cfg.get("add_bos_token"),
        )
        if cfg.model.get("padding_side") and hasattr(eval_model, "tokenizer"):
            if not eval_model.tokenizer.pad_token:
                eval_model.tokenizer.pad_token = eval_model.tokenizer.eos_token
            eval_model.tokenizer.padding_side = cfg.model.padding_side
        eval_model.cfg = cfg
    else:
        raise NotImplementedError(
            f"Model family {model_family} is not implemented for evaluation."
        )

    return eval_model


def load_tokenizer(cfg: DictConfig, **tokenizer_kwargs):

    # ---------------- Tokenizer loading ----------------
    tokenizer_kwargs["trust_remote_code"] = True
    model_family = get_model_family(cfg)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.path, **tokenizer_kwargs)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.model.get("padding_side"):
        tokenizer.padding_side = cfg.model.padding_side

    # ---------------- Model-specific customization ----------------
    # model_family = cfg.model.name.split("-")[0]
    match model_family:
        case "llada" | "dparallel_llada" | 'd2f_llada':
            tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
            tokenizer.eot_token = "<|eot_id|>"

            # fix bugs in chat template
            tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
        case "ultrallada":
            if tokenizer.mask_token != "<|mdm_mask|>":
                tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        case "dream" | "dparallel_dream" | 'd2f_dream':
            tokenizer.eot_token = "<|im_end|>"
            tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.eot_token
            )
        case "fast_dllm_v2_7b" | "fast_dllm_v2_1.5b":
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.pad_token or tokenizer.eos_token
            if tokenizer.mask_token is None:
                tokenizer.mask_token = "|<MASK>|"
            tokenizer.eot_token = tokenizer.eos_token
            tokenizer.eot_token_id = tokenizer.eos_token_id
    return tokenizer
