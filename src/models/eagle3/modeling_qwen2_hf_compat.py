from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2ForCausalLM as HFQwen2ForCausalLM,
    Qwen2Model as HFQwen2Model,
)


class Eagle3LegacyCacheAdapter(Cache):
    """Adapt Eagle3's legacy per-layer KVCache list to the HF Cache API."""

    def __init__(self, legacy_cache):
        super().__init__()
        self.legacy_cache = legacy_cache
        self.max_cache_len = legacy_cache[0][0].data.shape[2] if legacy_cache else 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self.legacy_cache[layer_idx]
        device = key_cache.data.device
        key_states = key_states.to(device)
        value_states = value_states.to(device)

        cache_position = None if cache_kwargs is None else cache_kwargs.get("cache_position")
        if cache_position is None:
            start = int(key_cache.current_length.item())
            cache_position = torch.arange(
                start,
                start + key_states.shape[2],
                device=device,
                dtype=torch.long,
            )
        else:
            cache_position = cache_position.to(device=device, dtype=torch.long)

        try:
            key_cache.data.index_copy_(2, cache_position, key_states)
            value_cache.data.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            key_cache.data[:, :, cache_position, :] = key_states
            value_cache.data[:, :, cache_position, :] = value_states

        new_length = max(
            int(key_cache.current_length.item()),
            int(cache_position[-1].item()) + 1,
        )
        key_cache.current_length.fill_(new_length)
        value_cache.current_length.fill_(new_length)

        return (
            key_cache.data[:, :, :new_length, :],
            value_cache.data[:, :, :new_length, :],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        return int(self.legacy_cache[layer_idx][0].current_length.item())

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len


class Eagle3CompatibleQwen2Model(HFQwen2Model):
    """Official HF Qwen2Model with Eagle3 tree-mask and legacy-cache support."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.tree_mask = None

    def _apply_tree_mask(self, causal_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if causal_mask is None or self.tree_mask is None:
            return causal_mask

        tree_mask = self.tree_mask.to(device=causal_mask.device)
        tree_len = tree_mask.size(-1)
        if tree_len == 0:
            return causal_mask

        masked = causal_mask.clone()
        min_value = torch.finfo(masked.dtype).min
        window = masked[:, :, -tree_len:, -tree_len:]
        window.masked_fill_(tree_mask == 0, min_value)
        return masked

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        causal_mask = super()._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=input_tensor,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )

        if self.tree_mask is None:
            return causal_mask

        if causal_mask is None:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else int(cache_position[-1].item()) + 1
            )
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask=attention_mask,
                sequence_length=input_tensor.shape[1],
                target_length=target_length,
                dtype=input_tensor.dtype,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
                config=self.config,
                past_key_values=past_key_values,
            )

        return self._apply_tree_mask(causal_mask)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        legacy_cache = None
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            legacy_cache = past_key_values
            past_key_values = Eagle3LegacyCacheAdapter(past_key_values)

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if attention_mask is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            attention_mask = torch.ones(
                (batch_size, past_seen_tokens + seq_length),
                dtype=torch.bool,
                device=device,
            )

        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

        num_layers = len(self.layers)
        selected_indices = (2, num_layers // 2, num_layers - 3)
        selected_hidden_states = tuple(outputs.hidden_states[idx] for idx in selected_indices)

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=legacy_cache if legacy_cache is not None else outputs.past_key_values,
            hidden_states=selected_hidden_states,
            attentions=outputs.attentions,
        )


class Eagle3CompatibleQwen2ForCausalLM(HFQwen2ForCausalLM):
    """Official HF Qwen2ForCausalLM with Eagle3-compatible model internals."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Eagle3CompatibleQwen2Model(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        del labels

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


Qwen2ForCausalLM = Eagle3CompatibleQwen2ForCausalLM
LlamaForCausalLM = Eagle3CompatibleQwen2ForCausalLM
