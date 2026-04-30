import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from contextlib import contextmanager

from src.frame import Frame, FrameDelta
from src.cache.base import dCache, AttentionContext

class dKVCache(dCache):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        
        self.previous_ids: torch.Tensor | None = None
        self.active_q_mask: torch.Tensor | None = None 
        
        self._q_lens: list[int] = []
        self.active_token_idx = []    
        
        self._full_seq_len: int = 0
        self._q_position_ids: torch.Tensor | None = None
        self._kv_position_ids: torch.Tensor | None = None

    def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
        self.key_cache = []
        self.value_cache = []
        self.previous_ids = frame.generated_tokens.clone()
        
        P = frame.prompts.shape[1]
        B = block_mask.shape[0]
        L = P + block_mask.shape[1]
        
        # 🌟 修复核心 1：Block 第一步必须强制全量运算 (Prompt + 已生成的前面Block + 当前Block)
        self.active_q_mask = torch.ones((B, L), dtype=torch.bool, device=block_mask.device)
        self._is_first_step = True

    def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
        # 🌟 修复核心 2：保护 on_block_start 设置的全量 True Mask 不被第一步覆盖
        if getattr(self, '_is_first_step', False):
            self._is_first_step = False
            return
            
        need_reload = (self.active_q_mask is None)
        
        P = frame.prompts.shape[1]
        current_ids = frame.generated_tokens
        is_mask = (current_ids == self.mask_token_id)
        is_changed = (current_ids != self.previous_ids)
        gen_mask = is_mask | is_changed
        
        if self.model_config.model_type.lower() == "dream":
            q_gen = F.pad(is_mask[:, 1:], (0, 1), value=False)     #is_mask 移动一位，补 False
            q_full = F.pad(q_gen, (P, 0), value=False)           
            q_full[:, P - 1] |= is_mask[:, 0]
            q_full |= F.pad(is_changed, (P, 0), value=False)  # is_changed 
            self.active_q_mask = q_full
        else:    
        # 增量计算时，Prompt 和前面已固定的 Token 都是 False，只算变动的
            self.active_q_mask = F.pad(gen_mask, (P, 0), value=False)
        self.previous_ids = current_ids.clone()
            
        if need_reload:
            self.active_q_mask = None

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        with super().model_forward(x) as ctx:
            B_prime, L, D = x.shape
            self._full_seq_len = L
            
            self._kv_position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B_prime, -1)
            
            if self.active_q_mask is None:
                B_full = self.previous_ids.size(0) if self.previous_ids is not None else B_prime
                self.active_q_mask = torch.ones(B_full, L, dtype=torch.bool, device=x.device) 
                
            if self._active_seq_mask is not None:
                mask = self.active_q_mask[self._active_seq_mask]
            else:
                mask = self.active_q_mask

            x_active_list = []
            pos_active_list = []
            self._q_lens = [] 
            
            for b in range(B_prime):
                idx = mask[b].nonzero(as_tuple=False).squeeze(-1) 
                x_active_list.append(x[b, idx]) 
                pos_active_list.append(idx)
                self._q_lens.append(idx.numel()) 
                
            self.active_token_idx = pos_active_list
            Nq_max = max(self._q_lens) if self._q_lens else 0
        
            if Nq_max == 0:
                x_active_input = x[:, 0:0, :] 
                self._q_position_ids = x.new_zeros((B_prime, 0), dtype=torch.long)
            else:   
                x_active_input = pad_sequence(x_active_list, batch_first=True)
                q_pos = pad_sequence(pos_active_list, batch_first=True, padding_value=0) 
                self._q_position_ids = q_pos

            ctx.x = x_active_input 
            yield ctx
            
            if ctx.logits is None:
                raise RuntimeError("logits缺失")
                
            if ctx.logits is not None:
                full_logits = torch.full(
                    (B_prime, L, ctx.logits.shape[-1]),
                    torch.finfo(ctx.logits.dtype).min,
                    device=ctx.logits.device,
                    dtype=ctx.logits.dtype,
                )

                for b in range(B_prime):
                    n = self._q_lens[b]
                    if n > 0:
                        idx = self.active_token_idx[b]
                        full_logits[b, idx] = ctx.logits[b, :n]

                ctx.logits = full_logits

    @contextmanager
    def attention(
        self,
        layer_idx: int,
        x: torch.Tensor,
        attn_norm: nn.Module,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        with super().attention(
            layer_idx, x, attn_norm, q_proj, k_proj, v_proj, attention_mask, position_ids,
        ) as ctx:

            B_prime = ctx.q.size(0)
            
            if self._active_seq_mask is not None:
                batch_indices = torch.nonzero(self._active_seq_mask).squeeze(-1)
            else:
                batch_indices = list(range(B_prime))

            # 🌟 修复核心 3：无论当前算出来的 K 有多短，底层 Cache 的容器必须永远保持 L 全长！
            if len(self.key_cache) <= layer_idx:
                B_full = self.active_q_mask.size(0)
                cache_shape = (B_full, self._full_seq_len, ctx.k.size(2))
                
                self.key_cache.append(torch.zeros(cache_shape, dtype=ctx.k.dtype, device=ctx.k.device))
                self.value_cache.append(torch.zeros(cache_shape, dtype=ctx.v.dtype, device=ctx.v.device))
                
                # 第一轮全量计算，依然采用安全的按索引赋值
                for i, b_orig in enumerate(batch_indices):
                    update_idx = self.active_token_idx[i]
                    if update_idx.numel() > 0:
                        n = update_idx.numel()
                        self.key_cache[layer_idx][b_orig, update_idx, :] = ctx.k[i, :n, :]
                        self.value_cache[layer_idx][b_orig, update_idx, :] = ctx.v[i, :n, :]
            else:
                # 增量步更新
                for i, b_orig in enumerate(batch_indices):
                    update_idx = self.active_token_idx[i]
                    if update_idx.numel() > 0:
                        n = update_idx.numel()
                        self.key_cache[layer_idx][b_orig, update_idx, :] = ctx.k[i, :n, :]
                        self.value_cache[layer_idx][b_orig, update_idx, :] = ctx.v[i, :n, :]
            
            if layer_idx == 0:
                current_v_cache = self.value_cache[layer_idx][batch_indices] if self._active_seq_mask is not None else self.value_cache[layer_idx]
                self._attention_mask = AttentionContext.convert_attention_mask(
                    attention_mask,
                    dtype=ctx.k.dtype,
                    query_length=ctx.q.shape[1],
                    key_value_length=current_v_cache.shape[1],
                )
                
            ctx.q_position_ids = self._q_position_ids
            
            if self._active_seq_mask is not None:
                ctx.k = self.key_cache[layer_idx][batch_indices]
                ctx.v = self.value_cache[layer_idx][batch_indices]
            else:
                ctx.k = self.key_cache[layer_idx]
                ctx.v = self.value_cache[layer_idx]
                
            ctx.kv_position_ids = self._kv_position_ids
            ctx.attention_mask = getattr(self, '_attention_mask', None)
            
            yield ctx
