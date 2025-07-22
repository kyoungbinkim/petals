import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
    Qwen3MLP,
    Qwen3RMSNorm,
    repeat_kv,
    rotate_half,
    apply_rotary_pos_emb,
    eager_attention_forward
)
from transformers.cache_utils import Cache, DynamicCache
from hivemind.utils import get_logger

from petals.utils.cuda_graphs import make_inference_graphed_callable

logger = get_logger(__name__)


class OptimizedQwen3Attention(Qwen3Attention):
    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.rotary_emb = Qwen3RotaryEmbedding(
            config
        )
        self.hidden_size = self.config.hidden_size
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_heads= self.config.num_attention_heads
        self._rotary_graph = None
        self.num_key_value_heads = config.num_key_value_heads

    def _optimized_apply_rotary(self, query_states, key_states, cos, sin):
        if self._rotary_graph is None:
            self._rotary_graph = make_inference_graphed_callable(
                apply_rotary_pos_emb, sample_args=(query_states, key_states, cos, sin)
            )
        return self._rotary_graph(query_states, key_states, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        assert not output_attentions
        
        bsz, q_len, _ = hidden_states.size()
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # 중복 계산 ...
        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + q_len, device=hidden_states.device
            ).unsqueeze(0)

        # 중복 계산
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            query_states, key_states = self._optimized_apply_rotary(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)        
        


        if past_key_value is not None:
            if key_states.shape[1] != past_key_value[0].shape[1]:
                key_states = key_states.view(bsz, self.num_key_value_heads, q_len, self.head_dim)
                value_states = value_states.view(bsz, self.num_key_value_heads, q_len, self.head_dim)
            
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training= self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output, None , past_key_value

class OptimizedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = OptimizedQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_attn_graph = None
        self.post_attn_graph = None
        self.layer_idx = layer_idx

    def _optimized_input_layernorm(self, hidden_states):
        if self.pre_attn_graph is None:
            self.pre_attn_graph = make_inference_graphed_callable(
                self.input_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.pre_attn_graph(hidden_states)

    def _optimized_post_attention_layernorm(self, hidden_states):
        if self.post_attn_graph is None:
            self.post_attn_graph = make_inference_graphed_callable(
                self.post_attention_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.post_attn_graph(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            hidden_states = self._optimized_input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            hidden_states = self._optimized_post_attention_layernorm(hidden_states)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class WrappedQwen3Block(OptimizedQwen3DecoderLayer):
    """
    Wrapper for Qwen3DecoderLayer that adapts it for use in Petals.
    This wrapper handles the differences in API between standard transformers and Petals.
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        start_time = time.time()
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past
                
        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            past_key_value = self._reorder_cache_from_bloom_to_qwen3(past_key_value, batch_size, past_key_values_length)

        assert position_ids is None

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
        )

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_embeddings = position_embeddings,
            **kwargs,
        )

        if use_cache:
            present_key_value = outputs[-1]
            present_key_value = self._reorder_cache_from_qwen3_to_bloom(
                present_key_value, batch_size, seq_length_with_past
            )
            outputs = outputs[:-1] + (present_key_value,)
        execution_time = time.time() - start_time
        logger.info(f"Qwen3{self.layer_idx} : {execution_time:.6f} 초")
        
        return outputs

    def _reorder_cache_from_bloom_to_qwen3(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        
        # Bloom format: [batch * num_heads, seq_len, head_dim]
        # Qwen3 format: [batch, num_heads, seq_len, head_dim]
        
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        
        return (key_states, value_states)

    def _reorder_cache_from_qwen3_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        
        # Qwen3 format: [batch, num_heads, seq_len, head_dim]
        # Bloom format: [batch * num_heads, seq_len, head_dim]
        
        key_states = key_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.permute(0, 2, 1)
        
        return (key_states, value_states)
    
    