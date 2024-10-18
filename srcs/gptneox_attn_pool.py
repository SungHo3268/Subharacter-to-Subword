import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXLayer, GPTNeoXAttention
)
from transformers.utils import logging as transformer_logging

from transformers.cache_utils import Cache
from transformers.utils import is_torch_fx_proxy
transformer_logger = transformer_logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class CustomGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config):
        super(CustomGPTNeoXAttention, self).__init__(config)
        # self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.key_value = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=config.attention_bias)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            before_pooled_states: Optional[Tuple[torch.FloatTensor]] = None,
            attention_mask: torch.FloatTensor = None,
            position_ids: torch.LongTensor = None,
            head_mask: Optional[torch.FloatTensor] = None,
            layer_past: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            padding_mask: Optional[torch.Tensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            before_pooled_states=before_pooled_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _attn_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        before_pooled_states: Optional[Tuple[torch.FloatTensor]] = None,
        position_ids: torch.LongTensor = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            # cos, sin = self.rotary_emb(value, position_ids)
            pass
        else:
            cos, sin = position_embeddings
        # query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if layer_past is not None:
            cache_kwargs = {
                # "sin": sin,
                # "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)

        return query, key, value, layer_past


class CustomGPTNeoXLayer(GPTNeoXLayer):
    def __init__(self, config):
        super(CustomGPTNeoXLayer, self).__init__(config)
        self.config = config
        self.attention = CustomGPTNeoXAttention(config=config)

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor],
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            layer_past: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):

        hidden_states, before_pooled_states = hidden_states

        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            before_pooled_states=before_pooled_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        # return outputs  # hidden_states, present, (attentions)
        return outputs[0]  # hidden_states
