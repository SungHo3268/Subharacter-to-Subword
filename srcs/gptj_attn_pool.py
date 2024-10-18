import torch
from typing import Optional, Tuple, Union
from transformers.models.gptj.modeling_gptj import (
    GPTJBlock, GPTJAttention, apply_rotary_pos_emb
)
from transformers.utils import logging as transformer_logging

from transformers.cache_utils import Cache
from transformers.utils import is_torch_fx_proxy
transformer_logger = transformer_logging.get_logger(__name__)



@torch.fx.wrap
def get_embed_positions(embed_positions, position_ids):
    return embed_positions.to(position_ids.device).repeat(position_ids.shape[0], 1, 1)

class CustomGPTJAttention(GPTJAttention):
    def __init__(self, config):
        super(CustomGPTJAttention, self).__init__(config)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            before_pooled_states: Optional[Tuple[torch.FloatTensor]] = None,
            layer_past: Optional[Cache] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(before_pooled_states)
        value = self.v_proj(before_pooled_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
            # The logic to conditionally copy to GPU could not be traced, so we do this
            # every time in the torch.fx case
            embed_positions = get_embed_positions(self.embed_positions, position_ids)
        elif position_ids is None:
            embed_positions = self.embed_positions
            # pass
        else:
            embed_positions = self._get_embed_positions(position_ids)

        # repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        # sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        # sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            # k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            # q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            # key = apply_rotary_pos_emb(key, sin, cos)
            # query = apply_rotary_pos_emb(query, sin, cos)
            pass

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            cache_kwargs = {
                # "sin": sin,
                # "cos": cos,
                "partial_rotation_size": self.rotary_dim,
                "cache_position": cache_position,
            }
            key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, layer_past)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class CustomGPTJBlock(GPTJBlock):
    def __init__(self, config):
        super(CustomGPTJBlock, self).__init__(config)
        self.config = config
        self.attn = CustomGPTJAttention(config=config)

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor],
            layer_past: Optional[Cache] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        hidden_states, before_pooled_states = hidden_states
        residual = hidden_states

        # hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states=hidden_states,
            before_pooled_states=before_pooled_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        # return outputs  # hidden_states, present, (attentions)
        return outputs[0]  # hidden_states
