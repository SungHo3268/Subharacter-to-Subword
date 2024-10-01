import torch
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block, GPT2Attention,
)
from transformers.utils import logging as transformer_logging
from transformers.pytorch_utils import Conv1D

transformer_logger = transformer_logging.get_logger(__name__)


class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super(CustomGPT2Attention, self).__init__(config, is_cross_attention, layer_idx)

        self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        self.cv_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        # self.k_attn = Conv1D(self.embed_dim, self.embed_dim)
        # self.v_attn = Conv1D(self.embed_dim, self.embed_dim)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            before_pooled_states: Optional[Tuple[torch.FloatTensor]] = None,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("This function is not yet implemented. Please use the original GPT2Attention class if you want to use original encoder cross attention.")
            # if not hasattr(self, "q_attn"):
            #     raise ValueError(
            #         "If class is used as cross attention, the weights `q_attn` have to be defined. "
            #         "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
            #     )
            # query = self.q_attn(hidden_states)
            # key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            # attention_mask = encoder_attention_mask
        else:
            query = self.q_attn(hidden_states)
            # before_key = self.k_attn(before_pooled_states)
            # before_value = self.v_attn(before_pooled_states)
            before_key, before_value = self.cv_attn(before_pooled_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        before_key = self._split_heads(before_key, self.num_heads, self.head_dim)
        before_value = self._split_heads(before_value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            before_key = torch.cat((past_key, before_key), dim=-2)
            before_value = torch.cat((past_value, before_value), dim=-2)

        if use_cache is True:
            present = (before_key, before_value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, before_key, before_value, attention_mask, head_mask)
        else:
            # print("\n\n\n")
            # print(f"query.shape: {query.shape}")
            # print(f"before_key.shape: {before_key.shape}")
            # print(f"before_value.shape: {before_value.shape}")
            # print("\n\n\n")
            attn_output, attn_weights = self._attn(query, before_key, before_value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class CustomGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx):
        super(CustomGPT2Block, self).__init__(config, layer_idx)
        attention_class = CustomGPT2Attention
        self.attn = attention_class(config=config, layer_idx=layer_idx, is_cross_attention=True)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        hidden_states, before_pooled_states = hidden_states
        residual = hidden_states

        # hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            before_pooled_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]       # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            raise NotImplementedError("This function is not yet implemented. Please use the original GPT2Attention class if you want to use original encoder cross attention.")
            # # add one self-attention block for cross-attention
            # if not hasattr(self, "crossattention"):
            #     raise ValueError(
            #         f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
            #         "cross-attention layers by setting `config.add_cross_attention=True`"
            #     )
            # residual = hidden_states
            # hidden_states = self.ln_cross_attn(hidden_states)
            # cross_attn_outputs = self.crossattention(
            #     hidden_states,
            #     attention_mask=attention_mask,
            #     head_mask=head_mask,
            #     encoder_hidden_states=encoder_hidden_states,
            #     encoder_attention_mask=encoder_attention_mask,
            #     output_attentions=output_attentions,
            # )
            # attn_output = cross_attn_outputs[0]
            # # residual connection
            # hidden_states = residual + attn_output
            # outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        # Layer Normalization
        hidden_states = self.ln_2(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        # return outputs  # hidden_states, present, (attentions, cross_attentions)
        return outputs[0]  # hidden_states
