import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block, GPT2Attention
)
from transformers.models.bert.modeling_bert import (
    BertAttention, BertSelfAttention
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
        self.config = config
        attention_class = CustomGPT2Attention
        self.attn = attention_class(config=config, layer_idx=layer_idx, is_cross_attention=False)

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
        if self.config.embedding_norm:
            hidden_states = self.ln_2(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        # return outputs  # hidden_states, present, (attentions, cross_attentions)
        return outputs[0]  # hidden_states


class CustomBertSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super(CustomBertSelfAttention, self).__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        before_pooled_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            raise NotImplementedError
        elif is_cross_attention:
            raise NotImplementedError
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(before_pooled_states))
            value_layer = self.transpose_for_scores(self.value(before_pooled_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(before_pooled_states))
            value_layer = self.transpose_for_scores(self.value(before_pooled_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class CustomBertAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super(CustomBertAttention, self).__init__(config)
        self.self = CustomBertSelfAttention(
            config, position_embedding_type=position_embedding_type
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        hidden_states, before_pooled_states = hidden_states

        self_outputs = self.self(
            hidden_states,
            before_pooled_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # return outputs
        return outputs[0]  # hidden_states
