import os
import sys
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block, GPT2Model, GPT2PreTrainedModel, GPT2ForSequenceClassification, GPT2Attention,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
)
from transformers.utils import logging as transformer_logging
from transformers.pytorch_utils import Conv1D
from triton.ops import attention

transformer_logger = transformer_logging.get_logger(__name__)

sys.path.append(os.getcwd())
from srcs.functions import trim_pad
from srcs.lora import LoRA_Config, LoRA_Layer, apply_lora_to_model


class CustomGPT2ForSequenceClassification(GPT2ForSequenceClassification):
    def __init__(self, config):
        super(CustomGPT2ForSequenceClassification, self).__init__(config)
        self.transformer = CustomGPT2Model(config)


class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super(CustomGPT2Attention, self).__init__(config, is_cross_attention, layer_idx)

        self.cv_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        # self.q_attn = Conv1D(self.embed_dim, self.embed_dim)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            kombo_hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            kombo_key, kombo_value = self.cv_attn(kombo_hidden_states).split(self.split_size, dim=2)
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        kombo_key = self._split_heads(kombo_key, self.num_heads, self.head_dim)
        kombo_value = self._split_heads(kombo_value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
            kombo_attn_output, kombo_attn_weights = self._upcast_and_reordered_attn(query, kombo_key, kombo_value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            kombo_attn_output, kombo_attn_weights = self._attn(query, kombo_key, kombo_value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        kombo_attn_output = self._merge_heads(kombo_attn_output, self.num_heads, self.head_dim)
        kombo_attn_output = self.c_proj(kombo_attn_output)
        kombo_attn_output = self.resid_dropout(kombo_attn_output)

        attn_output = attn_output + kombo_attn_output
        attn_weights = attn_weights + kombo_attn_weights

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class CustomGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx):
        super(CustomGPT2Block, self).__init__(config, layer_idx)
        attention_class = CustomGPT2Attention
        self.attn = attention_class(config=config, layer_idx=layer_idx)
        if config.add_cross_attention:
            self.crossattention = attention_class(config=config, is_cross_attention=True, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        kombo_hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            kombo_hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class CustomGPT2Model(GPT2Model):
    def __init__(self, config):
        super(CustomGPT2Model, self).__init__(config)
        self.h = nn.ModuleList([CustomGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            if type(inputs_embeds) == tuple:
                kombo_hidden_states = inputs_embeds[1]
                kombo_attention_mask = inputs_embeds[2]
                inputs_embeds = inputs_embeds[0]

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif _use_sdpa:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=(batch_size, input_shape[-1]),
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=past_length,
                )
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                transformer_logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    kombo_hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class KOMBO_Config:
    def __init__(self, tok_type, reducer, hidden_dim, kombo_max_length, max_length,
                 do_combination, combination_type, trans_config, num_attention_heads, intermediate_size, num_trans_layers,
                 add_lora=False, lora_config: LoRA_Config=None):
        self.tok_type = tok_type
        self.reducer = reducer
        self.hidden_dim = hidden_dim
        self.kombo_max_length = kombo_max_length
        self.max_length = max_length

        self.do_combination = do_combination
        self.combination_type = combination_type
        self.trans_config = trans_config
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_trans_layers = num_trans_layers

        self.add_lora = add_lora
        if add_lora:
            assert lora_config is not None
        self.lora_config = lora_config

class KOMBO_Combination_Layer(nn.Module):
    def __init__(self, config: KOMBO_Config, kombo_tokenizer, original_tokenizer):
        super(KOMBO_Combination_Layer, self).__init__()
        self.config = config
        self.original_tokenizer = original_tokenizer

        self.kombo_tokenizer = kombo_tokenizer
        self.pad_token_id = kombo_tokenizer.pad_token_id
        self.eos_token_id = kombo_tokenizer.eos_token_id
        if config.tok_type == 'stroke_var':
            self.cho_len = 4
            self.joong_len = 1
            self.jong_len = 4
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        elif config.tok_type == 'cji_var':
            self.cho_len = 1
            self.joong_len = 5
            self.jong_len = 1
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        elif config.tok_type == 'bts_var':
            self.cho_len = 4
            self.joong_len = 5
            self.jong_len = 4
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        elif config.tok_type == 'jamo_var':
            self.cho_len = 1
            self.joong_len = 1
            self.jong_len = 1
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        else:
            pass

        self.kombo_embedding = nn.Embedding(num_embeddings=kombo_tokenizer.vocab_size,
                                            embedding_dim=config.hidden_dim)

        if config.do_combination:
            if config.combination_type == 'gru':
                self.contextualization = nn.Sequential(
                    # *nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_trans_layers)]),
                    nn.GRU(
                        input_size=config.hidden_dim,
                        hidden_size=config.hidden_dim,
                        num_layers=1,
                        batch_first=True,
                    )
                )
            elif config.combination_type == 'trans_gru':
                self.contextualization = nn.Sequential(
                    *nn.ModuleList([GPT2Block(config.trans_config, layer_idx=i) for i in range(config.num_trans_layers)]),
                    nn.GRU(
                        input_size=config.hidden_dim,
                        hidden_size=config.hidden_dim,
                        num_layers=1,
                        batch_first=True,
                    )
                )
            else:
                raise NotImplementedError

            self.add_jongsung = nn.Sequential(
                Rearrange('b l n d -> b d l n'),
                nn.Conv2d(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    kernel_size=(2, 1),
                    stride=1,
                    padding='same',
                    padding_mode='zeros',
                    groups=config.hidden_dim
                ),
                nn.AvgPool2d(
                    kernel_size=(2, 1),
                    stride=1,
                    count_include_pad=True,
                ),
                Rearrange('b d l n -> b l n d')
            )
            self.get_org_shape_emb = nn.GRU(
                    input_size=config.hidden_dim,
                    hidden_size=config.hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )

            self.init_combination_layer()
        else:
            if config.reducer == 'linear':
                self.sequence_reducer = nn.Sequential(
                    Rearrange('b s d -> b d s'),
                    nn.Linear(config.kombo_max_length, config.max_length),
                    Rearrange('b d s -> b s d'),
                    nn.LayerNorm(config.hidden_dim)
                )
            else:
                raise NotImplementedError

            self.init_reducer()

    def init_combination_layer(self):
        print("Init combination layer")
        if self.config.combination_type == 'gru':
            self.contextualization[0].weight_hh_l0.data.normal_(mean=0.0, std=0.02)
            self.contextualization[0].weight_ih_l0.data.normal_(mean=0.0, std=0.02)
            self.contextualization[0].bias_hh_l0.data.zero_()
            self.contextualization[0].bias_ih_l0.data.zero_()
        elif self.config.combination_type == 'trans_gru':
            self.contextualization[-1].weight_hh_l0.data.normal_(mean=0.0, std=0.02)
            self.contextualization[-1].weight_ih_l0.data.normal_(mean=0.0, std=0.02)
            self.contextualization[-1].bias_hh_l0.data.zero_()
            self.contextualization[-1].bias_ih_l0.data.zero_()
        else:
            raise NotImplementedError
        self.add_jongsung[1].weight.data.normal_(mean=0.0, std=0.02)
        self.add_jongsung[1].bias.data.zero_()
        self.get_org_shape_emb.weight_hh_l0.data.normal_(mean=0.0, std=0.02)
        self.get_org_shape_emb.weight_ih_l0.data.normal_(mean=0.0, std=0.02)
        self.get_org_shape_emb.bias_hh_l0.data.zero_()
        self.get_org_shape_emb.bias_ih_l0.data.zero_()

    def init_reducer(self):
        print("Init reducer")
        if self.config.reducer == 'linear':
            self.sequence_reducer[1].weight.data.normal_(mean=0.0, std=0.02)
            self.sequence_reducer[1].bias.data.zero_()
        else:
            raise NotImplementedError

    def forward(self, x, text_input):
        """
        :param x: input for KOMBO layer which size of (batch_size, jamo_seq_len(=kombo_max_length)), this is the input_ids of jamo.
        """
        kombo_embedding = self.kombo_embedding(x)       # (batch_size, jamo_seq_len(=kombo_max_length), hidden_dim) = (B, N_max_jamo, D)
        # print("\n")
        # print(f"1) kombo_embedding.shape: {kombo_embedding.shape}")
        if self.config.do_combination:
            batch_size = kombo_embedding.shape[0]
            x, kombo_embedding = trim_pad(x, kombo_embedding, pad_value=self.pad_token_id)      # (B, N_jamo, D)
            # print(f"2) kombo_embedding.shape: {kombo_embedding.shape}")

            # eos_token_idx = (x == self.eos_token_id)
            # repeat_num = torch.ones_like(x).to(x.device)
            # repeat_num[eos_token_idx] = self.char_len
            #
            # x = repeat_interleave(x, repeats=repeat_num.detach().cpu(), dim=1)
            # kombo_embedding = repeat_interleave(kombo_embedding, repeats=repeat_num.detach().cpu(), dim=1)
            # x, kombo_embedding = trim_pad(x, kombo_embedding, pad_value=self.pad_token_id)
            
            try:
                assert x.shape[1] % self.char_len == 0
            except AssertionError:
                print("\n\n\n")
                print(text_input)
                print(x.shape)
                print(x)
                exit(-111)

            assert x.shape[1] == kombo_embedding.shape[1]

            # Get character representative input_ids
            x = x[:, ::self.char_len]       # (B, N_jamo) -> (B, N_char)

            # Get character representative embeddings
            """
            1) Contextualization
            """
            kombo_embedding = self.contextualization(kombo_embedding)[0]        # (B, N_jamo, D)
            # print(f"3) kombo_embedding.shape: {kombo_embedding.shape}")
            kombo_embedding = kombo_embedding.reshape(batch_size, -1, self.char_len, self.config.hidden_dim)        # # (B, N_jamo, D) -> (B, N_char, 3, D)
            # print(f"4) kombo_embedding.shape: {kombo_embedding.shape}")
            """
            2) Fusion of Chosung and Joongsung
            """
            cho_joong_inputs = torch.sum(kombo_embedding[:, :, :- self.jong_len], dim=2, keepdim=True)      # (B, N_char, 3, D)
            # print(f"5) cho_joong_inputs.shape: {cho_joong_inputs.shape}")
            jong_inputs = torch.sum(kombo_embedding[:, :, -self.jong_len:], dim=2, keepdim=True)        # (B, N_char, 2, D)
            # print(f"6) jong_inputs.shape: {jong_inputs.shape}")
            kombo_embedding = torch.concat([cho_joong_inputs, jong_inputs], dim=2)               # (B, N_char, 2, D)
            # print(f"7) kombo_embedding.shape: {kombo_embedding.shape}")             # (B, N_char, 2, D)
            """
            3) Addition of Jongsung (Rearrange & Conv)
            """
            kombo_embedding = rearrange(kombo_embedding, 'b n l d -> b l n d')      # (B, N_char, 2, D) -> (B, 2, N_char, D)
            kombo_embedding = self.add_jongsung(kombo_embedding).squeeze()          # (B, 2, N_char, D) -> (B, N_char, D)
            # print(f"8) kombo_embedding.shape: {kombo_embedding.shape}")
            if kombo_embedding.ndim == 2:           # it runs when the batch size is 1.
                kombo_embedding = kombo_embedding.unsqueeze(0)

            """
            4) Get original token representative input_ids and embeddings (e.g., subword, morpheme, etc.)
            """
            kombo_embedding = kombo_embedding.type(torch.float32)
            kombo_embedding = self.get_org_shape_emb(kombo_embedding)[0]        # (B, N_char, D)

            char_seq_len = kombo_embedding.shape[1]
            # print(f"9) kombo_embedding.shape: {kombo_embedding.shape}")
            end_indices = []
            for text in text_input:
                tokens = self.original_tokenizer.tokenize(text)
                if len(tokens) != 0:
                    if (len(tokens) > 1) and (tokens[-1] == "▁"):
                        tokens = tokens[:-1]

                    tokens[0] = tokens[0].replace("▁", "")
                    if len(tokens[0]) == 0:
                        tokens = tokens[1:]

                end_idx = np.cumsum([len(word) for word in tokens]) - 1     # '-1' is for making index start from 0.
                end_idx = [idx for idx in end_idx if idx < char_seq_len]    # Remove the index which is out of the range.

                end_indices.append(end_idx)
            try:
                kombo_embedding = [kombo_embedding[i, end_indices[i]] for i in range(batch_size)]   # (B, N_char, D) -> (B, N_subword(not consistent), D)
            except RuntimeError:
                print("\n\n\n")
                print(f"batch_size: {batch_size}")
                print(f"len(end_indices): {len(end_indices)}")
                print(f"kombo_embedding.shape: {kombo_embedding.shape}")
                print(f"end_indices: {end_indices}")
                print(f"text_input: {text_input}")
                print("\n\n\n")
                exit(-111)

            # Get original representative input_ids
            x = [x[i][end_indices[i]] for i in range(batch_size)]
            x = pad_sequence(x, batch_first=True, padding_value=self.pad_token_id)

            kombo_embedding = pad_sequence(kombo_embedding, batch_first=True, padding_value=self.pad_token_id)   # (B, N_subword, D)
            # print(f"10) kombo_embedding.shape: {kombo_embedding.shape}")

            # Padding
            kombo_embedding = torch.concat([
                kombo_embedding,
                torch.full(size=(batch_size, self.config.max_length-kombo_embedding.shape[1], self.config.hidden_dim), fill_value=self.pad_token_id, device=x.device)
            ], dim=1)           # (B, N_subword, D) -> (B, max_length, D)
            # print(f"11) kombo_embedding.shape: {kombo_embedding.shape}")
        else:
            kombo_embedding = self.sequence_reducer(kombo_embedding)        # (B, max_length, D)

        attention_mask = ((x != self.pad_token_id) * 1).to(x.device)
        attention_mask = torch.concat([
            attention_mask,
            torch.full(size=(attention_mask.shape[0], self.config.max_length - attention_mask.shape[1]),
                       fill_value=self.pad_token_id, device=x.device)
        ], dim=1)

        assert kombo_embedding.shape[1] == attention_mask.shape[1]

        return kombo_embedding, attention_mask

class KOMBO_LoRA_Layer(nn.Module):
    """
    Be careful with the name of weight parameters. Only the weight of KOMBO layer should have the name 'kombo_'.
    The name which has 'kombo_' will be trained. The other weights will be frozen.
    """
    def __init__(self, tokenizer, kombo_tokenizer, original_layer, config: KOMBO_Config):
        super(KOMBO_LoRA_Layer, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.kombo_tokenizer = kombo_tokenizer

        self.original_layer = original_layer
        if config.hidden_dim is None:
            config.hidden_dim = original_layer.weight.shape[1]

        self.kombo_combination = KOMBO_Combination_Layer(config, kombo_tokenizer, tokenizer)

        if config.add_lora:
            input_dim = config.hidden_dim                   # input dim size of lora A
            output_dim = original_layer.weight.shape[1]     # output dim size of lora B

            # Initialization
            lora_A_tensor = torch.empty(input_dim, config.lora_config.r)
            torch.nn.init.kaiming_uniform_(lora_A_tensor)
            self.kombo_lora_A = nn.Parameter(lora_A_tensor)

            lora_B_tensor = torch.zeros(config.lora_config.r, output_dim)
            self.kombo_lora_B = nn.Parameter(lora_B_tensor)

            self.scaling = config.lora_config.lora_alpha // config.lora_config.r

            if config.lora_config.lora_dropout > 0:
                self.dropout = nn.Dropout(p=config.lora_config.lora_dropout)
            else:
                self.dropout = lambda x: x  # pass

    def make_kombo_input(self, x, device):
        """
        :param x: original input text which is the string type.
        :return: kombo_x: input for KOMBO layer which size of (batch_size, jamo_seq_len)
        """
        kombo_encoded = self.kombo_tokenizer(x, padding="max_length", truncation=True, max_length=self.config.kombo_max_length, return_tensors='pt')
        kombo_x = kombo_encoded['input_ids'].to(device)
        return kombo_x

    def forward(self, x):
        device = x.device

        original_embedding = self.original_layer(x)

        text_input = self.tokenizer.batch_decode(x, skip_special_tokens=True)

        kombo_x = self.make_kombo_input(text_input, device)                 # (B, N(=max_kombo_length), D)
        kombo_embedding, attention_mask = self.kombo_combination(kombo_x, text_input)       # (B, N(=max_length), D)

        if self.config.add_lora:
            # Apply dropout before the matrix multiplication
            A_dropout = self.dropout(self.kombo_lora_A)
            B_dropout = self.dropout(self.kombo_lora_B)
            W = self.scaling * A_dropout @ B_dropout
            kombo_embedding = F.linear(kombo_embedding, W.T)
        else:
            pass
        return original_embedding+kombo_embedding, kombo_embedding, attention_mask

    def __repr__(self):
        if self.config.do_combination:
            if self.config.add_lora:
                return (f'{self.__class__.__name__}(\n'
                        f'  (original_layer): {self.original_layer},\n'
                        f'  (kombo_embedding): Embedding({self.kombo_combination.kombo_embedding.weight.shape}),\n'
                        f'  (kombo_combination): KOMBO_Combination_Layer(\n'
                        f'    (contextualization): {self.kombo_combination.contextualization},\n'
                        f'  ),\n'
                        f'  (kombo_lora_A): Parameter of size {self.kombo_lora_A.size()},\n'
                        f'  (kombo_lora_B): Parameter of size {self.kombo_lora_B.size()}\n'
                        f')'
                        )
            else:
                return (f'{self.__class__.__name__}(\n'
                        f'  (original_layer): {self.original_layer},\n'
                        f'  (kombo_embedding): Embedding({self.kombo_combination.kombo_embedding.weight.shape}),\n'
                        f'  (kombo_combination): KOMBO_Combination_Layer('
                        f'    (contextualization): {self.kombo_combination.contextualization},\n'
                        f'  )\n'
                        f')'
                        )
        else:
            if self.config.reducer == 'linear':
                reducer_msg = f"(sequence_reducer): {self.kombo_combination.sequence_reducer}"
            else:
                raise NotImplementedError

            if self.config.add_lora:
                return (f'{self.__class__.__name__}(\n'
                        f'  (original_layer): {self.original_layer},\n'
                        f'  (kombo_embedding): Embedding({self.kombo_combination.kombo_embedding.weight.shape}),\n'
                        f'  {reducer_msg}\n'
                        f'  ),\n'
                        f'  (kombo_lora_A): Parameter of size {self.kombo_lora_A.size()},\n'
                        f'  (kombo_lora_B): Parameter of size {self.kombo_lora_B.size()}\n'
                        f')'
                        )
            else:
                return (f'{self.__class__.__name__}(\n'
                        f'  (original_layer): {self.original_layer},\n'
                        f'  (kombo_embedding): Parameter of size {self.kombo_combination.kombo_embedding.size()},\n'
                        f'  {reducer_msg}\n'
                        f'  )\n'
                        f')'
                        )


def apply_kombo_to_model(model, tokenizer, kombo_tokenizer, config: KOMBO_Config, logger=None):
    print("\n")
    for name, module in model.named_modules():
        hierarchy = name.split('.')
        layer_name = hierarchy[-1]
        if len(hierarchy) > 1 and layer_name in ['wte']:    # Ensure the module is not the top-level module
            parent_module = model

            parent_names = hierarchy[:-1]
            for submodule_name in parent_names:  # Navigate to the parent module
                parent_module = getattr(parent_module, submodule_name)

            original_layer = getattr(parent_module, layer_name)
            if isinstance(original_layer, nn.Embedding):
                setattr(parent_module, layer_name, KOMBO_LoRA_Layer(tokenizer, kombo_tokenizer, original_layer, config))
                if logger:
                    logger.info(f"Replaced {name} with KOMBO_LoRA_Layer")
                else:
                    print(f"Replaced {name} with KOMBO_LoRA_Layer")
    return model


def make_only_kombo_and_lora_as_trainable(model, weight: str = 'none', bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if weight == 'kombo_only':
            if 'kombo_' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        elif weight == 'lora_only':
            if 'lora_' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        elif weight == 'kombo_lora_only':
            if ('kombo_' in n) or ('lora_' in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

    if bias == 'none':
        pass
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'kombo_only':
        for m in model.modules():
            if isinstance(m, KOMBO_Combination_Layer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, KOMBO_LoRA_Layer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    elif bias == 'kombo_lora_only':
        for m in model.modules():
            if ((isinstance(m, KOMBO_Combination_Layer) or isinstance(m, LoRA_Layer))
                    and hasattr(m, 'bias') and m.bias is not None):
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
    import os
    import sys
    sys.path.append(os.getcwd())
    from srcs.lora import print_trainable_parameters
    from pretraining.scripts.run_pretraining import get_gpt2_tokenizer

    model_name = "skt/kogpt2-base-v2"

    tok_type = "jamo_var"
    kombo_max_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kombo_tokenizer = get_gpt2_tokenizer(tok_type=tok_type, lang="ko", max_length=kombo_max_length,
                                         lowercase=True, clean_text=True, add_bos_token=False,
                                         bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>", unk_token="<unk>")


    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Check the original number of parameters
    origin_num = sum(p.numel() for p in model.parameters())
    print("Original number of parameters:", origin_num)

    # Configuration for Transformer
    trans_config = AutoConfig.from_pretrained(model_name)

    # Configuration for LoRA
    lora_config = LoRA_Config(
        r=16,
        lora_alpha=64,
        lora_dropout=0.03,
        target_modules=["c_attn", "c_proj"],
    )
    kombo_config = KOMBO_Config(
        tok_type=tok_type,
        reducer="linear",
        hidden_dim=768,
        kombo_max_length=kombo_max_length,
        max_length=256,
        do_combination=False,
        combination_type='gru',
        trans_config=trans_config,
        num_attention_heads=3,
        intermediate_size=3072,
        num_trans_layers=3,
        add_lora=True,
        lora_config=lora_config
    )

    # Apply LoRA to the model
    model = apply_kombo_to_model(model, tokenizer, kombo_tokenizer, kombo_config)
    model = apply_lora_to_model(model, lora_config)
    make_only_kombo_and_lora_as_trainable(model, weight='kombo_lora_only', bias='kombo_lora_only')
    _, _ = print_trainable_parameters(model)
