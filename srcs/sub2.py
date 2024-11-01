import os
import sys
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Config, GPTJConfig, GPTNeoXConfig, BertConfig
sys.path.append(os.getcwd())
from srcs.functions import trim_pad, repeat_interleave
from srcs.attn_pool import CustomGPT2Block, CustomBertAttention
from srcs.gptj_attn_pool import CustomGPTJBlock
from srcs.gptneox_attn_pool import CustomGPTNeoXLayer
from srcs.lora import LoRA_Config, LoRA_Layer, apply_lora_to_model


class Pooling(nn.Module):
    def __init__(self, pooling_module, pooling_size):
        super().__init__()
        self.pooling_module = pooling_module
        self.pooling_size = pooling_size

    def forward(self, x):
        before_pooled_states = rearrange(x, 'b n (k d) -> b (n k) d', k=self.pooling_size)
        after_pooled_states = self.pooling_module(x)
        return [after_pooled_states, before_pooled_states]


class SUB2_Config:
    def __init__(self, tok_type, reducer, hidden_dim, sub2_max_length, max_length,
                 do_combination, combination_type, trans_config, num_attention_heads, intermediate_size, num_trans_layers,
                 add_lora=False, is_bert=False, lora_config: LoRA_Config=None):
        self.tok_type = tok_type
        self.reducer = reducer
        self.hidden_dim = hidden_dim
        self.sub2_max_length = sub2_max_length
        self.max_length = max_length
        try:
            assert sub2_max_length % max_length == 0       # Ensure the sub2_max_length is divisible by max
        except AssertionError:
            print("\n\n\n")
            print(f"* sub2_max_length: {sub2_max_length}")
            print(f"* max_length: {max_length}")
            print(f"* sub2_max_length must be divisible by max_length.")
            print("\n\n\n")
        self.k = sub2_max_length // max_length

        self.do_combination = do_combination
        self.combination_type = combination_type
        self.trans_config = trans_config
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_trans_layers = num_trans_layers

        self.add_lora = add_lora
        if add_lora:
            assert lora_config is not None
        self.is_bert = is_bert
        self.lora_config = lora_config

class SUB2_Combination_Layer(nn.Module):
    def __init__(self, config: SUB2_Config, sub2_tokenizer, original_tokenizer):
        super(SUB2_Combination_Layer, self).__init__()
        self.config = config
        self.original_tokenizer = original_tokenizer

        self.sub2_tokenizer = sub2_tokenizer
        self.pad_token_id = sub2_tokenizer.pad_token_id
        self.eos_token_id = sub2_tokenizer.eos_token_id
        if config.is_bert:
            self.unk_token_id = sub2_tokenizer.unk_token_id
            self.sep_token_id = sub2_tokenizer.sep_token_id

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
        elif config.tok_type in ['jamo_var', 'jamo_distinct']:
            self.cho_len = 1
            self.joong_len = 1
            self.jong_len = 1
            self.max_jamo_len = max(self.cho_len, self.joong_len, self.jong_len)
            self.char_len = self.cho_len + self.joong_len + self.jong_len
        else:
            pass

        self.sub2_embedding = nn.Embedding(num_embeddings=len(sub2_tokenizer),
                                            embedding_dim=config.hidden_dim)

        if config.do_combination:
            if config.combination_type == 'gru':
                self.contextualization = nn.Sequential(
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
                    nn.Linear(config.sub2_max_length, config.max_length),
                    Rearrange('b d s -> b s d'),
                    nn.LayerNorm(config.hidden_dim)
                )
            elif config.reducer == 'linear_pool':           # Hourglass Transformer
                self.sequence_reducer = nn.Sequential(
                    Rearrange('b (n k) d -> b n k d', k=config.k),
                    Rearrange('b n k d -> b n (k d)'),
                    nn.Linear(config.k * config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim)
                )
            elif config.reducer == 'attention_pool':        # Funnel Transformer
                self.sequence_reducer = nn.Sequential(
                    Rearrange('b (n k) d -> b n k d', k=config.k),
                    Rearrange('b n k d -> b n (k d)'),
                    Pooling(nn.AvgPool1d(kernel_size=config.k,
                                         stride=config.k,
                                         count_include_pad=True
                                         ),
                            config.k
                            ),
                    CustomGPT2Block(config.trans_config, layer_idx=0),
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
        elif self.config.reducer == 'linear_pool':
            self.sequence_reducer[2].weight.data.normal_(mean=0.0, std=0.02)
            self.sequence_reducer[2].bias.data.zero_()
        elif self.config.reducer == 'attention_pool':
            pass
        else:
            raise NotImplementedError

    def forward(self, x, text_input):
        """
        :param x: input for SUB2 layer which size of (batch_size, jamo_seq_len(=sub2_max_length)), this is the input_ids of jamo.
        """

        sub2_embedding = self.sub2_embedding(x)       # (batch_size, jamo_seq_len(=sub2_max_length), hidden_dim) = (B, N_max_jamo, D)
        # print("\n")
        # print(f"1) sub2_embedding.shape: {sub2_embedding.shape}")
        if self.config.do_combination:
            batch_size = sub2_embedding.shape[0]
            if self.config.is_bert:
                x = x[:, 1:]                                    # Remove the [CLS] token
                sub2_cls_embedding = sub2_embedding[:, :1, :]
                sub2_embedding = sub2_embedding[:, 1:, :]       # Remove the [CLS] token
                # print(f"1.1) sub2_embedding.shape: {sub2_embedding.shape}")

                sep_token_idx = (x == self.sep_token_id) + (x == self.unk_token_id)
                repeat_num = torch.ones_like(x).to(x.device)
                repeat_num[sep_token_idx] = self.char_len

                # Calculate the context_input_ids too.
                x = repeat_interleave(x, repeats=repeat_num.detach().cpu(), dim=1)
                sub2_embedding = repeat_interleave(sub2_embedding, repeats=repeat_num.detach().cpu(), dim=1)
                x, sub2_embedding = trim_pad(x, sub2_embedding, pad_value=self.pad_token_id)

            x, sub2_embedding = trim_pad(x, sub2_embedding, pad_value=self.pad_token_id)      # (B, N_jamo, D)
            # print(f"2) sub2_embedding.shape: {sub2_embedding.shape}")

            try:
                assert x.shape[1] % self.char_len == 0
            except AssertionError:
                print("\n\n\n")
                print(text_input)
                print(x.shape)
                print(x)
                exit(-111)

            assert x.shape[1] == sub2_embedding.shape[1]

            # Get character representative input_ids
            x = x[:, ::self.char_len]

            # Get character representative embeddings
            """
            1) Contextualization
            """
            sub2_embedding = self.contextualization(sub2_embedding)[0]        # (B, N_jamo, D)
            # print(f"3) sub2_embedding.shape: {sub2_embedding.shape}")
            sub2_embedding = sub2_embedding.reshape(batch_size, -1, self.char_len, self.config.hidden_dim)        # # (B, N_jamo, D) -> (B, N_char, 3, D)
            # print(f"4) sub2_embedding.shape: {sub2_embedding.shape}")
            """
            2) Fusion of Chosung and Joongsung
            """
            cho_joong_inputs = torch.mean(sub2_embedding[:, :, :- self.jong_len], dim=2, keepdim=True)      # (B, N_char, 2, D)
            # print(f"5) cho_joong_inputs.shape: {cho_joong_inputs.shape}")
            jong_inputs = torch.mean(sub2_embedding[:, :, -self.jong_len:], dim=2, keepdim=True)        # (B, N_char, 1, D)
            # print(f"6) jong_inputs.shape: {jong_inputs.shape}")
            sub2_embedding = torch.concat([cho_joong_inputs, jong_inputs], dim=2)               # (B, N_char, 2, D)
            # print(f"7) sub2_embedding.shape: {sub2_embedding.shape}")             # (B, N_char, 2, D)

            """
            3) Addition of Jongsung (Rearrange & Conv)
            """
            sub2_embedding = rearrange(sub2_embedding, 'b n l d -> b l n d')      # (B, N_char, 2, D) -> (B, 2, N_char, D)
            sub2_embedding = self.add_jongsung(sub2_embedding).squeeze()          # (B, 2, N_char, D) -> (B, N_char, D)

            # print(f"8) sub2_embedding.shape: {sub2_embedding.shape}")
            if sub2_embedding.ndim == 2:           # it runs when the batch size is 1.
                if sub2_embedding.shape[0] == batch_size:
                    sub2_embedding = sub2_embedding.unsqueeze(1)
                else:
                    sub2_embedding = sub2_embedding.unsqueeze(0)

            # '''
            """
            4) Get original token representative embeddings (e.g., subword, morpheme, etc.)
            """
            sub2_embedding = sub2_embedding.type(torch.float32)
            sub2_embedding = self.get_org_shape_emb(sub2_embedding)[0]        # (B, N_char, D)

            char_seq_len = sub2_embedding.shape[1]
            # print(f"9) sub2_embedding.shape: {sub2_embedding.shape}")
            end_indices = []
            for text in text_input:
                if 'mGPT' in self.original_tokenizer.name_or_path:
                    tokens = []
                    continuous_token = []
                    for tok in self.original_tokenizer.encode(text):
                        if len(continuous_token) > 0:
                            continuous_token.append(tok)
                            decoded = self.original_tokenizer.decode(continuous_token)
                        else:
                            decoded = self.original_tokenizer.decode(tok)

                        if '�' in decoded:
                            if len(continuous_token) > 0:
                                pass
                            else:
                                continuous_token.append(tok)
                            continue
                        else:
                            continuous_token = []
                            tokens.append(decoded)
                else:
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

            sub2_embedding = [sub2_embedding[i, end_indices[i]] for i in range(batch_size)]   # (B, N_char, D) -> (B, N_subword(not consistent), D)
            sub2_embedding = pad_sequence(sub2_embedding, batch_first=True, padding_value=self.pad_token_id)   # (B, N_subword, D)
            # print(f"10) sub2_embedding.shape: {sub2_embedding.shape}")
            # Padding
            try:
                if self.config.is_bert:
                    sub2_embedding = torch.concat([
                        sub2_embedding,
                        torch.full(
                            size=(batch_size, self.config.max_length-1 - sub2_embedding.shape[1], self.config.hidden_dim),      # exclude the [CLS] token
                            fill_value=self.pad_token_id, device=x.device)
                    ], dim=1)  # (B, N_subword, D) -> (B, max_length, D)
                else:
                    sub2_embedding = torch.concat([
                        sub2_embedding,
                        torch.full(size=(batch_size, self.config.max_length-sub2_embedding.shape[1], self.config.hidden_dim), fill_value=self.pad_token_id, device=x.device)
                    ], dim=1)           # (B, N_subword, D) -> (B, max_length, D)
            except:
                for i in range(batch_size):
                    print(f"text_input[{i}]: {text_input[i]}")
                    print(f"end_indices[{i}]: {end_indices[i]}")
                exit(-111)
            # print(f"11) sub2_embedding.shape: {sub2_embedding.shape}")
            # '''
        else:
            if self.config.is_bert:
                sub2_cls_embedding = sub2_embedding[:, :1, :]
                sub2_embedding = sub2_embedding[:, 1:, :]       # Remove the [CLS] token
            sub2_embedding = self.sequence_reducer(sub2_embedding)        # (B, max_length, D)

        if self.config.is_bert:
            sub2_embedding = torch.cat([sub2_cls_embedding, sub2_embedding], dim=1)
        return sub2_embedding

class SUB2_LoRA_Layer(nn.Module):
    """
    Be careful with the name of weight parameters. Only the weight of SUB2 layer should have the name 'sub2_'.
    The name which has 'sub2_' will be trained. The other weights will be frozen.
    """
    def __init__(self, tokenizer, sub2_tokenizer, original_layer, config: SUB2_Config):
        super(SUB2_LoRA_Layer, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.sub2_tokenizer = sub2_tokenizer

        self.original_layer = original_layer
        if config.hidden_dim is None:
            config.hidden_dim = original_layer.weight.shape[1]

        self.sub2_combination = SUB2_Combination_Layer(config, sub2_tokenizer, tokenizer)

        if isinstance(config.trans_config, GPT2Config):
            self.sub2_injection = CustomGPT2Block(config.trans_config, layer_idx=0)
        elif isinstance(config.trans_config, GPTJConfig):
            self.sub2_injection = CustomGPTJBlock(config.trans_config)
        elif isinstance(config.trans_config, GPTNeoXConfig):
            self.sub2_injection = CustomGPTNeoXLayer(config.trans_config)
        elif isinstance(config.trans_config, BertConfig):
            self.sub2_injection = CustomBertAttention(config.trans_config)

        if config.add_lora:
            input_dim = config.hidden_dim                   # input dim size of lora A
            output_dim = original_layer.weight.shape[1]     # output dim size of lora B

            # Initialization
            lora_A_tensor = torch.empty(input_dim, config.lora_config.r)
            torch.nn.init.kaiming_uniform_(lora_A_tensor)
            self.sub2_lora_A = nn.Parameter(lora_A_tensor)

            lora_B_tensor = torch.zeros(config.lora_config.r, output_dim)
            self.sub2_lora_B = nn.Parameter(lora_B_tensor)

            self.scaling = config.lora_config.lora_alpha // config.lora_config.r

            if config.lora_config.lora_dropout > 0:
                self.dropout = nn.Dropout(p=config.lora_config.lora_dropout)
            else:
                self.dropout = lambda x: x  # pass

    def make_sub2_input(self, x, device):
        """
        :param x: original input text which is the string type.
        :return: sub2_x: input for SUB2 layer which size of (batch_size, jamo_seq_len)
        """
        sub2_encoded = self.sub2_tokenizer(x, padding="max_length", truncation=True, max_length=self.config.sub2_max_length, return_tensors='pt')
        sub2_x = sub2_encoded['input_ids'].to(device)
        return sub2_x

    def forward(self, x):
        device = x.device

        original_embedding = self.original_layer(x)

        if self.config.is_bert:
            text_input = self.tokenizer.batch_decode(x, skip_special_tokens=False)
            text_input = [text.replace("[CLS]", "").replace("[PAD]", "").strip() for text in text_input]
            # print("\n")
            # print(f"text_input[0]: {text_input[0]}")
        else:
            text_input = self.tokenizer.batch_decode(x, skip_special_tokens=True)

        sub2_x = self.make_sub2_input(text_input, device)                 # (B, N(=max_sub2_length), D)

        sub2_embedding = self.sub2_combination(sub2_x, text_input)       # (B, N(=max_length), D)

        if self.config.add_lora:
            # Apply dropout before the matrix multiplication
            A_dropout = self.dropout(self.sub2_lora_A)
            B_dropout = self.dropout(self.sub2_lora_B)
            W = self.scaling * A_dropout @ B_dropout
            sub2_embedding = F.linear(sub2_embedding, W.T)
        else:
            pass

        # For NLU tasks, the output dimension of the original layer and the SUB2 layer may be different.
        if original_embedding.shape[1] != sub2_embedding.shape[1]:
            sub2_embedding = sub2_embedding[:, :original_embedding.shape[1]]

        if self.config.is_bert:
            final_embedding = original_embedding + sub2_embedding
        else:
            final_embedding = self.sub2_injection([original_embedding, sub2_embedding.contiguous()])

        return final_embedding

    def __repr__(self):
        if self.config.do_combination:
            if self.config.add_lora:
                return (f'{self.__class__.__name__}(\n'
                        f'  (original_layer): {self.original_layer},\n'
                        f'  (sub2_embedding): Embedding({self.sub2_combination.sub2_embedding.weight.shape}),\n'
                        f'  (sub2_combination): SUB2_Combination_Layer(\n'
                        f'    (contextualization): {self.sub2_combination.contextualization},\n'
                        f'  ),\n'
                        f'  (sub2_lora_A): Parameter of size {self.sub2_lora_A.size()},\n'
                        f'  (sub2_lora_B): Parameter of size {self.sub2_lora_B.size()}\n'
                        f')'
                        )
            else:
                return (f'{self.__class__.__name__}(\n'
                        f'  (original_layer): {self.original_layer},\n'
                        f'  (sub2_embedding): Embedding({self.sub2_combination.sub2_embedding.weight.shape}),\n'
                        f'  (sub2_combination): SUB2_Combination_Layer('
                        f'    (contextualization): {self.sub2_combination.contextualization},\n'
                        f'  )\n'
                        f')'
                        )
        else:
            if self.config.reducer == 'linear':
                reducer_msg = f"(sequence_reducer): {self.sub2_combination.sequence_reducer}"
            elif self.config.reducer == 'linear_pool':
                reducer_msg = f"(sequence_reducer): {self.sub2_combination.sequence_reducer}"
            elif self.config.reducer == 'attention_pool':
                reducer_msg = f"(sequence_reducer): {self.sub2_combination.sequence_reducer}"
            else:
                raise NotImplementedError

            if self.config.add_lora:
                return (f'{self.__class__.__name__}(\n'
                        f'  (original_layer): {self.original_layer},\n'
                        f'  (sub2_embedding): Embedding({self.sub2_combination.sub2_embedding.weight.shape}),\n'
                        f'  {reducer_msg}\n'
                        f'  ),\n'
                        f'  (sub2_lora_A): Parameter of size {self.sub2_lora_A.size()},\n'
                        f'  (sub2_lora_B): Parameter of size {self.sub2_lora_B.size()}\n'
                        f')'
                        )
            else:
                return (f'{self.__class__.__name__}(\n'
                        f'  (original_layer): {self.original_layer},\n'
                        f'  (sub2_embedding): Parameter of size {self.sub2_combination.sub2_embedding.size()},\n'
                        f'  {reducer_msg}\n'
                        f'  )\n'
                        f')'
                        )


def apply_sub2_to_model(model, tokenizer, sub2_tokenizer, config: SUB2_Config, logger=None):
    print("\n")
    for name, module in model.named_modules():
        hierarchy = name.split('.')
        layer_name = hierarchy[-1]
        if len(hierarchy) > 1 and layer_name in ['wte', 'embed_in', 'word_embeddings']:    # Ensure the module is not the top-level module
            parent_module = model

            parent_names = hierarchy[:-1]
            for submodule_name in parent_names:  # Navigate to the parent module
                parent_module = getattr(parent_module, submodule_name)

            original_layer = getattr(parent_module, layer_name)
            if isinstance(original_layer, nn.Embedding):
                setattr(parent_module, layer_name, SUB2_LoRA_Layer(tokenizer, sub2_tokenizer, original_layer, config))
                if logger:
                    logger.info(f"Replaced {name} with SUB2_LoRA_Layer")
                else:
                    print(f"Replaced {name} with SUB2_LoRA_Layer")
    return model


def make_only_sub2_and_lora_as_trainable(model, weight: str = 'none', bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if weight == 'sub2_only':
            if 'sub2_' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        elif weight == 'lora_only':
            if 'lora_' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        elif weight == 'sub2_lora_only':
            if ('sub2_' in n) or ('lora_' in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

    if bias == 'none':
        pass
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'sub2_only':
        for m in model.modules():
            if isinstance(m, SUB2_Combination_Layer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, SUB2_LoRA_Layer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    elif bias == 'sub2_lora_only':
        for m in model.modules():
            if ((isinstance(m, SUB2_Combination_Layer) or isinstance(m, LoRA_Layer))
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
    from pretraining.scripts.run_gpt_pretraining import get_gpt2_tokenizer

    model_name = "skt/kogpt2-base-v2"

    tok_type = "jamo_var"
    sub2_max_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sub2_tokenizer = get_gpt2_tokenizer(tok_type=tok_type, lang="ko", max_length=sub2_max_length,
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
    sub2_config = SUB2_Config(
        tok_type=tok_type,
        reducer="linear",
        hidden_dim=768,
        sub2_max_length=sub2_max_length,
        max_length=256,
        do_combination=False,
        combination_type='gru',
        trans_config=trans_config,
        num_attention_heads=3,
        intermediate_size=3072,
        num_trans_layers=3,
        add_lora=True,
        is_bert=False,
        lora_config=lora_config
    )

    # Apply LoRA to the model
    model = apply_sub2_to_model(model, tokenizer, sub2_tokenizer, sub2_config)
    model = apply_lora_to_model(model, lora_config)
    make_only_sub2_and_lora_as_trainable(model, weight='sub2_lora_only', bias='sub2_lora_only')
    _, _ = print_trainable_parameters(model)