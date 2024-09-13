# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""

import os
import sys
import collections
from typing import List, Optional, Tuple, Union, Dict
import regex as re
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging
sys.path.append(os.getcwd())
from tokenization.srcs.functions import get_tokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


class KoGPT2Tokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:


    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
            word just as any other word.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        custom_tokenizer,
        max_length,
        padding_side,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        add_bos_token=True,
        **kwargs,
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.add_bos_token = add_bos_token

        self.max_length = max_length

        self.padding_side = padding_side

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'."
            )
        self.encoder = self.load_vocab(vocab_file)        # tokens_to_ids
        self.decoder = collections.OrderedDict([(ids, tok) for tok, ids in self.encoder.items()])


        self.special_tokens_encoder = {
            self.pad_token: 0,
            self.unk_token: 0,
            self.bos_token: 0,
            self.eos_token: 0,
        }
        self.special_tokens_decoder: Dict[str, int] = {
            v: k for k, v in self.special_tokens_encoder.items()
        }
        self._num_special_tokens = len(self.special_tokens_encoder)


        self.custom_tokenizer = custom_tokenizer

        self.tok_config_name = self.custom_tokenizer.config.name
        if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
            self.space_symbol = self.custom_tokenizer.config["space_symbol"]
            self.empty_jamo = self.custom_tokenizer.config["empty_jamo_symbol"]

            self.space_symbol_id = self.encoder[self.space_symbol]
            self.empty_jamo_id = self.encoder[self.empty_jamo]

        if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
            self.trunc_num = self.custom_tokenizer.char_len
        else:
            self.trunc_num = 1


        # with open(vocab_file, encoding="utf-8") as vocab_handle:
        #     self.encoder = json.load(vocab_handle)
        # self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding

        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    @staticmethod
    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.strip().split("\t")[0]
            vocab[token] = index
        return vocab

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                bos_token_ids = [self.bos_token_id] * self.trunc_num
            else:
                bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text):
        """Tokenize a string."""
        tokens = self.custom_tokenizer.tokenize(text)
        return tokens

    def truncate_sequences(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            num_tokens_to_remove: int = 0,
            truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
            stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (`int`, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                The strategy to follow for truncation. Can be:

                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will truncate
                  token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
                  batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
                  than the model maximum admissible input size).
            stride (`int`, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of
            overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair
            of sequences (or a batch of pairs) is provided.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
                truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None
        ):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                    if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                        remain = len(ids) % self.trunc_num
                        if remain != 0:
                            ids = ids[remain:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                    if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                        remain = len(ids) % self.trunc_num
                        if remain != 0:
                            ids = ids[:-remain]
                else:
                    raise ValueError(f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'.")

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. "
                )
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                            error_msg
                            + "Please select another truncation strategy than "
                              f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            # logger.warning(
            #     "Be aware, overflowing tokens are not returned for the setting you have chosen,"
            #     f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
            #     "truncation strategy. So the returned list will always be empty even if some "
            #     "tokens have been removed."
            # )
            for _ in range(num_tokens_to_remove // self.trunc_num + 1):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if self.truncation_side == "right":
                        ids = ids[:-self.trunc_num]
                    elif self.truncation_side == "left":
                        ids = ids[self.trunc_num:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
                else:
                    if self.truncation_side == "right":
                        pair_ids = pair_ids[:-self.trunc_num]
                    elif self.truncation_side == "left":
                        pair_ids = pair_ids[self.trunc_num:]
                    else:
                        raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == "right":
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                    if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                        remain = len(pair_ids) % self.trunc_num
                        if remain != 0:
                            pair_ids = pair_ids[:-remain]
                elif self.truncation_side == "left":
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                    if self.tok_config_name in ['bts_units_var_info', 'jamo_var_info']:
                        remain = len(pair_ids) % self.trunc_num
                        if remain != 0:
                            pair_ids = pair_ids[remain:]
                else:
                    raise ValueError("invalid truncation strategy:" + str(self.truncation_side))
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    "for instance 'longest_first' or 'only_first'."
                )
        return (ids, pair_ids, overflowing_tokens)


    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length."""
        if tokens_b is None:
            while True:
                total_length = len(tokens_a)
                if total_length <= max_num_tokens:
                    break

                trunc_tokens = tokens_a
                assert len(trunc_tokens) >= 1

                # We want to sometimes truncate from the front and sometimes from the
                # back to add more randomness and avoid biases.
                if self.rng.random() < 0.5:
                    del trunc_tokens[: self.trunc_num]
                else:
                    del trunc_tokens[-self.trunc_num:]
        else:
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_num_tokens:
                    break

                trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                assert len(trunc_tokens) >= 1

                # We want to sometimes truncate from the front and sometimes from the
                # back to add more randomness and avoid biases.
                if self.rng.random() < 0.5:
                    del trunc_tokens[: self.trunc_num]
                else:
                    del trunc_tokens[-self.trunc_num:]

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self._added_tokens_decoder:
                return self._added_tokens_decoder[ids].content
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self._added_tokens_decoder:
                tokens.append(self._added_tokens_decoder[index].content)
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = self.custom_tokenizer.detokenize(tokens)
        return out_string

    def decode(
            self,
            token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
    ) -> str:

        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens)
        if (not skip_special_tokens) and (self.tok_config_name == 'jamo_var_info'):
            tokens_padded = []
            for tok in tokens:
                if tok in self.special_tokens_encoder:
                    tokens_padded.extend([tok] + [self.custom_tokenizer.empty_jamo] * (self.custom_tokenizer.char_len-1))
                else:
                    tokens_padded.append(tok)
            tokens = tokens_padded
        else:
            pass
        sequence = self.custom_tokenizer.detokenize(tokens)
        return sequence

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, "vocab.txt")
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    @property
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"


if __name__ == '__main__':
    tok_type = "morphemeSubword"
    tok_vocab_size = "32k"
    lang = "ko"
    max_length = 128

    if tok_type in ['subword', 'morphemeSubword']:
        tok_name = f"{tok_type}_{lang}_wiki_{tok_vocab_size}"
    else:
        tok_name = f"{tok_type}_{lang}_{tok_vocab_size}"

    custom_tokenizer = get_tokenizer(tok_type)
    if tok_type in ['subword', 'morphemeSubword']:
        custom_tokenizer.load_model(f"tokenization/resources/{tok_name}/tok.model")
    vocab_file = f"tokenization/resources/{tok_name}/tok.vocab"

    tokenizer = KoGPT2Tokenizer(vocab_file=vocab_file,
                                custom_tokenizer=custom_tokenizer,
                                max_length=max_length,
                                lowercase=True,
                                clean_text=True
                                )

    # text = "감사합니다."
    text = "한영 자판 상태에서 히라가나를 입력할 경우 ㄸ+한자 키를 누르면 않된다. 가타카나의 경우 장음 등 일부 문자를 제외하면, 꼭 ㅃ+한자 조합을 해야 한다. \t뒤에꺼는 테스트 문장입니다."

    text1 = "한영 자판 상태에서 히라가나를 입력할 경우 ㄸ+한자 키를 누르면 않된다. 가타카나의 경우 장음 등 일부 문자를 제외하면, 꼭 ㅃ+한자 조합을 해야 한다."
    text2 = "뒤에꺼는 테스트 문장입니다."

    print("[original inputs]")
    print(f"text: {text}")
    print(f"text1: {text1}")
    print(f"text2: {text2}")
    print("")

    print("[tokenize method]")
    print(f"- tokenizer.tokenize: {tokenizer.tokenize(text)}")
    print(f"- tokenizer.custom_tokenizer.tokenize{tokenizer.custom_tokenizer.tokenize(text)}")
    print("")

    print("[encode method]")
    print(f"- one input (text): {tokenizer(text).input_ids}")
    print(f"- two inputs (text1, text2): {tokenizer(text1, text2).input_ids}")
    print("")

    print("[encode method - w max_length]")
    print(f"- one input (text): {tokenizer(text, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids}")
    print(f"- two inputs (text1, text2): {tokenizer(text1, text2, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids}")
    print("")

    print("[decode method - w special tokens]")
    print(f"- one input (text):          {tokenizer.decode(tokenizer(text).input_ids)}")
    print(f"- two inputs (text1, text2): {tokenizer.decode(tokenizer(text1, text2).input_ids)}")
    print("")

    print("[decode method - w/o special tokens]")
    print(f"- one input (text):          {tokenizer.decode(tokenizer(text).input_ids, skip_special_tokens=True)}")
    print(f"- two inputs (text1, text2): {tokenizer.decode(tokenizer(text1, text2).input_ids, skip_special_tokens=True)}")
    print("")

    print("[decode method - w special tokens and max_length]")
    print(f"- one input (text):          {tokenizer.decode(tokenizer(text, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids)}")
    print(f"- two inputs (text1, text2): {tokenizer.decode(tokenizer(text1, text2, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids)}")
    print("")

    print("[decode method - w/o special tokens and max_length]")
    print(f"- one input (text):          {tokenizer.decode(tokenizer(text, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids, skip_special_tokens=True)}")
    print(f"- two inputs (text1, text2): {tokenizer.decode(tokenizer(text1, text2, truncation=True, padding='max_length', max_length=tokenizer.max_length).input_ids, skip_special_tokens=True)}")
    print("")
