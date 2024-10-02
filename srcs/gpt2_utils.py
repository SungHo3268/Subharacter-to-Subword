import os
import numpy as np
from future.utils.surrogateescape import encoded
from tqdm import tqdm


def check_pretraining_data(raw_data_path="datasets/pretraining/concatenated.txt", doc_split=True, toyset=False):
    # Load the dataset
    dataset_path = raw_data_path
    if doc_split:
        dataset_path = dataset_path.replace(".txt", "_doc.txt")
    if toyset:
        dataset_path = dataset_path.replace(".txt", "_toy.txt")

    if os.path.exists(dataset_path):
        with open(dataset_path, "rbU") as f:
            num_lines = sum(1 for _ in f)
        if doc_split:
            print("Number of documents in the dataset:", num_lines)
        else:
            print("Number of lines in the dataset:", num_lines)
        print("Dataset already exists.")

    else:
        print("Dataset does not exist. Preparing the dataset...")
        with open(raw_data_path, "rbU") as f:
            num_lines = sum(1 for _ in f)
        print("Number of lines(sentences) in the dataset:", num_lines)

        # Do Doc Split
        fr = open(dataset_path, "r")
        next_text = 0
        dataset = []
        texts = []
        for i in tqdm(range(num_lines), desc="Loading dataset...", bar_format="{l_bar}{bar:15}{r_bar}"):
            line = fr.readline().strip()
            if line == "":
                next_text += 1
                if next_text == 2:
                    next_text = 0
                    dataset.append(' '.join(texts))
                    texts = []
            else:
                texts.append(line)
            # if len(dataset) == 1000:
            #     break
        fr.close()

        with open(dataset_path.replace(".txt", "_doc.txt"), "w") as fw:
            for doc in dataset:
                fw.write(doc + "\n")
        with open(dataset_path.replace(".txt", "_doc_toy.txt"), "w") as fw:
            for doc in dataset[: 1000]:
                fw.write(doc + "\n")

        del dataset
    print("Dataset is ready.")


def doc_tokenization(doc, tokenizer, max_length):
    if (tokenizer.custom_tokenizer.config.name in ["jamo_var_info", "bts_units_var_info"] and
            max_length % tokenizer.trunc_num != 0):
        max_length = max_length - (max_length % tokenizer.trunc_num)

    if "text" in doc:
        outputs = tokenizer(doc["text"])  # {"input_ids": (# of docs, # of tokens), "attention_mask": (# of docs, # of tokens)}
    elif "sentence" in doc:
        outputs = tokenizer(doc["sentence"])  # {"input_ids": (# of docs, # of tokens), "attention_mask": (# of docs, # of tokens)}
    else:
        raise NotImplementedError

    trimmed_outputs = {key: [] for key in outputs}
    for key in outputs:
        for tokenized_doc_data in outputs[key]:     # tokenized_doc_data = (# of tokens, )
            doc_max_tokens = len(tokenized_doc_data) // max_length * max_length
            trimmed_outputs[key].extend(tokenized_doc_data[: doc_max_tokens])

    trimmed_outputs = {key: np.array(trimmed_outputs[key]).reshape(-1, max_length) for key in trimmed_outputs}
    return trimmed_outputs


def text_tokenization_for_classification(doc, tokenizer, max_length):
    if "text" in doc:
        outputs = tokenizer(doc["text"])  # {"input_ids": (# of batch(=doc), # of tokens), "attention_mask": (# of batch(=doc), # of tokens)}
    elif "sentence" in doc:
        outputs = tokenizer(doc["sentence"])  # {"input_ids": (# of batch(=doc), # of tokens), "attention_mask": (# of batch(=doc), # of tokens)}
    elif "sentence1" in doc and "sentence2" in doc:
        outputs = tokenizer(doc["sentence1"], doc["sentence2"])
    else:
        raise NotImplementedError

    # max_batch_seq_length = max([len(line) for line in outputs["input_ids"]])
    # max_batch_seq_length = max_length if max_batch_seq_length > max_length else max_batch_seq_length
    max_batch_seq_length = max_length

    trimmed_outputs = {key: [] for key in outputs}
    for key in outputs:
        for tokenized_doc_data in outputs[key]:  # tokenized_doc_data = (# of tokens, )
            if key == 'input_ids':
                tokenized_doc_data += [tokenizer.pad_token_id] * (max_batch_seq_length - len(tokenized_doc_data))
            elif key == 'attention_mask':
                tokenized_doc_data += [0] * (max_batch_seq_length - len(tokenized_doc_data))
            else:
                raise NotImplementedError
            trimmed_outputs[key].append(tokenized_doc_data[: max_batch_seq_length])

    trimmed_outputs = {key: np.array(trimmed_outputs[key]) for key in trimmed_outputs}
    trimmed_outputs["label"] = np.array(doc["label"])
    return trimmed_outputs


def text_tokenization_for_mc(doc, tokenizer, max_length):
    encoded_choices = []
    cls_token_location = []
    for i, line in enumerate(doc['choices']):
        line = [choice + ' ' + tokenizer.cls_token for choice in line]
        cur_encoded = []
        cur_cls_token = []
        for choice in line:
            encoded_choice = tokenizer.encode(choice, padding="max_length", truncation=True, max_length=max_length)
            cur_encoded.append(encoded_choice)
            cur_cls_token.append(encoded_choice.index(tokenizer.cls_token_id))
        encoded_choices.append(cur_encoded)
        cls_token_location.append(cur_cls_token)

    encoded_choices = np.array(encoded_choices)
    cls_token_location = np.array(cls_token_location)

    trimmed_outputs = {
        'input_ids': encoded_choices,
        'mc_token_ids': cls_token_location,
        'mc_labels': np.array(doc["label"])
    }
    return trimmed_outputs


def text_tokenization_for_casuallm(batch, tokenizer, max_length, max_new_tokens, task_name, mode):
    # if (tokenizer.custom_tokenizer.config.name in ["jamo_var_info", "bts_units_var_info"] and
    #         max_length % tokenizer.trunc_num != 0 and
    #         max_new_tokens % tokenizer.trunc_num != 0):
    #     max_length = max_length - (max_length % tokenizer.trunc_num)
    #     max_new_tokens = max_new_tokens - (max_new_tokens % tokenizer.trunc_num)

    if task_name == 'KoCommonGen':
        context = [', '.join(line.split('#')) for line in batch['morpheme_set']]
        target = batch['target']
        if mode == 'test':
            target = [' = '.join(line) for line in target]
        sep_ids = tokenizer('. ')['input_ids']
    elif task_name == 'XL_Sum':
        context = batch['text']
        target = batch['summary']
        sep_ids = tokenizer(" 요약: ")['input_ids']
    else:
        raise NotImplementedError

    encoded_context = tokenizer(context, max_length=max_length, truncation=True, padding=False)
    encoded_target = tokenizer(target, max_length=max_new_tokens, truncation=True, padding=False)

    if mode == 'train':
        encoded_inputs = {'input_ids': [], 'attention_mask': []}
        for key in encoded_inputs:
            for ctxt, tgt in zip(encoded_context[key], encoded_target[key]):
                if key == 'input_ids':
                    encoded_inputs[key].append(ctxt + sep_ids + tgt)
                elif key == 'attention_mask':
                    encoded_inputs[key].append(ctxt + [1]*len(sep_ids) + tgt)
                else:
                    raise NotImplementedError
    else:
        encoded_inputs = {'input_ids': [], 'attention_mask': []}
        for key in encoded_inputs:
            for ctxt in encoded_context[key]:
                if key == 'input_ids':
                    encoded_inputs[key].append(ctxt + sep_ids)
                elif key == 'attention_mask':
                    encoded_inputs[key].append(ctxt + [1]*len(sep_ids))
                else:
                    raise NotImplementedError

    max_batch_seq_length = max([len(line) for line in encoded_inputs["input_ids"]])
    # max_batch_seq_length = max_length if max_batch_seq_length > max_length else max_batch_seq_length

    trimmed_inputs = {key: [] for key in encoded_inputs}
    for key in encoded_inputs:
        for tokens in encoded_inputs[key]:  # tokenized_doc_data = (# of tokens, )
            if key == 'input_ids':
                tokens = [tokenizer.pad_token_id] * (max_batch_seq_length - len(tokens)) + tokens       # left side padding
            elif key == 'attention_mask':
                tokens = [0] * (max_batch_seq_length - len(tokens)) + tokens
            else:
                raise NotImplementedError
            trimmed_inputs[key].append(tokens)

    if mode == 'train':
        pass
    else:
        labels = tokenizer(target, max_length=max_new_tokens, truncation=True, padding=False)['input_ids']

        max_batch_label_length = max([len(line) for line in labels])
        for l, tokenized_label in enumerate(labels):
            tokenized_label += [-100] * (max_batch_label_length - len(tokenized_label))
            labels[l] = tokenized_label
        trimmed_inputs['labels'] = labels

    trimmed_inputs = {key: np.array(trimmed_inputs[key]) for key in trimmed_inputs}
    return trimmed_inputs
