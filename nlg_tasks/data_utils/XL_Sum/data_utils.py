import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


"""
XLSum, a comprehensive and diverse dataset comprising 1.35 million professionally annotated article-summary pairs from BBC, 
extracted using a set of carefully designed heuristics. 
[https://github.com/csebuetnlp/xl-sum]
[https://huggingface.co/datasets/csebuetnlp/xlsum]

The dataset covers 45 languages ranging from low to high-resource, for many of which no public dataset is currently available. 
XL-Sum is highly abstractive, concise, and of high quality, as indicated by human and intrinsic evaluation.

45 languages = { amharic, arabic, azerbaijani, bengali, burmese, chinese_simplified, chinese_traditional, english, 
                 french, gujarati, hausa, hindi, igbo, indonesian, japanese, kirundi, korean, kyrgyz, marathi, nepali, 
                 oromo, pashto, persian, pidgin, portuguese, punjabi, russian, scottish_gaelic, serbian_cyrillic, 
                 serbian_latin, sinhala, somali, spanish, swahili, tamil, telugu, thai, tigrinya, turkish, ukrainian, 
                 urdu, uzbek, vietnamese, welsh, yoruba }
"""


def load_task_dataset():
    task_name = 'XL_Sum'
    data_dir = f"datasets/nlg_tasks/{task_name}/"
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n##### Loading the {task_name} dataset #####")
    print(f"Data path: {data_dir}\n")

    if len(os.listdir(data_dir)) != 0:
        total_dataset = json.load(open(os.path.join(data_dir, "xlsum_korean.json"), "r"))
    else:
        raw_dataset = load_dataset('csebuetnlp/xlsum', 'korean')

        total_dataset = dict()
        for mode in ['train', 'dev', 'test']:
            cur_mode_dataset = raw_dataset[mode] if mode != 'dev' else raw_dataset['validation']
            dataset = {'text': list(),
                       'summary': list()
                       }
            for line in cur_mode_dataset:
                dataset['text'].append(line['text'])
                dataset['summary'].append(line['summary'])

            assert len(dataset['text']) == len(dataset['summary'])

            total_dataset[mode] = dataset

        json.dump(total_dataset, open(os.path.join(data_dir, "xlsum_korean.json"), "w"), ensure_ascii=True, indent=2)

    return total_dataset


if __name__ == '__main__':
    dataset = load_task_dataset()

    from pretraining.scripts.run_pretraining import get_gpt2_tokenizer
    tokenizer = get_gpt2_tokenizer(tok_type="morphemeSubword",
                                   lang="ko",
                                   max_length=512,
                                   lowercase=True,
                                   clean_text=True,
                                   add_bos_token=False
                                   )


    text_lengths = [len(tokenizer.tokenize(text)) for text in tqdm(dataset['train']['text'])]
    summary_lengths = [len(tokenizer.tokenize(text)) for text in tqdm(dataset['train']['summary'])]

    print(f"train data context mean length: {np.mean(text_lengths):.2f}")
    print(f"train data summary mean length: {np.mean(summary_lengths):.2f}")

    print(f"train data context max length: {np.max(text_lengths):.2f}")
    print(f"train data summary max length: {np.max(summary_lengths):.2f}")
