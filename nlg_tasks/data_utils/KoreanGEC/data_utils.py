import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


"""

"""


def load_task_dataset():
    task_name = 'KoreanGEC'
    data_dir = f"datasets/nlg_tasks/{task_name}/"
    f_name = "korean_gec.json"
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n##### Loading the {task_name} dataset #####")
    print(f"Data path: {data_dir}\n")

    if os.path.exists(os.path.join(data_dir, f_name)):
        total_dataset = json.load(open(os.path.join(data_dir, f_name), "r"))
    else:
        total_dataset = dict()
        for subset in ['korean_learner', 'native', 'union']:
            total_dataset[task_name + "_" + subset] = dict()
            for mode in ['train', 'val', 'test']:
                raw_dataset = open(os.path.join(data_dir, f"{subset}/{subset}_{mode}.txt"), "r", encoding="utf-8").readlines()
                raw_dataset = [[l.strip() for l in line.strip().split('\t')] for line in raw_dataset]

                if mode == 'val':
                    mode = 'dev'

                dataset = {'wrong_text': list(),
                           'correct_text': list()
                           }
                for (sen1, sen2) in raw_dataset:
                    dataset['wrong_text'].append(sen1)
                    dataset['correct_text'].append(sen2)

                assert len(dataset['wrong_text']) == len(dataset['correct_text'])

                total_dataset[task_name + "_" + subset][mode] = dataset

        json.dump(total_dataset, open(os.path.join(data_dir, f_name), "w"), ensure_ascii=True, indent=2)

    return total_dataset


if __name__ == '__main__':
    dataset = load_task_dataset()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

    for subset in dataset:
        text_lengths = [len(tokenizer.tokenize(text)) for text in tqdm(dataset[subset]['test']['wrong_text'])]
        summary_lengths = [len(tokenizer.tokenize(text)) for text in tqdm(dataset[subset]['test']['correct_text'])]

        print(f"train data context mean length: {np.mean(text_lengths):.2f}")
        print(f"train data summary mean length: {np.mean(summary_lengths):.2f}")

        print(f"train data context max length: {np.max(text_lengths):.2f}")
        print(f"train data summary max length: {np.max(summary_lengths):.2f}")
