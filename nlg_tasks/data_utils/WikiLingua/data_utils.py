import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


"""
WikiLingua, 
a large-scale, multilingual dataset for the evaluation of cross-lingual abstractive summarization systems.

[https://huggingface.co/datasets/esdurmus/wiki_lingua]

Extracted article and summary pairs in 18 languages from WikiHow, a high quality, collaborative resource of 
how-to guides on a diverse set of topics written by human authors.

18 languages = { English, Spanish, Portuguese, French, German, Russian, Italian, Indonesian, Dutch, Arabic, Vietnamese, 
                 Chinese, Thai, Japanese, Korean, Hindi, Czech, Turkish }

Gold-standard article-summary alignments across languages by aligning the images that are used to describe each 
how-to step in an article.

Data Fields
    * url: WikiHow URL of the article
    * article: A dictionary containing section_name, document and summary
        - section_name: List of section headings in an article
        - document: List of documents, one for each section in the section_name list
        - summary: List of summarized document
"""


def load_task_dataset(split_ratio="0.8_0.1_0.1"):
    task_name = 'WikiLingua'
    data_dir = f"datasets/nlg_tasks/{task_name}/"
    os.makedirs(data_dir, exist_ok=True)

    train_ratio, dev_ratio, test_ratio = map(float, split_ratio.split("_"))
    ratio = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}

    print(f"\n##### Loading the {task_name} dataset #####")
    print(f"Data path: {data_dir}\n")

    if len(os.listdir(data_dir)) != 0:
        total_dataset = json.load(open(os.path.join(data_dir, "wikilingua_korean.json"), "r"))
    else:
        raw_dataset = load_dataset("esdurmus/wiki_lingua", 'korean')['train']['article']

        np.random.seed(42)
        rand_idx = np.random.permutation(np.arange(len(raw_dataset)))

        train_dataset = [raw_dataset[idx] for idx in rand_idx[: int(len(raw_dataset) * ratio['train'])]]
        dev_dataset = [raw_dataset[idx] for idx in rand_idx[int(len(raw_dataset) * ratio['train']): int(len(raw_dataset) * (ratio['train'] + ratio['dev']))]]
        test_dataset = [raw_dataset[idx] for idx in rand_idx[int(len(raw_dataset) * (ratio['train'] + ratio['dev'])):]]

        assert len(train_dataset) + len(dev_dataset) + len(test_dataset) == len(raw_dataset)

        raw_dataset = {'train': train_dataset,
                       'dev': dev_dataset,
                       'test': test_dataset}

        total_dataset = dict()
        for mode in ['train', 'dev', 'test']:
            cur_mode_dataset = raw_dataset[mode]
            dataset = {'text': list(),
                       'summary': list()
                       }
            for docs in cur_mode_dataset:
                for i in range(len(docs['document'])):
                    dataset['text'].append(docs['document'][i])
                    dataset['summary'].append(docs['summary'][i])

            assert len(dataset['text']) == len(dataset['summary'])

            total_dataset[mode] = dataset

        json.dump(total_dataset, open(os.path.join(data_dir, "wikilingua_korean.json"), "w"), ensure_ascii=True, indent=2)

    return total_dataset


if __name__ == '__main__':
    dataset = load_task_dataset()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")


    text_lengths = [len(tokenizer.tokenize(text)) for text in tqdm(dataset['train']['text'])]
    summary_lengths = [len(tokenizer.tokenize(text)) for text in tqdm(dataset['train']['summary'])]

    print(f"train data context max length: {np.max(text_lengths)}")
    print(f"train data summary max length: {np.max(summary_lengths)}")

    print(f"train data context mean length: {np.mean(text_lengths)}")
    print(f"train data summary mean length: {np.mean(summary_lengths)}")
