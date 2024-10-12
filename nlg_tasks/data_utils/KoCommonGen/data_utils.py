import os
import json
import numpy as np
from tqdm import tqdm

"""
Korean Text Generation for Commonsense Reasoning (KoCommonGen)
[https://github.com/J-Seo/Korean-CommonGen]

It is composed of 43,188 data for training, 1,000 for development, and 2,040 for test.

When a set of morphemes is given, the model should generate the plausible sentence by utilizing given morphemes. 

For the Decoder-only model, due to the in-context learning, the input shape would be a sentence as follows:
[ morpheme_1, ..., morpheme_n ] + [ EOS ] + [ sentence ]

Example for GPT-2 model:
Given,
    morpheme set: {차#있#거리#버스},
    sentence: "거리에 버스와 차가 있다."
Input: "차, 있, 거리, 버스 |<endoftext>| 거리에 버스와 차가 있다."


For the Encoder-Decoder model, the morpheme set would be the input of the encoder and 
the sentence would be the input of the decoder as follows:
Encoder: [ morpheme_1, ..., morpheme_n ]
Decoder: [ sentence ]
"""


def load_task_dataset():
    task_name = 'KoCommonGen'
    data_dir = f"datasets/nlg_tasks/{task_name}/"

    print(f"\n##### Loading the {task_name} dataset #####")
    print(f"Data path: {data_dir}\n")

    total_dataset = dict()
    for mode in ['train', 'dev', 'test']:
        raw_dataset = json.load(open(os.path.join(data_dir, mode, f"korean_commongen_{mode}_labeled.json"), "r"))['concept_set']
        dataset = {'morpheme_set': list(),
                   'target': list()
                   }
        for line in raw_dataset:
            dataset['morpheme_set'].append(line['concept_set'])

            if mode == 'test':
                dataset['target'].append([line['reference_1'],
                                          line['reference2'],
                                          line['reference3']]
                                         )
            else:
                dataset['target'].append(line['reference_1'])

        assert len(dataset['morpheme_set']) == len(dataset['target'])

        total_dataset[mode] = dataset
    return total_dataset


if __name__ == '__main__':
    dataset = load_task_dataset()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

    morpheme_lengths = [len(tokenizer.tokenize(text)) for text in tqdm(dataset['train']['morpheme_set'])]
    target_lengths = [len(tokenizer.tokenize(text)) for text in tqdm(dataset['train']['target'])]

    print(f"train data morpheme_set mean length: {np.mean(morpheme_lengths)}")
    print(f"train data morpheme_set max length: {np.max(morpheme_lengths)}")
    print(f"train data target mean length: {np.mean(target_lengths)}")
    print(f"train data target max length: {np.max(target_lengths)}")
