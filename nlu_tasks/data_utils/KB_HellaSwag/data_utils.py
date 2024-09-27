from tqdm import tqdm
from datasets import load_dataset

def load_task_dataset():
    """
    This dataset is constructed by referencing the HellaSwag dataset (Zellers et al., 2019).
    This task evaluates whether a system can utilize passage of time and order to complete the last sentence in a series of sentences.
    The data has four variables: context, and four alternatives.
    The label 0 to 3 corresponds to the correct answer among the four alternatives.
    """
    dataset = load_dataset("skt/kobest_v1", "hellaswag")
    trimmed_dataset = {'train': dict(),
                       'dev': dict(),
                       'test': dict()}
    for d_type in dataset.keys():
        cur_dataset = dataset[d_type]
        if d_type == 'validation':
            d_type = 'dev'

        for data in tqdm(cur_dataset, desc=f"Trimming {d_type} dataset...", bar_format="{l_bar}{bar:15}{r_bar}"):
            new_data = {
                'choices': [data['context'] + data['ending_1'],
                            data['context'] + data['ending_2'],
                            data['context'] + data['ending_3'],
                            data['context'] + data['ending_4']
                            ],
                'label': data['label']
            }
            for key in new_data.keys():
                if key not in trimmed_dataset[d_type]:
                    trimmed_dataset[d_type][key] = [new_data[key]]
                else:
                    trimmed_dataset[d_type][key].append(new_data[key])

    trimmed_dataset['label_map'] = {0: "ending_1", 1: "ending_2", 2: "ending_3", 3: "ending_4"}
    return trimmed_dataset


if __name__ == "__main__":
    dataset = load_task_dataset()


    import numpy as np

    total_sentence_length = []
    for key in dataset:
        if 'label' not in key:
            cur_dataset = dataset[key]
            sentence_length = [np.mean([len(choice) for choice in choices]) for choices in cur_dataset['choices']]
            total_sentence_length.extend(sentence_length)

    print(f"Average sentence length: {sum(total_sentence_length) / len(total_sentence_length)}")
    print(f"Max sentence length: {max(total_sentence_length)}")


    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

    total_sentence_length = []
    for key in dataset:
        if 'label' not in key:
            cur_dataset = dataset[key]

            sentence_length = [np.mean([len(tokenizer(choice)['input_ids']) for choice in choices] )for choices in cur_dataset['choices']]
            total_sentence_length.extend(sentence_length)
    print(f"Average 'token' length: {sum(total_sentence_length) / len(total_sentence_length)}")
    print(f"Max 'token' length: {max(total_sentence_length)}")
