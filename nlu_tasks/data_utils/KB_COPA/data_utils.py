from tqdm import tqdm
from datasets import load_dataset

def load_task_dataset():
    """
    This dataset is constructed by referencing the choice of plausible alternatives (COPA) (Roemmele et al.,2011) dataset.
    The data has four variables: premise, two alternatives, and a question that asks a model to decide
    the cause or effect of the premise from the two alternatives.
    The label is 0 if the first alternative is the correct answer, and 1 if the second alternative is the correct answer.
    """
    dataset = load_dataset("skt/kobest_v1", "copa")
    trimmed_dataset = {'train': dict(),
                       'dev': dict(),
                       'test': dict()}
    for d_type in dataset.keys():
        cur_dataset = dataset[d_type]
        if d_type == 'validation':
            d_type = 'dev'

        for data in tqdm(cur_dataset, desc=f"Trimming {d_type} dataset...", bar_format="{l_bar}{bar:15}{r_bar}"):
            new_data = {
                'choices': [data['premise'] + ' ' + data['question'] + ': ' + data['alternative_1'],
                            data['premise'] + ' ' + data['question'] + ': ' + data['alternative_2']
                            ],
                'label': data['label']
            }
            for key in new_data.keys():
                if key not in trimmed_dataset[d_type]:
                    trimmed_dataset[d_type][key] = [new_data[key]]
                else:
                    trimmed_dataset[d_type][key].append(new_data[key])

    trimmed_dataset['label_map'] = {0: "alternative_1", 1: "alternative_2"}
    return trimmed_dataset


if __name__ == "__main__":
    dataset = load_task_dataset()


    import numpy as np

    total_sentence_length = []
    for key in dataset:
        if 'label' not in key:
            cur_dataset = dataset[key]
            sentence_length = [np.mean([len(choices[0]), len(choices[1])]) for choices in cur_dataset['choices']]
            total_sentence_length.extend(sentence_length)

    print(f"Average sentence length: {sum(total_sentence_length) / len(total_sentence_length)}")
    print(f"Max sentence length: {max(total_sentence_length)}")
