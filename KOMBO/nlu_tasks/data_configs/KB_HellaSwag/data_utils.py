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
                'sentence1': data['context'] + ' [SEP] ' + data['ending_1'],
                'sentence2': data['context'] + ' [SEP] ' + data['ending_2'],
                'sentence3': data['context'] + ' [SEP] ' + data['ending_3'],
                'sentence4': data['context'] + ' [SEP] ' + data['ending_4'],
                'label': data['label']
            }
            for key in new_data.keys():
                if key not in trimmed_dataset[d_type]:
                    trimmed_dataset[d_type][key] = [new_data[key]]
                else:
                    trimmed_dataset[d_type][key].append(new_data[key])

    trimmed_dataset['label_map'] = {0: "ending_1", 1: "ending_2", 2: "ending_3", 3: "ending_4"}
    return trimmed_dataset


if __name__ == '__main__':
    data = load_task_dataset()
