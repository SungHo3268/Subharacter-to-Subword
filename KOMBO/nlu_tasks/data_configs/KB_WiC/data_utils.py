from datasets import load_dataset


def load_task_dataset():
    """
    This dataset is constructed by referencing the Word-in-Context (WiC) dataset.
    An instance is composed of a target homonym and two different contexts that contain the target word.
    The label is 0 if the two contexts are not semantically equivalent, and 1 if they are.
    """
    dataset = load_dataset("skt/kobest_v1", "wic")
    trimmed_dataset = {'train': dict(),
                       'dev': dict(),
                       'test': dict()}
    for d_type in dataset.keys():
        cur_dataset = dataset[d_type]
        if d_type == 'validation':
            d_type = 'dev'

        for data in cur_dataset:
            new_data = {
                'sentence1': data['context_1'],
                'sentence2': data['context_2'],
                'label': data['label']
            }
            for key in new_data.keys():
                if key not in trimmed_dataset[d_type]:
                    trimmed_dataset[d_type][key] = [new_data[key]]
                else:
                    trimmed_dataset[d_type][key].append(new_data[key])

    trimmed_dataset['label_map'] = {0: "False", 1: "True"}
    return trimmed_dataset


if __name__ == '__main__':
    data = load_task_dataset()
