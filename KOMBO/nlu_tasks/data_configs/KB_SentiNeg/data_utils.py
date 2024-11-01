from datasets import load_dataset


def load_task_dataset():
    """
    This dataset is for a two-class sentiment analysis task by generating product reviews
    based on real product reviews available on e-commerce websites.
    The label is 0 if the given sentence has negative meaning and 1 if it has positive meaning.
    """
    dataset = load_dataset("skt/kobest_v1", "sentineg")
    trimmed_dataset = {'train': dict(),
                       'dev': dict(),
                       'test': dict()}
    for d_type in dataset.keys():
        if d_type == 'test_originated':
            continue

        cur_dataset = dataset[d_type]
        if d_type == 'validation':
            d_type = 'dev'

        for data in cur_dataset:
            new_data = {
                'sentence': data['sentence'],
                'label': data['label']
            }
            for key in new_data.keys():
                if key not in trimmed_dataset[d_type]:
                    trimmed_dataset[d_type][key] = [new_data[key]]
                else:
                    trimmed_dataset[d_type][key].append(new_data[key])

    trimmed_dataset['label_map'] = {0: "Negative", 1: "Positive"}
    return trimmed_dataset


if __name__ == '__main__':
    data = load_task_dataset()
