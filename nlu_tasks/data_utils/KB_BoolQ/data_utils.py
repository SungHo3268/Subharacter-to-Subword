from tqdm import tqdm
from datasets import load_dataset

def load_task_dataset():
    """
    This task is a binary classification task. The dataset consists of paragraphs and questions.
    The model is expected to predict whether the answer to the question is True(1) or False(0).
    :return: dataset
    """
    dataset = load_dataset("skt/kobest_v1", "boolq")
    trimmed_dataset = {'train': dict(),
                       'dev': dict(),
                       'test': dict()}
    for d_type in dataset.keys():
        cur_dataset = dataset[d_type]
        if d_type == 'validation':
            d_type = 'dev'

        for data in cur_dataset:
            new_data = {
                'sentence1': data['paragraph'],
                'sentence2': data['question'],
                'label': data['label']
            }
            for key in new_data.keys():
                if key not in trimmed_dataset[d_type]:
                    trimmed_dataset[d_type][key] = [new_data[key]]
                else:
                    trimmed_dataset[d_type][key].append(new_data[key])

    trimmed_dataset['label_map'] = {0: "False", 1: "True"}
    return trimmed_dataset


if __name__ == "__main__":
    dataset = load_task_dataset()

    total_sentence_length = []
    for key in dataset:
        if 'label' not in key:
            cur_dataset = dataset[key]
            sentence_length = [len(sen1) + len(sen2) for sen1, sen2 in zip(cur_dataset['sentence1'], cur_dataset['sentence2'])]
            total_sentence_length.extend(sentence_length)

    print(f"Average sentence length: {sum(total_sentence_length) / len(total_sentence_length)}")
    print(f"Max sentence length: {max(total_sentence_length)}")
