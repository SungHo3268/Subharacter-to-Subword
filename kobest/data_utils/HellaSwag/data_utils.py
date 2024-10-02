import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from utils.data_preprocess import clean_text
from datasets import Dataset, load_dataset


"""
KoBEST dataset
[https://huggingface.co/datasets/skt/kobest_v1]                                             
"""


def load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True):
    task_name='hellaswag'
    data_dir = f"/data3/user21/KOMBO/datasets/kobest/{task_name}/"
    # data_dir = f"/data2/user13/workspace/KOMBO_Generation/datasets/nlu_tasks/{task_name}/"
    if do_hangeulize:
        data_path = os.path.join(data_dir, f'processed_data_{remain_lang}_hangeulized.json')
    else:
        data_path = os.path.join(data_dir, f'processed_data_{remain_lang}.json')
    if data_remove:
        data_path = data_path.replace(".json", "_dr.json")

    print(f"\n##### Loading the {task_name} dataset #####")
    print(f"Data path: {data_path}\n")
    # if os.path.exists(data_path):
    #     total_dataset = json.load(open(data_path, "r"))
    # else:
    total_dataset = {'train': dict(),
                        'validation': dict(),
                        'test': dict()
                        }

    raw_dataset = load_dataset('skt/kobest_v1', task_name)
    for d_type in total_dataset:
        dataset = {'query': [], 'choices': [], 'gold': []}
        for i, data in tqdm(enumerate(raw_dataset[d_type]), desc=f"* {d_type.upper()} set...", bar_format="{l_bar}{bar:10}{r_bar}", total=len(raw_dataset)):
            query = clean_text(data['context'])
            query = f"문장: {query}"

            sentence1 = clean_text(data['ending_1'], remain_lang, do_hangeulize, data_remove)
            sentence2 = clean_text(data['ending_2'], remain_lang, do_hangeulize, data_remove)
            sentence3 = clean_text(data['ending_3'], remain_lang, do_hangeulize, data_remove)
            sentence4 = clean_text(data['ending_4'], remain_lang, do_hangeulize, data_remove)
            if sentence1 is None or len(sentence1) == 0:
                continue
            if sentence2 is None or len(sentence2) == 0:
                continue
            if sentence3 is None or len(sentence3) == 0:
                continue
            if sentence4 is None or len(sentence4) == 0:
                continue
            
            choices = [sentence1, sentence2, sentence3, sentence4]
            gold = int(data['label'])
            
            dataset['query'].append(query)
            dataset['choices'].append(choices)
            dataset['gold'].append(gold)
        total_dataset[d_type] = dataset

    total_dataset['label_map'] = None
        # json.dump(total_dataset, open(data_path, "w"))
    return total_dataset


if __name__ == '__main__':
    data = load_task_dataset(remain_lang="ko_en_punc", do_hangeulize=True, data_remove=True)
    # raw_dataset = load_dataset('skt/kobest_v1', 'hellaswag')
    # print(raw_dataset)