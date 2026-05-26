import json
import os.path as osp
import re
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


@LOAD_DATASET.register_module()
class WmtDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()

        filename = osp.join(path, f'{name}.jsonl')
        raw_data = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                curr_data = json.loads(line)
                raw_data.append({
                    'input': curr_data['input'],
                    'golden': curr_data['ground_truth']
                })
        # Hard set dev and test split, for 5-shot setting
        dev_data = raw_data[:5]
        # test_data = raw_data[5:]
        test_data = raw_data[5:]

        dataset['dev'] = Dataset.from_list(dev_data)
        dataset['test'] = Dataset.from_list(test_data)
        return dataset


@TEXT_POSTPROCESSORS.register_module('flores')
def flores_postprocess(text: str) -> str:
    text = text.strip().split('\n')[0]
    return text


@TEXT_POSTPROCESSORS.register_module('flores-chinese')
def flores_postprocess_chinese(text: str) -> str:
    import jieba
    truncated_text = text.strip().split('\n')[0]
    cleaned_text = re.sub(r'\s+', ' ', truncated_text).strip()
    cleaned_text = ' '.join(jieba.cut(cleaned_text))
    return cleaned_text
