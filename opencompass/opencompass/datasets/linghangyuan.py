import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LinghangyuanDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.jsonl')
        zh_dataset = []
        en_dataset = []
        with open(filename, 'r') as f:
            for line in f:
                line = json.loads(line)

                data = {}
                data['question'] = line['prompt'][0]['content']
                data['answer'] = line['reward_model']['ground_truth']
                if any('\u4e00' <= char <= '\u9fff'
                       for char in data['question']):
                    zh_dataset.append(data)
                else:
                    en_dataset.append(data)
        return DatasetDict({
            'zh': Dataset.from_list(zh_dataset),
            'en': Dataset.from_list(en_dataset)
        })
