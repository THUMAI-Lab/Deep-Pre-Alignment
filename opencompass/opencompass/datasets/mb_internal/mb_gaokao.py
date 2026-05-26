import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class MbGaokaoDataset(BaseDataset):

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
                    'golden': curr_data['golden']
                })
        # Hard set dev and test split, for 5-shot setting
        dev_data = raw_data[:5]
        test_data = raw_data[5:]

        dataset['dev'] = Dataset.from_list(dev_data)
        dataset['test'] = Dataset.from_list(test_data)
        return dataset
