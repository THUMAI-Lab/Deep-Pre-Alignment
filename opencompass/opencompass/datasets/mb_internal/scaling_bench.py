import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class ScalingBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, file_name: str = 'data.jsonl'):
        dataset = DatasetDict()
        filename = osp.join(path, file_name)
        with open(filename) as f:
            raw_data = f.readlines()
            raw_data = [json.loads(data) for data in raw_data]
        processed_data = []
        for source_data in raw_data:
            input_str = source_data['input']
            answer_str = source_data['output']
            processed_data.append({
                'input': input_str,
                'answer': answer_str,
                'answer_key': answer_str
            })
        dataset['dev'] = Dataset.from_list(processed_data)
        return dataset
