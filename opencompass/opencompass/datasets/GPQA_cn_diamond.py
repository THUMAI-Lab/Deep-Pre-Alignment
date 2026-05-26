import json
import os.path as osp

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GPQACnDiamondDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.jsonl')
        dataset = []
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)
        return Dataset.from_list(dataset)
