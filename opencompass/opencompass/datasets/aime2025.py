import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class Aime2025Dataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = [json.loads(line) for line in f]
        return Dataset.from_list(data)
