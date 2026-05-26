import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class MbGaokao2025MockDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()

        # 读取CSV文件
        filename = osp.join(path, f'{name}.csv')
        # df = pd.read_csv(filename)

        raw_data = []

        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            _ = next(reader)  # skip the header
            for row in reader:
                assert len(row) == 7
                raw_data.append({
                    'question': row[1],
                    'A': row[2],
                    'B': row[3],
                    'C': row[4],
                    'D': row[5],
                    'answer': row[6]
                })
        dataset['dev'] = Dataset.from_list(raw_data[:1])
        dataset['test'] = Dataset.from_list(raw_data[:])
        return dataset
