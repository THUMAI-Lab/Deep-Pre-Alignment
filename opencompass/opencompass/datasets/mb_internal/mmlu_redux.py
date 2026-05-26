from datasets import load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUReduxDataset(BaseDataset):

    @staticmethod
    def load(path: str, name):
        path = get_data_path(path)

        dataset = load_dataset(path, name)

        def process_data(example):
            example['A'] = example['choices'][0]
            example['B'] = example['choices'][1]
            example['C'] = example['choices'][2]
            example['D'] = example['choices'][3]
            example['input'] = example['question']
            answer = 'ABCD'[example['answer']]
            if example['correct_answer'] and example[
                    'error_type'] == 'wrong_groundtruth' and example[
                        'correct_answer'] in 'ABCD':
                answer = example['correct_answer']
                example['error_type'] = 'ok'
            example['target'] = answer
            return example

        dataset = dataset.map(process_data)
        dataset = dataset.filter(lambda x: x['error_type'] == 'ok')
        dataset['train'] = dataset['test']
        return dataset
