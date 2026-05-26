import json
import os

import pandas as pd
from datasets import Dataset, DatasetDict

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset


def read_md(md_path: str) -> str:
    with open(md_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


@LOAD_DATASET.register_module()
class MbLongTextDataset(BaseDataset):

    @staticmethod
    def load(path):
        datasets = {}
        for split in [
                '超长文本1',
                '超长文本2',
        ]:

            split_excel_path = os.path.join(path, split, '超长文本.xlsx')
            # 读取 Excel 文件
            # data/mb_long_text/超长文本1/超长文本/yuxiaoling.md
            # data/mb_long_text/超长文本1/超长文本/yuxuailing.md
            df = pd.read_excel(split_excel_path)
            # 将DataFrame转换为list[dict]格式
            dataset = df.to_dict(orient='records')

            # 输出结果
            # print(dataset)
            new_dataset = []
            for data in dataset:
                new_data = {}
                long_text_path = os.path.join(
                    path, split, '超长文本',
                    f'{str(data["文件路径"]).replace("/", ":")}.md')
                long_text_content = read_md(long_text_path)
                new_data['题目'] = data['题目'].replace(
                    '{{' + str(data['文件路径']) + '}}', long_text_content)
                new_data['answer'] = ''
                new_dataset.append(new_data.copy())
            # dataset = []
            # with open(split_path, 'r', encoding='utf-8') as f:
            #     for line in f:
            #         line = json.loads(line.strip())
            #         dataset.append(line)

            # 将list[dict]存储为jsonl文件
            with open(os.path.join(path, split, f'{split}.jsonl'),
                      'w',
                      encoding='utf-8') as file:
                for item in new_dataset:
                    json_line = json.dumps(item, ensure_ascii=False)
                    file.write(json_line + '\n')

            datasets[split] = Dataset.from_list(new_dataset)
        return DatasetDict(datasets)


class BlankEvaluator(BaseEvaluator):

    def is_equal(self, pred, refer):
        try:
            if pred == refer:
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
