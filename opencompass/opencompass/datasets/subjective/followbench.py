# flake8: noqa
import json
import os.path as osp
import re
import statistics
from typing import Optional

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import get_judgeanswer_and_reference


@LOAD_DATASET.register_module()
class FollowBenchDataset(BaseDataset):

    def load(self, path: str, name: str, cate: str, *args, **kwargs):

        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.json')
        raw_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if cate == 'llm':
                    raw_data.append({
                        'instruction': item['instruction'],
                        'judge_prompt': item['judge_prompt'],
                        'judge': item
                    })
                # elif cate == 'rule':
                #     raw_data.append({
                #         'instruction': item['instruction'],
                #         'judge': item
                #     })
                else:
                    raise NotImplementedError(
                        f"Category '{cate}' is not implemented.")

        dataset = Dataset.from_list(raw_data)
        return dataset


def post_process_followbench(item):
    generation, level = item['prediction'], item['gold']['level']
    try:
        satisfy = generation.strip('```').strip().split('\n')[-1]

        if level == 1:
            if 'YES' in satisfy:
                return 1, 1
            elif 'NO' in satisfy:
                return 0, 0
            else:
                raise Exception('Invalid evaluation for level 1.')
        else:
            satisfy_list = re.search(r'\[.*\]', satisfy)
            if satisfy_list:
                satisfy_list = eval(satisfy_list.group())
                if len(satisfy_list) == level:
                    num_true = 0
                    for i in satisfy_list:
                        if i == 'YES' or i == 'True':
                            num_true += 1
                        elif i in [
                                'NO', 'False', 'PARTIAL', 'MAYBE', 'UNKNOWN',
                                'N/A'
                        ]:
                            num_true += 0
                        else:
                            raise Exception('Invalid element in the list.')
                    return int(num_true == level), num_true / level
                else:
                    raise Exception('Invalid number of elements in the list.')
            else:
                raise Exception('Invalid list that cannot be parsed.')

    except Exception as e:
        return -1, -1


def get_scores(judged_answers, references):
    results = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    n_group = len(judged_answers) // 5
    n_groups = [n_group] * 5

    for judged_answer, reference in zip(judged_answers, references):
        if judged_answer[0] == -1:
            n_groups[reference['level'] - 1] -= 1
        else:
            results[0][reference['level'] - 1] += judged_answer[0]
            results[1][reference['level'] - 1] += judged_answer[1]

    for i in range(len(results)):
        for j in range(len(results[i])):
            if n_groups[j] != 0:
                results[i][j] = results[i][j] / n_groups[j]
            else:
                results[i][j] = 0
    temp_dict = {
        'HSR_AVG': statistics.mean(results[0]),
        'SSR_AVG': statistics.mean(results[1])
    }
    for idx, s in enumerate(results[0]):
        temp_dict[f'HSR_L{idx+1}'] = s
    for idx, s in enumerate(results[1]):
        temp_dict[f'SSR_L{idx+1}'] = s

    return temp_dict


@DICT_POSTPROCESSORS.register_module('followbench')
def followbench_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers, references = get_judgeanswer_and_reference(
        output, output_path, post_process_followbench)

    results = get_scores(judged_answers, references)
    results['details'] = output
    return results
