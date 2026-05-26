import json
import os.path as osp
import re

import pandas as pd
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path, get_logger

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class GenericExcelDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.xlsx')
        df = pd.read_excel(filename)
        dataset = []
        for _, row in df.iterrows():
            item = {}
            for col in df.columns:
                if pd.notna(row[col]):
                    item[col] = str(row[col])
            dataset.append(item)
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class GenericJsonlDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{name}.jsonl')
        dataset = []
        with open(filename, 'r') as f:
            for line in f:
                line = json.loads(line)
                if 'choices' in line and isinstance(line['choices'], list):
                    for choice in line['choices']:
                        line['question'] += choice
                dataset.append(line)
        return Dataset.from_list(dataset)


@ICL_EVALUATORS.register_module()
class GenericBlankEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        # if len(predictions) != len(references):
        #     return {
        #         'error': 'predictions and references have different length'
        #     }

        # processed_preds = [p.lower() for p in predictions]
        # # References are already in correct format
        # processed_refs = [r.lower() for r in references]

        # details = []
        # correct_count = 0

        # for pred, ref in zip(processed_preds, processed_refs):
        #     if pred:
        #         correct = True
        #     else:
        #         correct = False
        #     details.append({'pred': pred, 'answer': ref, 'correct': correct})
        #     correct_count += int(correct)

        # score = (correct_count / len(predictions)) * 100
        return {'score': 0, 'details': ['no evaluator']}


@ICL_EVALUATORS.register_module()
class GenericGenericLLMEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        processed_preds = [p.lower() for p in predictions]
        # References are already in correct format
        processed_refs = [r.lower() for r in references]

        details = []
        correct_count = 0

        for pred, ref in zip(processed_preds, processed_refs):
            if pred:
                correct = True
            else:
                correct = False
            details.append({
                'pred': pred,
                'answer': ref,
                'correct': correct,
            })
            correct_count += int(correct)

        score = (correct_count / len(predictions)) * 100
        return {'score': score, 'details': details}


def _zero_one_llmjudge_postprocess(judgement: str):
    if '</think>' in judgement:
        judgement = judgement.split('</think>')[-1]
    print(f'judgement: {judgement}')
    match = re.search(r'(0|1)', judgement)
    grade_letter = (match.group(0) if match else 'unknown'
                    )  # Return 'unknown' if no match
    return grade_letter


def zero_one_llmjudge_postprocess(
    output: dict,
    output_path: str,
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _zero_one_llmjudge_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            try:
                references.append(v['gold'])

            except KeyError:
                get_logger().warning(
                    f'No gold answer for {k}, use empty string as reference!')
                references.append('')
    results = get_final_results(judged_answers, references, origial_responses)
    results['details'] = output
    return results


def get_final_results(judged_answers,
                      references,
                      origial_responses,
                      metric_name='accuracy',
                      verbose=True):
    count = 0
    is_correct_count = 0
    is_incorrect_count = 0
    is_not_attempted_count = 0
    attempted_judge_count = 0
    details = []
    for i, j, k in zip(judged_answers, references, origial_responses):
        if i in ['0', '1', 0, 1]:
            attempted_judge_count += 1
        grade_letter = i
        detail = {
            'pred': k,
            'ref': j,
            'origin_grade_response': i,
            'grade_letter': grade_letter,
            'correct': False,
        }
        count += 1
        if grade_letter == '1' or grade_letter == 1:
            is_correct_count += 1
            detail['correct'] = True
        elif grade_letter == '0' or grade_letter == 0:
            is_incorrect_count += 1
        else:
            is_not_attempted_count += 1
        if verbose:
            get_logger().info(f'detail: {detail}')
        details.append(detail)

    is_correct = is_correct_count / count
    is_incorrect = is_incorrect_count / count
    is_given_attempted = is_correct + is_incorrect
    accuracy_given_attempted = (is_correct / is_given_attempted
                                if is_given_attempted > 0 else 0)
    attempted_judge_ratio = attempted_judge_count / count

    f1 = (2 * accuracy_given_attempted * is_correct /
          (accuracy_given_attempted + is_correct) if
          (accuracy_given_attempted + is_correct) > 0 else 0)
    result = {
        metric_name: is_correct * 100,
        f'{metric_name}_given_attempted': accuracy_given_attempted * 100,
        'f1': f1,
        'attempted_ratio': attempted_judge_ratio * 100,
        'correct_count': is_correct_count,
        'incorrect_count': is_incorrect_count,
        'not_attempted_count': is_not_attempted_count,
        'details': details,
    }
    return result


@ICL_EVALUATORS.register_module()
class GenericSpeedEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        # if len(predictions) != len(references):
        #     return {
        #         'error': 'predictions and references have different length'
        #     }

        details = []

        total_token_num = 0
        total_processing_time = 0
        avg_token_per_second = 0

        for idx, pred in enumerate(predictions):
            pred, token_num, processing_time, token_per_second = pred.split(
                '[####本数据集用于模型测速####]')
            print(f'pred: {pred}, token_num: {token_num}, '
                  f'processing_time: {processing_time}, '
                  f'token_per_second: {token_per_second}')

            total_token_num += int(token_num)
            total_processing_time += float(processing_time)

            avg_token_per_second = float(total_token_num /
                                         total_processing_time)
            avg_second_per_data = float(total_processing_time / (idx + 1))

            print(f'avg_token_per_second: {avg_token_per_second}, '
                  f'total_token_num: {total_token_num}, '
                  f'total_processing_time: {total_processing_time}, '
                  f'avg_second_per_data: {avg_second_per_data}')

            details.append({
                'pred': pred,
                'token_num': int(token_num),
                'processing_time': float(processing_time),
                'token_per_second': float(token_per_second),
            })

        return {
            'token/s': avg_token_per_second,
            'avg_second_per_data': avg_second_per_data,
            'total_token_num': total_token_num,
            'total_processing_time': total_processing_time,
            'details': details
        }
