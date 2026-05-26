# flake8: noqa
# yapf: disable
import json
import os
from typing import Dict, Optional

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_logger

from .base import BaseDataset


def bbh_loader(path: str, dataset_kwargs: Optional[Dict] = dict()):
    from opencompass.datasets import BBHDataset
    dataset = BBHDataset.load(path, dataset_kwargs['name'])
    data = []
    for index, item in enumerate(dataset):
        text = item['input']
        data.append({
            'input': text,
        })
    return data


def humaneval_loader(path: str, dataset_kwargs: Optional[Dict] = dict()):
    from opencompass.datasets import HumanevalDataset
    dataset = HumanevalDataset.load(path)
    data = []
    for index, item in enumerate(dataset):
        text = item['prompt']  # HumanevalDataset uses 'prompt' as input column
        data.append({
            'input': text,
        })
    return data


def sanitized_mbpp_loader(path: str, dataset_kwargs: Optional[Dict] = dict()):
    from opencompass.datasets import SanitizedMBPPDataset
    dataset = SanitizedMBPPDataset.load(path)
    data = []
    for index, item in enumerate(dataset['test']):
        # SanitizedMBPPDataset uses 'text' as input columns
        text = item['text']
        data.append({
            'input': text,
        })
    return data


def gsm8k_loader(path: str, dataset_kwargs: Optional[Dict] = dict()):
    from opencompass.datasets import GSM8KDataset
    dataset = GSM8KDataset.load(path)
    data = []
    for index, item in enumerate(dataset['test']):
        text = item['question']  # GSM8KDataset uses 'question' as input column
        data.append({
            'input': text,
        })
    return data


def math_loader(path: str, dataset_kwargs: Optional[Dict] = dict()):
    from opencompass.datasets import MATHDataset
    dataset = MATHDataset.load(path, file_name='test_prm800k_500.json')
    data = []
    for index, item in enumerate(dataset['test']):
        text = item['problem']  # MATHDataset uses 'problem' as input column
        data.append({
            'input': text,
        })
    return data


def ceval_loader(path: str, dataset_kwargs: Optional[Dict] = dict()):
    from opencompass.datasets import CEvalDataset
    dataset = CEvalDataset.load(path, dataset_kwargs['name'])
    data = []
    for index, item in enumerate(dataset['val']):
        # CEvalDataset uses ['question', 'A', 'B', 'C', 'D'] as input columns
        text = f"{item['question']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
        data.append({
            'input': text,
        })
    return data


def cmmlu_loader(path: str, dataset_kwargs: Optional[Dict] = dict()):
    from opencompass.datasets import CMMLUDataset
    dataset = CMMLUDataset.load(path, dataset_kwargs['name'])
    data = []
    for index, item in enumerate(dataset['test']):
        # CMMLUDataset uses ['question', 'A', 'B', 'C', 'D'] as input columns
        text = f"{item['question']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
        data.append({
            'input': text,
        })
    return data


def mmlu_loader(path: str, dataset_kwargs: Optional[Dict] = dict()):
    from opencompass.datasets import MMLUDataset
    dataset = MMLUDataset.load(path, dataset_kwargs['name'])
    data = []
    for index, item in enumerate(dataset['test']):
        # MMLUDataset uses ['input', 'A', 'B', 'C', 'D'] as input columns
        text = f"{item['input']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
        data.append({
            'input': text,
        })
    return data


@LOAD_DATASET.register_module()
class BenBenchDataset(BaseDataset):
    @staticmethod
    def load(path: str, tokenizer_path: str = None, tokenizer_kwargs: Optional[Dict] = dict(), num_gram: int = 5, num_replica: int = 5, dataset_kwargs: Optional[Dict] = dict()):
        import numpy as np
        from transformers import AutoTokenizer

        logger = get_logger()
        dataset_loaders = {
            'gsm8k': gsm8k_loader,
            'math500': math_loader,
            'humaneval': humaneval_loader,
            'sanitized_mbpp': sanitized_mbpp_loader,
            'bbh': bbh_loader,
            'ceval': ceval_loader,
            'cmmlu': cmmlu_loader,
            'mmlu': mmlu_loader,
        }
        if not tokenizer_path:
            if os.environ.get('LOCAL_PATH'):
                tokenizer_path = os.environ.get('LOCAL_PATH')
            else:
                logger.warning(
                    f'LOCAL_PATH is not set, using default tokenizer path: Qwen/Qwen2.5-7B-Instruct')
                tokenizer_path = 'Qwen/Qwen2.5-7B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, **tokenizer_kwargs)
        # path = get_data_path(path)
        data = []
        raw_data = []
        if dataset_kwargs.get('dataset'):
            loader = dataset_loaders[dataset_kwargs['dataset']]
            dataset = loader(path, dataset_kwargs)
            for index, item in enumerate(dataset):
                text = item['input']
                raw_data.append(text)
        else:
            with open(path, encoding='utf-8') as f:
                for index, line in enumerate(f):
                    line = json.loads(line)
                    if 'rewritten' in path:
                        text = line['rewritten_question'] + \
                            ' ' + line['rewritten_answer']
                    elif 'origin' in path:
                        text = line['question'] + ' ' + line['answer']
                    else:
                        # raise ValueError(f'Unknown file type: {path}')
                        text = line.get('question', line.get(
                            'problem', '')) + ' ' + line.get('answer', line.get('solution', ''))
                    raw_data.append(text)
        for index, text in enumerate(raw_data):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) >= num_gram + max(num_replica, 2):
                starting_points = np.linspace(
                    2, len(tokens) - num_gram, num=num_replica, endpoint=True, dtype=int).tolist()
            else:
                starting_points = np.linspace(
                    2, max(2, len(tokens)), num=num_replica, endpoint=True, dtype=int).tolist()
            for s in starting_points:
                data.append({
                    'index': index,
                    'prompt': tokenizer.decode(tokens[:s]),
                    'reference': tokenizer.decode(tokens[s:s+num_gram])
                })
        dataset = Dataset.from_list(data)
        return dataset


def exact_match_score(predicted_text, original_text):
    return predicted_text == original_text


def edit_similarity_score(predicted_text, original_text):
    # Calculate normalized edit distance
    import editdistance

    edit_dist = editdistance.eval(predicted_text, original_text)
    max_length = max(len(predicted_text), len(original_text), 1)
    edit_similarity = 1 - (edit_dist / max_length)
    return edit_similarity


def rouge_l_score(predicted_text, original_text):
    # Calculate Rouge-L score
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(original_text, predicted_text)[
        'rougeL'].fmeasure
    return rouge_score


@ICL_EVALUATORS.register_module()
class BenBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'pred and refr length mismatch'}

        valid_exact_match, valid_edit_similarity, valid_rouge_score = 0, 0, 0
        total = len(predictions)
        for pred, ref in zip(predictions, references):
            exact_match = exact_match_score(pred, ref)
            edit_similarity = edit_similarity_score(pred, ref)
            rougeL = rouge_l_score(pred, ref)

            valid_exact_match += exact_match
            valid_edit_similarity += edit_similarity > 0.75
            valid_rouge_score += rougeL > 0.75

        return {
            'exact_match': valid_exact_match / total * 100,
            'edit_similarity': valid_edit_similarity / total * 100,
            'rougeL': valid_rouge_score / total * 100,
        }
