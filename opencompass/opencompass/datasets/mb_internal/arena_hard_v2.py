# flake8: noqa: W605
import json
import os.path as osp
import re
from copy import deepcopy
from typing import Dict, List, Optional

import mmengine
from datasets import Dataset, DatasetDict
from mmengine.config import ConfigDict

from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import build_dataset_from_cfg, get_data_path
from opencompass.utils.logging import get_logger

from ..base import BaseDataset

logger = get_logger(__name__)


@LOAD_DATASET.register_module()
class ArenaHardV2Dataset(BaseDataset):
    """Arena Hard v2.0 Dataset with support for hard prompts and creative
    writing.

    Arena Hard v2.0 contains:
    - 500 hard prompts for general evaluation
    - 250 creative writing prompts for creative evaluation

    Features:
    - Style Control support (markdown elements, token length)
    - Creative writing specialized evaluation
    - Multi-judge ensemble support (GPT-4.1 + Gemini-2.5)
    """

    def load(self, path: str, name: str, baseline: str, *args, **kwargs):
        """Load Arena Hard v2.0 dataset.

        Args:
            path: Path to the dataset
            name: Dataset name
            baseline: Answer of baseline model to compare with
        """
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, 'question.jsonl')
        baseline_answers = self._load_model_answers(path, baseline)

        raw_data = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                problem = json.loads(line)
                if problem['subcategory'] != name:
                    continue

                raw_data.append({
                    'uid': problem['uid'],
                    'category': problem['category'],
                    'subcategory': problem['subcategory'],
                    'language': problem['language'],
                    'prompt': problem['prompt'],
                    'baseline': baseline_answers[problem['uid']],
                    'reference': '',
                })

        return Dataset.from_list(raw_data)

    def _load_model_answers(self, path: str, baseline: str) -> Dict[str, str]:
        """Load baseline model answers."""
        model_answers = {}
        answer_file = osp.join(path, 'model_answer', f'{baseline}.jsonl')

        with open(answer_file, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                model_answers[
                    data['uid']] = data['messages'][-1]['content']['answer']

        return model_answers


class ArenaHardV2Evaluator(GenericLLMEvaluator):
    """Arena Hard V2 Evaluator with dual-round evaluation to eliminate position
    bias."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def score(self,
              predictions,
              references: Optional[List] = None,
              test_set: Optional[Dataset] = None) -> Dict:
        """Apply dual-round scoring to eliminate position bias."""
        assert len(predictions) == len(references), \
            'predictions and references must have the same length'

        # Build inferencer and process predictions
        self.build_inferencer()
        predictions = self.pred_postprocess(predictions)

        # Build pairwise evaluation dataset
        dataset = build_dataset_from_cfg(self.dataset_cfg)
        raw_data = self._build_pairwise_data(predictions, dataset.test)

        # Create LMEvalDataset for ZeroRetriever compatibility
        pairwise_dataset = self._create_eval_dataset(raw_data)

        # Run LLM Judge inference
        retriever = ZeroRetriever(pairwise_dataset)
        self.inferencer.inference(retriever=retriever,
                                  prompt_template=self.prompt_template)

        # Post-process results
        output = mmengine.load(self.output_path)
        return super().output_postprocess(output, pairwise_dataset)

    def _build_pairwise_data(self, predictions: List[str],
                             test_data) -> List[Dict]:
        """Build pairwise comparison data for dual-round evaluation."""
        raw_data = []

        for pred, line in zip(predictions, test_data):
            question = line['prompt']
            baseline_answer = line['baseline']

            # Round 1: baseline (A) vs model (B)
            raw_data.append({
                **line, 'test': {
                    'QUESTION': question,
                    'ANSWER_A': baseline_answer,
                    'ANSWER_B': pred,
                }
            })

            # Round 2: model (A) vs baseline (B)
            raw_data.append({
                **line, 'test': {
                    'QUESTION': question,
                    'ANSWER_A': pred,
                    'ANSWER_B': baseline_answer,
                }
            })

        return raw_data

    def _create_eval_dataset(self, raw_data: List[Dict]):
        """Create LMEvalDataset compatible with ZeroRetriever."""
        from opencompass.datasets.lmeval import LMEvalDataset

        if not raw_data:
            data_dict = {
                col: []
                for col in ['QUESTION', 'ANSWER_A', 'ANSWER_B', 'reference']
            }
        else:
            # Extract data from test field
            data_dict = {
                'QUESTION': [],
                'ANSWER_A': [],
                'ANSWER_B': [],
                'reference': []
            }

            for item in raw_data:
                test_data = item.get('test', {})
                data_dict['QUESTION'].append(test_data.get('QUESTION', ''))
                data_dict['ANSWER_A'].append(test_data.get('ANSWER_A', ''))
                data_dict['ANSWER_B'].append(test_data.get('ANSWER_B', ''))
                data_dict['reference'].append('')

        input_columns = ['QUESTION', 'ANSWER_A', 'ANSWER_B'
                         ] if raw_data else []

        return LMEvalDataset(
            reader_cfg=dict(
                input_columns=input_columns,
                output_column='reference',
                train_split='test',
            ),
            **data_dict,
        )


@DICT_POSTPROCESSORS.register_module('arenahard_v2_postprocess')
def arenahard_v2_postprocess(output: dict, output_path: str) -> Dict:
    """Post-process Arena Hard v2 results with dual-round evaluation
    support."""
    patterns = [r'\[\[([AB<>=]+)\]\]', r'\[([AB<>=]+)\]']

    # Process dual-round evaluation results
    sorted_keys = sorted(output.keys(),
                         key=lambda x: int(x) if str(x).isdigit() else 0)

    all_judgments = []
    all_details = []

    for i in range(0, len(sorted_keys), 2):
        assert i + 1 < len(sorted_keys), f'not enough pairs at index {i}'

        # Extract dual-round judgments
        round1_output = output[sorted_keys[i]]
        round2_output = output[sorted_keys[i + 1]]

        logger.info(f'Processing pair {i//2}: '
                    f'round1=...{round1_output.get("prediction", "")[-100:]}')
        logger.info(f'Processing pair {i//2}: '
                    f'round2=...{round2_output.get("prediction", "")[-100:]}')

        # Extract scores from both rounds
        score1 = _extract_score(round1_output.get('prediction', ''), patterns)
        score2 = _extract_score(round2_output.get('prediction', ''), patterns)

        if score1 is None or score2 is None:
            logger.warning(f'Failed to extract scores for pair {i//2}: '
                           f'score1={score1}, score2={score2}')
            continue

        # Calculate final score
        final_score = _calculate_arena_score(score1, score2)

        # Store detailed information
        detail = {
            'round1_judgment': round1_output.get('prediction', ''),
            'round2_judgment': round2_output.get('prediction', ''),
            'round1_score': score1,
            'round2_score': score2,
            'final_score': final_score,
            'baseline_answer': round1_output.get('gold',
                                                 {}).get('baseline', ''),
            'model_answer': round1_output.get('gold', {}).get('reference', ''),
            'question': round1_output.get('gold', {}).get('prompt', ''),
        }

        all_judgments.append(final_score)
        all_details.append(detail)

    if not all_judgments:
        logger.error('No valid judgments found!')
        return {'score': 0.0, 'details': []}

    # Calculate win rate
    win_count = sum(1 for score in all_judgments if score > 0.5)
    total_count = len(all_judgments)
    win_rate = (win_count / total_count) * 100 if total_count > 0 else 0.0

    # Build OpenCompass-compatible details format
    details_list = []
    for detail in all_details:
        details_list.append({
            'pred': (detail['round1_judgment'][:100] +
                     '...' if len(detail['round1_judgment']) > 100 else
                     detail['round1_judgment']),
            'answer':
            f"Round1: {detail['round1_score']}, Round2: {detail['round2_score']}",
            'correct':
            detail['final_score'] > 0.5,
            'final_score':
            detail['final_score'],
            'round1_score':
            detail['round1_score'],
            'round2_score':
            detail['round2_score'],
            'question':
            detail['question'],
            'baseline_answer':
            detail['baseline_answer'],
            'model_answer':
            detail['model_answer'],
        })

    results = {
        'score': round(win_rate, 2),
        'details': details_list,
        'win_count': win_count,
        'total_count': total_count,
        'win_rate': win_rate,
        'raw_output': output
    }

    logger.info(
        f'ArenaHard V2 Results: {win_count}/{total_count} wins, {win_rate:.2f}% win rate'
    )
    return results


def _calculate_arena_score(score1: str, score2: str, weight: int = 3) -> float:
    """Calculate ArenaHard final score from dual-round evaluation.

    This follows the original arena-hard-auto scoring logic:
    scores = label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]

    Args:
        score1: Round 1 score (baseline vs model)
        score2: Round 2 score (model vs baseline)
        weight: Weight for strong preferences (>> and <<)

    Returns:
        float: Final score between 0-1, >0.5 means model wins
    """
    # Map label to score following original arena-hard-auto logic
    label_to_score = {
        'A>B': [1],
        'A>>B': [1] * weight,
        'A=B': [0.5],
        'A<<B': [0] * weight,
        'A<B': [0],
        'B>A': [0],
        'B>>A': [0] * weight,
        'B=A': [0.5],
        'B<<A': [1] * weight,
        'B<A': [1],
    }

    # Create games format matching original repo
    x = [
        {
            'score': score1
        },
        {
            'score': score2
        },
    ]

    # Calculate scores using original arena-hard formula:
    # scores = label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]
    scores = label_to_score[x[1]['score']] + [
        1 - s for s in label_to_score[x[0]['score']]
    ]

    # For compatibility with OpenCompass, return mean of the score distribution
    # Note: Original arena-hard uses bootstrap sampling on this score list
    return sum(scores) / len(scores)


def _extract_score(judgment: str, patterns: List[str]) -> Optional[str]:
    """Extract score from judgment text using regex patterns."""
    for pattern in patterns:
        matches = re.findall(pattern, judgment.upper())
        matches = [m for m in matches if m]
        if matches:
            return matches[-1].strip()
    return None


# Placeholder functions for future extensions
@DICT_POSTPROCESSORS.register_module('arenahard_v2_style_control')
def arenahard_v2_style_control_postprocess(output: dict,
                                           output_path: str) -> Dict:
    """Post-process Arena Hard v2 results with style control support."""
    # TODO: Implement style-specific evaluation logic
    return arenahard_v2_postprocess(output, output_path)


@DICT_POSTPROCESSORS.register_module('arenahard_v2_ensemble')
def arenahard_v2_ensemble_postprocess(output: dict, output_path: str) -> Dict:
    """Post-process Arena Hard v2 results with ensemble support."""
    # TODO: Implement ensemble evaluation logic
    return arenahard_v2_postprocess(output, output_path)
