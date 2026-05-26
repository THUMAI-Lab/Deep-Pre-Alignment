import json
import os
import re
from collections import Counter

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.math import TEXT_POSTPROCESSORS
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import first_option_postprocess


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                     (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                     (',\\text{and}', ','), ('\\text{and}', ','),
                     ('\\text{m}', '\\text{}'), ('\\le', '<')]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f'
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    assert '\n' not in final_answer
    assert '\r' not in final_answer
    assert '\f' not in final_answer
    if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
        final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

    if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
        final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]

    if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
        final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

    if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
        final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
    final_answer = final_answer.strip()
    if 'rac' in final_answer and '\\frac' not in final_answer:
        final_answer = final_answer.replace('rac', '\\frac')

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def get_number(options):
    """生成选项编号字符串 (A.xxx\nB.

    xxx\n...)
    """
    return ''.join(f'{chr(i + 65)}. {option.strip()}\n'
                   for i, option in enumerate(options))


def analyze_jsonl_question_types(file_path):
    """分析jsonl文件中的题目类型分布，返回题目类型计数和总题目数."""
    question_type_counts = Counter()
    total_questions = 0

    try:
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if line := line.strip():
                    data = json.loads(line)
                    question_type_counts[data.get('question_type',
                                                  'unknown')] += 1
                    total_questions += 1
    except (FileNotFoundError, Exception) as e:
        print(f'Warning: Could not read {file_path}: {e}')
        return Counter(), 0

    return question_type_counts, total_questions


def print_question_type_analysis(file_path):
    """打印jsonl文件的题目类型分析结果."""
    counts, total = analyze_jsonl_question_types(file_path)

    print(f'\n=== 文件分析: {os.path.basename(file_path)} ===')
    print(f'总题目数: {total}')
    print('题目类型分布:')

    for question_type, count in counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f'  {question_type}: {count} 题 ({percentage:.1f}%)')

    print('建议创建的数据集:')
    dataset_mapping = {
        '单选': 'SingleChoiceDataset',
        '多选': 'MultipleChoiceDataset',
        '填空题': 'ClozeDataset',
        '解答题': 'SubjectiveDataset'
    }

    for question_type in counts.keys():
        dataset_name = dataset_mapping.get(question_type,
                                           f'未知类型: {question_type}')
        if question_type in dataset_mapping:
            print(f'  - {dataset_name} (用于 {question_type})')
        else:
            print(f'  - {dataset_name}')
    print('=' * 50)


# 题目类型常量
VALID_QUESTION_TYPES = ['单选', '多选', '填空题', '解答题']

QUESTION_TYPE_MAPPING = {
    '单选': 'single_choice',
    '多选': 'multi_choice',
    '填空题': 'cloze',
    '解答题': 'subjective'
}


def load_questions_by_type(path: str,
                           name: str,
                           target_types: list,
                           process_options=False):
    """通用的题目加载函数，减少重复代码."""
    data_list = []
    path = get_data_path(path, local_mode=True)
    try:
        with open(os.path.join(path, f'{name}.jsonl'), encoding='utf-8') as f:
            for line in f:
                if line := line.strip():
                    data = json.loads(line)
                    if data.get('question_type') not in VALID_QUESTION_TYPES:
                        data['question_type'] = '解答题'
                    if data.get('question_type') not in target_types:
                        continue
                    # if process_options and 'options' in data:
                    if data.get('question_type') in {'单选', '多选'
                                                     } and 'options' in data:
                        # and not all(x in data['question'] for x in 'ABCD'):
                        data['question'] = data['question'].strip(
                        ) + '\n' + get_number(data['options'])
                    if isinstance(data['answer'], str):
                        data['answer'] = [data['answer']]
                    # if data.get("question_type") == '多选':
                    #     if 'answer' in data:
                    #       data['answer'] = list(data['answer'])
                    # if data.get("question_type") == '填空题':
                    #     data['answer'] = list(data['answer'].strip())
                    data_list.append(data)
    except Exception as e:
        print(f'Error loading {name}: {e}')

    return Dataset.from_list(data_list)


@LOAD_DATASET.register_module()
class Mbgaokao2025Dataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, question_type: str):
        return load_questions_by_type(path, name, [question_type])


class Mbgaokao2025Evaluator(BaseEvaluator):

    def __init__(self, question_type: str) -> None:
        super().__init__()
        assert question_type in VALID_QUESTION_TYPES, \
            f'Invalid question type: {question_type}'

        self.question_type = question_type
        self.postprocessor_mapping = {
            '单选': first_option_postprocess,
            '多选': mbgaokao2025_postprocess,
            '填空题': mbgaokao2025_postprocess,
            '解答题': mbgaokao2025_postprocess
        }

        self.scorer_mapping = {
            '单选': self.single_choice_scorer,
            '多选': self.multi_choice_scorer,
            '填空题': self.fill_in_the_blank_scorer,
            '解答题': self.subjective_scorer
        }

    def single_choice_scorer(self, pred, ref):
        if isinstance(ref, list):
            ref = ref[0]
        return pred == ref

    def multi_choice_scorer(self, pred, ref):
        """多选题评分：提取ABCD字符并比较."""
        pred_choices = ''.join(sorted([c for c in pred.upper()
                                       if c in 'ABCD']))
        if isinstance(ref, list):
            ref = ''.join(ref)
        else:
            ref = ref.upper()
        return pred_choices == ref

    def _fix_fracs(self, string):
        substrs = string.split('\\frac')
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += '\\frac'
                if len(substr) > 0 and substr[0] == '{':
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != '{':
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += '{' + a + '}{' + b + '}' + post_substr
                        else:
                            new_str += '{' + a + '}{' + b + '}'
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += '{' + a + '}' + b + post_substr
                        else:
                            new_str += '{' + a + '}' + b
        string = new_str
        return string

    def _fix_a_slash_b(self, string):
        if len(string.split('/')) != 2:
            return string
        a = string.split('/')[0]
        b = string.split('/')[1]
        try:
            a = int(a)
            b = int(b)
            assert string == '{}/{}'.format(a, b)
            new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
            return new_string
        except AssertionError:
            return string

    def _remove_right_units(self, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing
        # units
        if '\\text{ ' in string:
            splits = string.split('\\text{ ')
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(self, string):
        if '\\sqrt' not in string:
            return string
        splits = string.split('\\sqrt')
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != '{':
                a = split[0]
                new_substr = '\\sqrt{' + a + '}' + split[1:]
            else:
                new_substr = '\\sqrt' + split
            new_string += new_substr
        return new_string

    def _fix_sqrt_v2(self, string):
        _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
        return _string

    def _strip_string(self, string):
        # linebreaks
        string = string.replace('\n', '')

        # remove inverse spaces
        string = string.replace('\\!', '')

        # replace \\ with \
        string = string.replace('\\\\', '\\')

        # replace tfrac and dfrac with frac
        string = string.replace('tfrac', 'frac')
        string = string.replace('dfrac', 'frac')

        # remove \left and \right
        string = string.replace('\\left', '')
        string = string.replace('\\right', '')

        # Remove circ (degrees)
        string = string.replace('^{\\circ}', '')
        string = string.replace('^\\circ', '')

        # remove dollar signs
        string = string.replace('\\$', '')

        # remove units (on the right)
        string = self._remove_right_units(string)

        # remove percentage
        string = string.replace('\\%', '')
        string = string.replace('\%', '')  # noqa: W605

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(' .', ' 0.')
        string = string.replace('{.', '{0.')
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == '.':
            string = '0' + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split('=')) == 2:
            if len(string.split('=')[0]) <= 2:
                string = string.split('=')[1]

        # fix sqrt3 --> sqrt{3}
        string = self._fix_sqrt(string)

        # remove spaces
        string = string.replace(' ', '')

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works
        # with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == '0.5':
            string = '\\frac{1}{2}'

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix
        # in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        return string

    def _strip_string_v2(self, string):
        string = str(string).strip()
        # linebreaks
        string = string.replace('\n', '')

        # right "."
        string = string.rstrip('.')

        # remove inverse spaces
        string = string.replace('\\!', '')
        string = string.replace('\\ ', '')

        # replace \\ with \
        string = string.replace('\\\\', '\\')
        string = string.replace('\\\\', '\\')

        # replace tfrac and dfrac with frac
        string = string.replace('tfrac', 'frac')
        string = string.replace('dfrac', 'frac')

        # remove \left and \right
        string = string.replace('\\left', '')
        string = string.replace('\\right', '')

        # Remove unit: miles, dollars if after is not none
        _string = re.sub(r'\\text{.*?}$', '', string).strip()
        if _string != '' and _string != string:
            string = _string

        # Remove circ (degrees)
        string = string.replace('^{\\circ}', '')
        string = string.replace('^\\circ', '')

        # remove dollar signs
        string = string.replace('\\$', '')
        string = string.replace('$', '')

        string = string.replace('\\text', '')
        string = string.replace('x\\in', '')

        # remove percentage
        string = string.replace('\\%', '')
        string = string.replace('\%', '')  # noqa: W605
        string = string.replace('%', '')

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(' .', ' 0.')
        string = string.replace('{.', '{0.')

        # cdot
        string = string.replace('\\cdot', '')

        # inf
        string = string.replace('infinity', '\\infty')
        if '\\infty' not in string:
            string = string.replace('inf', '\\infty')
        string = string.replace('+\\inity', '\\infty')

        # and
        string = string.replace('and', '')
        string = string.replace('\\mathbf', '')

        # use regex to remove \mbox{...}
        string = re.sub(r'\\mbox{.*?}', '', string)

        # quote
        string.replace("'", '')
        string.replace('"', '')

        # i, j
        if 'j' in string and 'i' not in string:
            string = string.replace('j', 'i')

        # replace a.000b where b is not number or b is end, with ab, use regex
        string = re.sub(r'(\d+)\.0+([^\d])', r'\1\2', string)
        string = re.sub(r'(\d+)\.0+$', r'\1', string)

        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == '.':
            string = '0' + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split('=')) == 2:
            if len(string.split('=')[0]) <= 2:
                string = string.split('=')[1]

        string = self._fix_sqrt_v2(string)
        string = string.replace(' ', '')

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
        # Even works with \frac1{72} (but not \frac{72}1).
        # Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple
        # cases fix in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        return string

    def is_equiv(self, str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print('WARNING: Both None')
            return True
        if str1 is None or str2 is None:
            return False

        if self.version == 'v1':
            strip_string_func = self._strip_string
        elif self.version == 'v2':
            strip_string_func = self._strip_string_v2
        else:
            raise NotImplementedError

        try:
            ss1 = strip_string_func(str1)
            ss2 = strip_string_func(str2)
            if verbose:
                print(ss1, ss2)
            if ss1 == ss2:
                return True
            ss1 = normalize_final_answer(ss1)
            ss2 = normalize_final_answer(ss2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        try:
            ss1 = normalize_final_answer(str1)
            ss2 = normalize_final_answer(str2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        return str1 == str2

    def math_verify_scorer(self, pred, ref):
        try:
            from latex2sympy2_extended import NormalizationConfig
            from math_verify import (ExprExtractionConfig,
                                     LatexExtractionConfig, parse, verify)
        except ImportError:
            raise ImportError('Failed to import required modules. Please '
                              'install the necessary packages: '
                              'pip install math_verify latex2sympy2_extended')

        ref_with_env = f'${ref}$'
        gold_parsed = parse(
            ref_with_env,
            extraction_mode='first_match',
            extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
            ],
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct
            # latex (no malformed operators)
            answer_parsed = parse(
                pred,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed='all',
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode='first_match',
            )

            answer_correct = float(verify(answer_parsed, gold_parsed))
        else:
            answer_correct = False
        return answer_correct

    def math_evaluator2_scorer(self, pred, ref, verbose=False):
        # if self.version == 'v1':
        #     strip_string_func = self._strip_string
        # elif self.version == 'v2':

        strip_string_func = self._strip_string_v2
        # else:
        #     raise NotImplementedError

        try:
            ss1 = strip_string_func(pred)
            ss2 = strip_string_func(ref)
            if verbose:
                print(ss1, ss2)
            if ss1 == ss2:
                return True
            ss1 = normalize_final_answer(ss1)
            ss2 = normalize_final_answer(ss2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        try:
            ss1 = normalize_final_answer(pred)
            ss2 = normalize_final_answer(ref)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        if pred == ref:
            return True
        return False

    def fill_in_the_blank_scorer(self, pred, ref, verbose=False):
        """填空题评分：优先使用数学验证，回退到ROUGE评分."""
        if isinstance(ref, list):
            ref = ref[0]
        if pred is None and ref is None:
            print('WARNING: Both None')
            return True
        if pred is None or ref is None:
            return False
        try:
            pred = json.loads(pred)
            pred = pred[0]
            ref = ref[0]
        except Exception:
            print(f'pred非list形式, pred: {pred}, ref: {ref}')
            # return False

        print(f'pred: {pred}, ref: {ref}')
        print('使用【math_verify】评分')
        if self.math_verify_scorer(pred, ref):
            return True
        print('【math_verify】未通过，使用【math_evaluator2】评分')
        if self.math_evaluator2_scorer(pred, ref):
            return True
        print('【math_evaluator2】未通过，使用【ROUGE】评分')
        return False
        # # 回退到ROUGE评分
        # from rouge_score import rouge_scorer
        # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        # return scorer.score(pred, ref[0])['rougeL'].fmeasure > 0.75

    def subjective_scorer(self, pred, ref):
        return pred == ref

    def score(self, predictions, references, origin_prompt):
        """评分主函数."""
        details = {}
        correct = 0

        for index, (pred, ref) in enumerate(zip(predictions, references)):
            # 关键修复：使用处理后的预测结果进行评分
            if self.question_type == '单选':
                processed_pred = first_option_postprocess(pred, 'ABCD')
            else:
                processed_pred = self.postprocessor_mapping[
                    self.question_type](pred)

            # 使用处理后的预测结果评分
            is_correct = self.scorer_mapping[self.question_type](
                processed_pred, ref)
            correct += is_correct

            details[str(index)] = {
                'prompt': origin_prompt[index],
                'pred': pred,
                'processed_pred': processed_pred,
                'ref': ref,
                'is_correct': is_correct,
            }

        total = len(predictions)
        return {
            'accuracy': correct / total * 100 if total > 0 else 0,
            'details': details
        }


@TEXT_POSTPROCESSORS.register_module('mbgaokao2025_postprocess')
def mbgaokao2025_postprocess(raw_str):
    """通过正则表达式提取答案."""
    raw_str = raw_str.strip()

    # # 尝试不同的正则表达式模式
    # patterns = [
    #     r'答案[:：]\s*(.+)',      # 答案：xxx
    #     r'答案是[:：]\s*(.+)',    # 答案是：xxx
    #     r'答[:：]\s*(.+)',        # 答：xxx
    #     r'解[:：]\s*(.+)',        # 解：xxx
    # ]

    # for pattern in patterns:
    #     if match := re.search(pattern, raw_str):
    #         return match.group(1).strip()

    # 使用任意分隔符匹配
    if any(x in raw_str for x in ['答案', '答', '解']):
        return raw_str.split(':')[-1].split('：')[-1].strip()

    # 如果没有找到任何匹配，返回原始字符串的第一行
    return raw_str.strip().split('\n')[-1].strip()


# 保留这些常量以向后兼容
# valid_mbgaokao2025_question_types = VALID_QUESTION_TYPES
# valid_mbgaokao2025_question_types_dict = QUESTION_TYPE_MAPPING
