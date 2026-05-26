import json
import os
import re

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class ExamBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = {}
        for split in ['shot', 'eval']:
            filename = os.path.join(path, split, f'{name}.jsonl')
            question_type = '_'.join(name.split('_')[2:])
            with open(filename, 'r', encoding='utf-8') as f:
                raw_data = f.readlines()
                raw_data = [json.loads(data) for data in raw_data]

            for entry in raw_data:

                truth = entry['ground_truth']
                target_scores = entry['target_scores']

                # 修改question
                if question_type in [
                        'cloze_the_blank', 'reading_comprehension'
                ]:
                    options = ''
                    for index, opt in enumerate(target_scores):
                        opt_idx = opt.split('. ')[0]
                        opt = '. '.join(opt.split('. ')[1:])
                        if opt_idx not in options:
                            options += \
                                f'\n{opt_idx}. {chr(65+(index%4))}. {opt}'
                        else:
                            options += f'    {chr(65+(index%4))}. {opt}'
                    entry['question'] += f'\n{(options.strip())}'
                elif question_type in ['5_out_of_7', 'multiple_choice']:
                    options = '\n'.join([
                        f'{chr(65+idx)}. {opt}'
                        for idx, opt in enumerate(target_scores.keys())
                    ])
                    entry['question'] += f'\n{(options)}'

                # 修改ground_truth
                if question_type in [
                        'cloze_the_blank', 'reading_comprehension',
                        '5_out_of_7'
                ]:
                    entry['ground_truth'] = '\n'.join(
                        [f'{i+1}. {char}' for i, char in enumerate(truth)])

            dataset[split] = Dataset.from_list(raw_data)
        return DatasetDict(dataset)


valid_exambench_question_types = [
    'cloze_the_blank', 'reading_comprehension', 'multiple_choice',
    'calculation', 'true_false_question', 'fill_in_the_blank', 'word_problem',
    '5_out_of_7'
]


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


class ExamBenchEvaluator(BaseEvaluator):

    def __init__(self, question_type) -> None:
        super().__init__()
        assert question_type in valid_exambench_question_types
        self.question_type = question_type

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

    def _fix_sqrt_v2(self, string):
        _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
        return _string

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

        strip_string_func = self._strip_string_v2

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

    def exambench_answer_postprocess(self, raw_str):
        processed_answer = []
        if self.question_type in [
                'cloze_the_blank', 'reading_comprehension', '5_out_of_7'
        ]:
            matches = re.findall(r'(\d+)[\.\s]*([A-Z])', raw_str)
            # 将提取结果转换为字典
            processed_answer = {num: choice for num, choice in matches}
        elif self.question_type == 'fill_in_the_blank':
            # phi 长输出 \n\n切割
            raw_str = raw_str.strip().split('\n\n')[0]
            raw_answers = re.split(r'[；]', raw_str)
            if len(raw_answers) == 1:
                processed_answer.append(raw_str.strip())
            else:
                processed_answer = [s.strip() for s in raw_answers]
        elif self.question_type == 'multiple_choice':
            match = re.search(r'[A-D]', raw_str)
            processed_answer = match.group(0) if match else ''
        elif self.question_type == 'true_false_question':
            temp = ''
            if '正确' in raw_str and '错误' in raw_str:
                temp = '正确' if raw_str.index('正确') < raw_str.index(
                    '错误') else '错误'
            elif '正确' in raw_str or 'true' in raw_str:
                temp = '正确'
            elif '错误' in raw_str or 'false' in raw_str:
                temp = '错误'
            processed_answer.append(temp)
        # else:
        # processed_answer.append(self.normalize_str(raw_str))
        return processed_answer

    def score(self, predictions, references):
        details = {}
        correct_score, total_score = 0, 0
        for index, (pred, refr) in enumerate(zip(predictions, references)):
            ori_pred = pred
            is_corrects = []
            if self.question_type in [
                    'cloze_the_blank', 'reading_comprehension', '5_out_of_7'
            ]:
                pred = self.exambench_answer_postprocess(pred)
                refr = self.exambench_answer_postprocess(refr)
                if len(pred) != len(refr):
                    print('模型输出的答案长度与预期不符')
                    print(refr)
                for key, value in refr.items():
                    if key in pred and value == pred[key]:
                        correct_score += 1
                        is_corrects.append(True)
                    else:
                        is_corrects.append(False)
                    total_score += 1
            elif self.question_type == 'fill_in_the_blank':
                pred = self.exambench_answer_postprocess(pred)
                refr = self.exambench_answer_postprocess(refr)
                if len(pred) != len(refr):
                    print('模型输出的答案长度与预期不符')
                    print(refr)
                total_score += len(refr)
                for idx in range(min(len(pred), len(refr))):
                    is_correct = self.is_equiv(pred[idx], refr[idx])
                    correct_score += is_correct
                    is_corrects.append((idx, is_correct))
            elif self.question_type == 'true_false_question':
                pred = self.exambench_answer_postprocess(pred)[0]
                is_correct = pred == refr
                correct_score += is_correct
                total_score += 1
                is_corrects.append(is_correct)
            else:
                if self.question_type == 'multiple_choice':
                    pred = self.exambench_answer_postprocess(pred)
                    refr = self.exambench_answer_postprocess(refr)
                    is_correct = pred == refr
                else:
                    # phi 长输出 \n\n切割
                    pred = pred.strip().split('\n\n')[0]
                    is_correct = self.is_equiv(pred, refr)
                correct_score += is_correct
                total_score += 1
                is_corrects.append(is_correct)

            details[str(index)] = {
                'original_pred': ori_pred,
                'pred': pred,
                'refr': refr,
                'is_correct': is_corrects
            }
        return {'score': correct_score / total_score * 100, 'details': details}


for question_type in valid_exambench_question_types:
    # fix classic closure problem
    def _exambench_register(question_type):
        ICL_EVALUATORS.register_module(
            name='ExamBenchEvaluator' + '_' + question_type,
            module=lambda *args, **kwargs: ExamBenchEvaluator(
                question_type=question_type, *args, **kwargs))

    _exambench_register(question_type)
