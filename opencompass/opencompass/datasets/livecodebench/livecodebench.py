# Copyright (c) 2024, LiveCodeBench and its contributors.
# Copyright (c) 2023, OpenCompass and its contributors.

import base64
import json
import os
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from opencompass.utils import get_data_path  # noqa: F401, F403

from ..base import BaseDataset
from .prompts import SelfRepairPromptConstants  # noqa: F401, F403
from .prompts import TestOutputPromptConstants  # noqa: F401, F403
from .prompts import (CodeGenerationPromptConstants,
                      get_generic_question_template_answer_self_repair,
                      get_generic_question_template_test_completion,
                      make_code_execution_prompt)


class Platform(Enum):
    LEETCODE = 'leetcode'
    CODEFORCES = 'codeforces'
    ATCODER = 'atcoder'


class Difficulty(Enum):
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'


class TestType(Enum):
    STDIN = 'stdin'
    FUNCTIONAL = 'functional'


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


class LCBCodeGenerationDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'opencompass/code_generation_lite',
             local_mode: bool = False,
             release_version: str = 'release_v1',
             start_date: str = None,
             end_date: str = None):

        def transform(item):
            # Define the dataitem mapping logic

            # starter_code
            if item['starter_code']:
                format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'  # noqa: E501
                format_prompt += f"```python\n{item['starter_code']}\n```\n\n"  # noqa: Q000, E501
            else:
                format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n'  # noqa: E501
                format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'

            item['format_prompt'] = format_prompt

            # load test cases
            public_test_cases = item['public_test_cases']
            public_test_cases = json.loads(item['public_test_cases'])

            private_test_cases = item['private_test_cases']
            try:
                private_test_cases = json.loads(item['private_test_cases'])
            except Exception as e:  # noqa: F841
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(private_test_cases.encode(
                                'utf-8'))  # type: ignore
                        )))  # type: ignore

            # load metadata
            metadata = json.loads(item['metadata'])
            evaluation_sample = json.dumps({
                'inputs':
                [t['input'] for t in public_test_cases + private_test_cases],
                'outputs':
                [t['output'] for t in public_test_cases + private_test_cases],
                'fn_name':
                metadata.get('func_name', None),
            })
            item['evaluation_sample'] = evaluation_sample

            return item

        path = get_data_path(path, local_mode=local_mode)

        # Define which files to load based on release version
        ALLOWED_FILES = {
            'release_v1': ['test.jsonl'],
            'release_v2': [
                'test.jsonl',
                'test2.jsonl',
            ],
            'release_v3': [
                'test.jsonl',
                'test2.jsonl',
                'test3.jsonl',
            ],
            'release_v4': [
                'test.jsonl',
                'test2.jsonl',
                'test3.jsonl',
                'test4.jsonl',
            ],
            'release_v5': [
                'test.jsonl',
                'test2.jsonl',
                'test3.jsonl',
                'test4.jsonl',
                'test5.jsonl',
            ],
            'release_v6': [
                'test.jsonl',
                'test2.jsonl',
                'test3.jsonl',
                'test4.jsonl',
                'test5.jsonl',
                'test6.jsonl',
            ],
        }
        v_list = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        for v in v_list:
            ALLOWED_FILES[v] = [
                f'test{v[1:]}.jsonl' if v != 'v1' else 'test.jsonl'
            ]
        n_vs = len(v_list)
        for idx1 in range(1, n_vs + 1):
            for idx2 in range(idx1 + 1, n_vs + 1):
                ALLOWED_FILES[v_list[idx1 - 1] + '_' + v_list[idx2 - 1]] = [
                    f'test{idx}.jsonl' if idx != 1 else 'test.jsonl'
                    for idx in range(idx1, idx2 + 1)
                ]

        # Load data from JSONL files directly
        files_to_load = ALLOWED_FILES.get(release_version)
        all_data = []

        for filename in files_to_load:
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line.strip()))

        # Create dataset from the loaded data
        dataset = Dataset.from_list(all_data)
        dataset = dataset.map(transform)

        if start_date is not None:
            p_start_date = datetime.strptime(start_date, '%Y-%m-%d')
            dataset = dataset.filter(
                lambda e: p_start_date <= datetime.fromisoformat(e[
                    'contest_date']))  # noqa: E501
        if end_date is not None:
            p_end_date = datetime.strptime(end_date, '%Y-%m-%d')
            dataset = dataset.filter(lambda e: datetime.fromisoformat(e[
                'contest_date']) <= p_end_date)  # noqa: E501

        return DatasetDict({'test': dataset, 'train': dataset})


class LCBCodeExecutionDataset(BaseDataset):

    @staticmethod
    def load(
        path: str = 'opencompass/execution-v2',
        local_mode: bool = False,
        cot: bool = False,
        # release_version: str = "release_v1"
    ):
        # path = get_data_path(path, local_mode=local_mode)

        def transform(item):
            code, input = item['code'], item['input']
            prompt = make_code_execution_prompt(code, input, cot=cot)

            item['prompt'] = prompt

            evaluation_sample = json.dumps({
                'code': item['code'],
                'input': item['input'],
                'output': item['output']
            })
            item['evaluation_sample'] = evaluation_sample

            return item

        path = get_data_path(path, local_mode=local_mode)
        dataset = load_dataset(path,
                               split='test')  # 'livecodebench/execution-v2'
        dataset = dataset.map(transform)

        return DatasetDict({'test': dataset, 'train': dataset})


class LCBTestOutputPredictionDataset(BaseDataset):

    @staticmethod
    def load(
        path: str = 'opencompass/test_generation',
        local_mode: bool = False,
        # release_version: str = "release_v1"
    ):
        # path = get_data_path(path, local_mode=local_mode)

        def transform(item):
            question_content = item['question_content']
            starter_code = item['starter_code']
            test = json.loads(item['test'])

            testcase_input = test[0]['input']
            testcase_output = test[0]['output']

            item['testcase_input'] = testcase_input
            item['testcase_output'] = testcase_output

            item['prompt'] = get_generic_question_template_test_completion(
                question_content=question_content,
                starter_code=starter_code,
                testcase_input=testcase_input)

            evaluation_sample = json.dumps({
                'input':
                item['question_content'],
                'output':
                json.loads(item['test'])[0]['output']
            })
            item['evaluation_sample'] = evaluation_sample

            return item

        path = get_data_path(path, local_mode=local_mode)
        # 'livecodebench/test_generation',
        dataset = load_dataset(path, split='test')
        dataset = dataset.map(transform)

        return DatasetDict({'test': dataset, 'train': dataset})


class LCBSelfRepairDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'livecodebench/code_generation_lite',
             local_mode: bool = False,
             release_version: str = 'release_v1'):

        def transform(item):
            # define the data item mapping logic

            question = item['question_content']
            code = item['code_list'][0]
            metadata = item['metadata']

            prompt = get_generic_question_template_answer_self_repair(
                question=question, code=code, metadata=metadata)
            item['prompt'] = prompt

            return item

        # Define which files to load based on release version
        ALLOWED_FILES = {
            'release_v1': ['test.jsonl'],
            'release_v2': ['test.jsonl', 'test2.jsonl'],
            'release_v3': ['test.jsonl', 'test2.jsonl', 'test3.jsonl'],
            'release_v4':
            ['test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl'],
            'release_v5': [
                'test.jsonl',
                'test2.jsonl',
                'test3.jsonl',
                'test4.jsonl',
                'test5.jsonl',
            ],
            'release_v6': [
                'test.jsonl',
                'test2.jsonl',
                'test3.jsonl',
                'test4.jsonl',
                'test5.jsonl',
                'test6.jsonl',
            ],
        }

        # Get the actual path - if it's the default, use code_generation_lite
        if path == 'livecodebench/code_generation_lite':
            actual_path = get_data_path('opencompass/code_generation_lite',
                                        local_mode=local_mode)
        else:
            actual_path = get_data_path(path, local_mode=local_mode)

        # Load data from JSONL files directly
        files_to_load = ALLOWED_FILES.get(release_version, ['test.jsonl'])
        all_data = []

        for filename in files_to_load:
            file_path = os.path.join(actual_path, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line.strip()))

        # Create dataset from the loaded data
        dataset = Dataset.from_list(all_data)
        dataset = dataset.map(transform)

        return DatasetDict({'test': dataset, 'train': dataset})


class CompassBenchCodeExecutionDataset(BaseDataset):

    @staticmethod
    def load(
        path: str = 'opencompass/execution-v2',
        local_mode: bool = False,
        cot: bool = False,
        # release_version: str = "release_v1"
    ):
        # path = get_data_path(path, local_mode=local_mode)

        def transform(item):
            code, input = item['code'], item['input']
            prompt = make_code_execution_prompt(code, input, cot=cot)

            item['prompt'] = prompt

            evaluation_sample = json.dumps({
                'code': item['code'],
                'input': item['input'],
                'output': item['output']
            })
            item['evaluation_sample'] = evaluation_sample

            return item

        path = get_data_path(path, local_mode=local_mode)
        dataset = load_from_disk(path)  # 'livecodebench/execution-v2'
        dataset = dataset['test']
        dataset = dataset.map(transform)

        return DatasetDict({'test': dataset, 'train': dataset})
