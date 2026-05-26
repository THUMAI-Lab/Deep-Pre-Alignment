import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from opencompass.models.huggingface_above_v4_33 import _convert_chat_messages
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


def luca_generate(prompt):
    url = 'https://load-balancer.ali.modelbest.co/api/v1/api'

    headers = {
        'X-Model-Best-Model': 'tainshu-80b-int4-compare',
        'X-Model-Best-Trace-ID': 'test-trace2',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    data = {
        'query':
        f'<用户>你叫露卡，是由面壁智能和知乎联合研发的大型语言模型。\n你的知识库截止至2022年4月，当前时间是2023年9月15日。<用户>{prompt}<AI>',
        'config': {
            'max_length': 4096,
            'ngram_penalty': 1.02,
            # "num_results": 1,
            'repetition_penalty': 1.05,
            'seed': 1499431418,
            'temperature': 0.1,
            'stop_token_ids': [2, 119690],
            'top_p': 0.1,
            'type': 'random'
        },
    }
    response = requests.post(url, headers=headers, json=data)
    # print(prompt)
    print(response.text)
    time.sleep(2)
    # print(response.json()["data"].get("result"))
    return response.json()['data'].get('result')


@MODELS.register_module()
class LucaLLMCenterAPI(BaseAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        path (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        openai_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
        tokenizer_path (str, optional): The path to the tokenizer. Use path if
            'tokenizer_path' is None, otherwise use the 'tokenizer_path'.
            Defaults to None.
    """

    is_api: bool = True

    def __init__(
            self,
            path: str = '',
            max_seq_len: int = 4096,
            query_per_second: int = 1,
            retry: int = 2,
            meta_template: Optional[Dict] = None,
            # openai_api_base: str = OPENAI_API_BASE,
            # mode: str = 'none',
            app_token: str = '',
            logprobs: Optional[bool] = False,
            top_logprobs: Optional[int] = None,
            temperature: Optional[float] = None,
            tokenizer_path: Optional[str] = None,
            model_id: Optional[int] = None):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)

        self.temperature = temperature
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.tokenizer_path = tokenizer_path
        self.model_id = model_id
        self.app_token = app_token

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 temperature: float = 0.7,
                 **kwargs) -> List[str]:
        if self.temperature is not None:
            temperature = self.temperature
            kwargs['temperature'] = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs),
                             [kwargs] * len(inputs)))

        return results

    def _generate(self, input: PromptType, max_out_len: int,
                  temperature: float, kwargs: dict) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        # assert isinstance(input, (str, PromptList))
        # print(input)
        # print(type(input))
        assert isinstance(input, str)

        # messages = _convert_chat_messages(inputs)

        # context = trans_to_template(messages)
        context = input
        response = luca_generate(context)

        return response
