import logging
import time  # 如果需要在重试之间添加延迟，可以使用 time.sleep
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

logger = logging.getLogger(__name__)

load_dotenv()

API_URL = 'https://minicpm3-4b.modelbest.cn/v1/chat/completions'  # 替换为实际的 API URL
API_KEY = 'modelbesta5a3ceabaa9f'  # 替换为你的 API 密钥

PromptType = Union[PromptList, str]


def get_api_response(question, temperature=0.7, top_p=0.7, max_tokens=512):
    # def ask_question(question, temperature=0.7, top_p=0.7, repetition_penalty=1):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }

    # 请求体数据
    data = {
        'model': 'minicpm3-4b',
        'messages': [{
            'role': 'user',
            'content': question
        }],
        # "temperature": temperature,
        # "top_p": top_p,
        # "repetition_penalty": repetition_penalty,
        'max_tokens': max_tokens,
        'frequency_penalty': 0.0,
        'length_penalty': 1.0,
        # "repetition_penalty": 1.0,
        # "temperature": 0.3,
        # "top_p": 0.8,
        'temperature': 0.7,
        'top_p': 0.7,
        'repetition_penalty': 1,
        'top_k': 1,
    }

    response = requests.post(API_URL, json=data, headers=headers)
    try:
        # 发送 POST 请求
        answer = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        print(response)
        print(response.text)
        return ''
    return answer


@MODELS.register_module()
class ModelBestApi(BaseAPIModel):
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
            # rpm_verbose: bool = False,
            retry: int = 2,
            # key: str = 'None',
            # org: Optional[Union[str, List[str]]] = None,
            meta_template: Optional[Dict] = None,
            # openai_api_base: str = OPENAI_API_BASE,
            # mode: str = 'none',
            app_token: str = '',
            logprobs: Optional[bool] = False,
            top_logprobs: Optional[int] = None,
            temperature: Optional[float] = None,
            tokenizer_path: Optional[str] = None,
            model_id: Optional[int] = None):

        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            query_per_second=query_per_second,
            # rpm_verbose=rpm_verbose,
            retry=retry)

        self.temperature = temperature
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.tokenizer_path = tokenizer_path
        self.model_id = model_id
        self.app_token = app_token

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        # if isinstance(org, str):
        #     self.orgs = [org]
        # else:
        #     self.orgs = org
        self.org_ctr = 0
        # self.url = openai_api_base
        self.path = path

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 temperature: float = 0.7,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        # print(inputs)
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
        # def _inference(model_id, prompt, **kwargs) -> str:
        # access_token = await self.get_access_token()

        assert isinstance(input, (str, PromptList))

        # if isinstance(input, str):
        #     # messages = [{'role': 'user', 'content': input}]
        #     messages = [{'role': 'USER', 'contents': [{'type': 'TEXT', 'pairs': input}]}]
        # else:
        #     print(input)
        assert isinstance(input, str)
        max_retries = 5  # 最大重试次数

        for attempt in range(1, max_retries + 1):
            try:
                # infer.infer(audio_path, system_prompt.format(text=text), save_name=save_path)
                answer = get_api_response(input,
                                          temperature=temperature,
                                          max_tokens=max_out_len)
                if answer:
                    return answer
                print('空输出')
                # print("Task succeeded!")
                # break  # 如果任务成功，退出循环
            except Exception as e:
                print(e)
                print(f'Attempt {attempt} failed: {e}')
                if attempt < max_retries:
                    print('Retrying...')
                    time.sleep(1)  # 可选，等待1秒后重试
                else:
                    print('All retries failed. Task aborted.')
