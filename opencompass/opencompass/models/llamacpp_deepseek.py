import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel
from .huggingface_above_v4_33 import (_convert_chat_messages,
                                      _format_with_fast_chat_template,
                                      _get_meta_template,
                                      _get_possible_max_seq_len)

PromptType = Union[PromptList, str]


def trans_chat_template(prompt):
    # <｜begin▁of▁sentence｜>
    # <｜User｜>Tell me something about large language models.<｜Assistant｜>
    # <｜User｜>\nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\nPlease reason step by step, and put your final answer within \\boxed{}.<｜Assistant｜>\n<think>\n
    # str_prompt = "<|im_start|>user\n"
    # str_prompt = "<|user|>\n"
    # str_prompt = "<｜begin▁of▁sentence｜>User: "
    str_prompt = '<｜User｜>'

    # for prompt in prompt_list:
    str_prompt += prompt

    # str_prompt += "<|im_end|>\n<|im_start|> assistant\n"
    # str_prompt += "<|end|>\n<|assistant|>\n"
    # str_prompt += "\nAssistant:"
    str_prompt += '<｜Assistant｜>'
    # str_prompt = f"<｜User｜>\n{prompt}<｜Assistant｜>\n<think>\n"
    return str_prompt


def add_jsonl(js_content, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(js_content, ensure_ascii=False) + '\n')
    return


@MODELS.register_module()
class LlamaCppDeepSeek(BaseAPIModel):
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
            path: str = 'llamacpp_DeepSeek-R1-UD-IQ1_S',
            max_seq_len: int = 4096,
            query_per_second: int = 1,
            # rpm_verbose: bool = False,
            retry: int = 2,
            # key: str = 'None',
            # org: Optional[Union[str, List[str]]] = None,
            #  meta_template: Optional[Dict] = None,
            generation_kwargs: dict = dict(),
            # openai_api_base: str = OPENAI_API_BASE,
            # mode: str = 'none',
            app_token: str = '',
            logprobs: Optional[bool] = False,
            top_logprobs: Optional[int] = None,
            temperature: Optional[float] = None,
            tokenizer_path: Optional[str] = None,
            model_id: Optional[int] = None,
            port: Optional[int] = None):

        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            #  meta_template=meta_template,
            query_per_second=query_per_second,
            generation_kwargs=generation_kwargs,
            # rpm_verbose=rpm_verbose,
            retry=retry)

        # import tiktoken
        self.logger = get_logger()
        # self.tiktoken = tiktoken
        self.temperature = temperature
        self.generation_kwargs = generation_kwargs
        # assert mode in ['none', 'front', 'mid', 'rear']
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.tokenizer_path = tokenizer_path
        self.model_id = model_id
        self.app_token = app_token

        self.invalid_keys = set()

        self.key_ctr = 0
        self.port = port.split('llamacpp_deepseek_')[-1]
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
        if self.temperature is not None:
            temperature = self.temperature
            kwargs['temperature'] = self.temperature

        results = []
        # for input in inputs:
        #     results.append(
        #         self._generate
        #         (input,
        #         max_out_len,
        #         temperature,
        #         kwargs)
        #         )
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
        assert isinstance(input, (str, PromptList))

        work_dir = os.environ['work_dir']
        reasoning_folder = os.path.join(work_dir, 'reasoning')
        os.makedirs(reasoning_folder, exist_ok=True)
        # for prompt in input:
        #     print(prompt)
        # if isinstance(input, str):
        #     # messages = [{'role': 'user', 'content': input}]
        #     messages = [{'role': 'USER', 'contents': [{'type': 'TEXT', 'pairs': input}]}]
        # else:
        # prompt = trans_chat_template(input)
        # for input_ in input:
        #     print(input_)
        prompt = trans_chat_template(input)
        self.logger.info('-' * 100)
        self.logger.info(f'Prompt: {prompt!r}')
        self.logger.info('-' * 100)
        # from datetime import datetime

        # # 获取当前日期和时间
        # now = datetime.now()

        # # 格式化为字符串
        # date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        # with open("llamacpp_debug.log", "a") as f:
        #     f.write(date_time_str + "\n")
        #     f.write("-----" + "\n")
        #     f.write(f"{[prompt]}" + "\n\n")
        # os.exit()
        #     assert isinstance(input, str)
        # messages = []
        # for item in input:
        #     msg = {'content': item['prompt']}
        #     if item['role'] == 'HUMAN':
        #         msg['role'] = 'user'
        #     elif item['role'] == 'BOT':
        #         msg['role'] = 'assistant'
        #     elif item['role'] == 'SYSTEM':
        #         msg['role'] = 'system'
        #     messages.append(msg)

        # for prompt in input:

        url = f'http://localhost:{self.port}/completion'
        headers = {'Content-Type': 'application/json'}

        data = {
            'n_predict': 32768,
            'temperature': 0.6,
            'top_p': 0.95,
            'prompt': f'{prompt}'
        }

        # param_map = {
        #     "max_out_len": "n_predict",
        #     "top_k": "top_k",
        #     "top_p": "top_p",
        #     "repetition_penalty": "repeat_penalty",
        #     "presence_penalty": "presence_penalty",
        #     "frequency_penalty": "frequency_penalty",
        #     "temperature": "temperature",
        # }

        # for param in param_map:
        #     if param in kwargs:
        #         data[param_map[param]] = kwargs[param]

        retry = 1
        while retry:
            # prompt = ""
            reasoning = ''
            generated_text = ''
            try:
                s_time = time.time()
                response = requests.post(url, headers=headers, json=data)
                e_time = time.time()

                self.logger.info(f'[one request time], {e_time - s_time}')

                # if response.status_code == 200:
                self.logger.info(f'{response.text}')
                msg_data = response.json()
                generated_text = msg_data['content']
                if '</think>' in generated_text:
                    reasoning = generated_text.split('</think>')[0].strip()
                    self.logger.info(f'think progress: {reasoning!r}')
                    generated_text = generated_text.split(
                        '</think>')[-1].strip()
                    self.logger.info(f'Generated text: {generated_text!r}')
                else:
                    self.logger.info(f'Generated text: {generated_text!r}')
                    self.logger.info('*** NO SPECIAL TOKEN**')
                add_jsonl(
                    {
                        'prompt': prompt,
                        'reasoning': reasoning,
                        'generated_text': generated_text
                    }, os.path.join(reasoning_folder, 'reasoning.jsonl'))
                return generated_text

            except Exception as err:
                retry -= 1
                #     max_num_retries += 1
                self.logger.error('Response Error:{}'.format(err))

        self.logger.error('####[WRONG RETURN]####')
        return '####[WRONG RETURN]####'

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """

        return len(prompt.split(' '))
