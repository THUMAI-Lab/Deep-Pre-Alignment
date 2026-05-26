import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests
from openai import APIStatusError, BadRequestError, OpenAI

from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


def add_jsonl(js_content, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(js_content, ensure_ascii=False) + '\n')
    return


class BaiduyunAPI(BaseAPIModel):
    """Model wrapper around SiliconFlowAPI.

    Documentation:

    Args:
        path (str): The name of SiliconFlowAPI model.
            e.g. `moonshot-v1-32k`
        key (str): Authorization key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """
    is_api: bool = True

    def __init__(
        self,
        path: str,
        key: str,
        url: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 100,
        system_prompt: str = '',
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        # self.headers = {
        #     'Content-Type': 'application/json',
        #     'Authorization': 'Bearer ' + key,
        # }
        self.url = url
        self.model = path
        self.system_prompt = system_prompt

        self.client = OpenAI(
            # 从环境变量中读取您的方舟API Key
            api_key=key,
            base_url=self.url,
        )

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        self.flush()
        return results

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        work_dir = os.environ['work_dir']
        reasoning_folder = os.path.join(work_dir, 'reasoning')
        os.makedirs(reasoning_folder, exist_ok=True)

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            msg_buffer, last_role = [], None
            for item in input:
                item['role'] = 'assistant' if item['role'] == 'BOT' else 'user'
                if item['role'] != last_role and last_role is not None:
                    messages.append({
                        'content': '\n'.join(msg_buffer),
                        'role': last_role
                    })
                    msg_buffer = []
                msg_buffer.append(item['prompt'])
                last_role = item['role']
            messages.append({
                'content': '\n'.join(msg_buffer),
                'role': last_role
            })

        # if self.system_prompt:
        #     system = {'role': 'system', 'content': self.system_prompt}
        #     messages.insert(0, system)

        print('messages', messages)

        max_num_retries = 0
        while max_num_retries < self.retry:

            self.acquire()
            try:
                # raw_response = requests.request('POST',
                #                                 url=self.url,
                #                                 headers=self.headers,
                #                                 json=data)
                s_time = time.time()
                responses = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.6,
                    top_p=0.95,
                )
                e_time = time.time()
                print(responses)

                print('[one request time]', e_time - s_time)

                if not responses.choices:
                    self.logger.error(
                        'Response is empty, it is an internal server error \
                            from the API provider.')
                try:
                    print('### think process start ###')
                    print(responses.choices[0].message.reasoning_content)
                    print('### think process end ###')

                    # print(responses)
                    add_jsonl(
                        {
                            'prompt': messages,
                            'reasoning':
                            responses.choices[0].message.reasoning_content,
                            'generated_text':
                            responses.choices[0].message.content
                        }, os.path.join(reasoning_folder, 'reasoning.jsonl'))
                except Exception as e:
                    print(e)
                return responses.choices[0].message.content

                # raw_answer = responses.choices[0].message.content
                # generated_text = raw_answer
                # if "</think>" in generated_text:
                #     print(f"Think Progress: {generated_text.split('</think>')[0].strip()!r}")
                #     generated_text = generated_text.split("</think>")[-1].strip()
                #     print(f"Generated Text: {generated_text!r}")
                # else:
                #     print("*** NO SPECIAL TOKEN**")
                #     print(f"Generated Text: {generated_text!r}")
                # return generated_text

            # except Exception as err:
            #     print('Request Error:{}'.format(err))
            #     time.sleep(2)
            #     continue

            # try:
            #     response = raw_response#.json()
            except Exception as err:
                max_num_retries += 1
                print('Response Error:{}'.format(err))
                response = None
            self.release()

            # if response is None:
            #     print('Connection error, reconnect.')
            #     # if connect error, frequent requests will casuse
            #     # continuous unstable network, therefore wait here
            #     # to slow down the request
            #     self.wait()
            #     continue

            # if raw_response.status_code == 200:
            #     # msg = json.load(response.text)
            #     # response
            #     # msg = response['choices'][0]['message']['content']

            #     self.logger.debug(f'Generated: {msg}')
            #     generated_text = msg
            #     if "</think>" in generated_text:
            #         print(f"think progress: {generated_text.split('</think>')[0].strip()!r}")
            #         generated_text = generated_text.split("</think>")[-1].strip()
            #         print(f"Generated text: {generated_text!r}")
            #     else:
            #         print(f"Generated text: {generated_text!r}")
            #         print("*** NO SPECIAL TOKEN**")
            #     return generated_text

            # if raw_response.status_code == 401:
            #     print('请求被拒绝 api_key错误')
            #     continue
            # elif raw_response.status_code == 400:
            #     print(messages, response)
            #     print('请求失败，状态码:', raw_response)
            #     msg = 'The request was rejected because high risk'
            #     return msg
            # elif raw_response.status_code == 429:
            #     print(messages, response)
            #     print('请求失败，状态码:', raw_response)
            #     time.sleep(5)
            #     continue
            # else:
            #     print(messages, response)
            #     print('请求失败，状态码:', raw_response)
            #     time.sleep(1)

        raise RuntimeError(responses)
