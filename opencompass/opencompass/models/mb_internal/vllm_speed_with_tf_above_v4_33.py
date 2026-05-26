# flake8: noqa
# yapf: disable
import time
from typing import Dict, List, Optional

import numpy as np

from opencompass.models.base import BaseModel
from opencompass.models.huggingface_above_v4_33 import (
    _convert_chat_messages, _format_with_fast_chat_template,
    _get_meta_template, _get_possible_max_seq_len)
from opencompass.models.vllm_with_tf_above_v4_33 import VLLMwithChatTemplate
from opencompass.utils import get_logger

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None


class VLLMSpeedwithChatTemplate(VLLMwithChatTemplate):

    def __init__(
        self,
        path: str,
        model_kwargs: dict = dict(),
        tokenizer_only: bool = False,
        generation_kwargs: dict = dict(),
        max_seq_len: int = None,
        meta_template: Optional[Dict] = None,
        fastchat_template: Optional[str] = None,
        stop_words: List[str] = [],
    ):
        assert LLM, ('Please install VLLM with `pip install vllm`. note: torch==2.1.2 is required.')

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        if tokenizer_only:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True)
        else:
            self._load_model(path, model_kwargs)
            self.tokenizer = self.model.get_tokenizer()

        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)
        self.fastchat_template = fastchat_template
        self.stop_words = list(
            set(stop_words + self._get_potential_stop_words(path)))

    def generate(self, inputs: List[str], max_out_len: int, stopping_criteria: List[str] = [], **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        messages = _convert_chat_messages(inputs)
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(
                messages, self.fastchat_template)
        else:
            messages = [self.tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False) for m in messages]
            # vLLM tokenize prompts by AutoTokenizer with its default parameter "add_special_token=True"
            # OC add bos_token in the prompt, which requires tokenizing prompts using "add_speicial_token=False"
            # But vLLM doesn't have "add_speicial_token" in the pipeline API. So, we remove bos_token
            # from messages as a workaround
            if self.tokenizer.bos_token:
                bos_token = self.tokenizer.bos_token
                messages = [message.removeprefix(bos_token) if message.startswith(
                    bos_token) else message for message in messages]
        DEFAULT_GENERATION_KWARGS = {
            'temperature': 0,
            'max_tokens': max_out_len,
            'stop': list(set(self.stop_words + stopping_criteria))
        }
        sampling_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        sampling_kwargs.update(self.generation_kwargs)
        sampling_kwargs.update(kwargs)
        sampling_kwargs = SamplingParams(**sampling_kwargs)
        self.logger.info('Sampling Params of vLLM: ')
        self.logger.info(sampling_kwargs)

        time_start = time.time()
        outputs = self.model.generate(messages, sampling_kwargs)
        time_end = time.time()
        processing_time = time_end - time_start
        self.logger.info(f'Prompt: {messages}, length: {len(messages)}')
        self.logger.info(f'Processing time: {processing_time} seconds')
        self.logger.info(f'Processing time per message: {processing_time / len(messages)} seconds')

        prompt_list, output_strs = [], []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            token_num = len(output.outputs[0].token_ids)
            token_per_second = token_num / processing_time
            self.logger.info(f'Token num: {token_num}, Token per second: {token_per_second}')

            generated_text += f'[####本数据集用于模型测速####]{token_num}'
            generated_text += f'[####本数据集用于模型测速####]{processing_time}'
            generated_text += f'[####本数据集用于模型测速####]{token_per_second}'
            prompt_list.append(prompt)
            output_strs.append(generated_text)
            self.logger.info(f'prompt:\n{prompt}')
            self.logger.info(f'generated output:\n{generated_text}')

        return output_strs
