# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional

import numpy as np

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

from .huggingface_above_v4_33 import (_convert_chat_messages,
                                      _format_with_fast_chat_template,
                                      _get_meta_template,
                                      _get_possible_max_seq_len)
from .vllm_with_tf_above_v4_33 import VLLMwithChatTemplate

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None


class Qwen3VLLM(VLLMwithChatTemplate):

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
        enable_thinking: bool = True,
        mode: str = 'none'
    ):
        assert LLM, ('Please install VLLM with `pip install vllm`. note: torch==2.1.2 is required.')
        assert mode in ['none', 'mid'], 'mode must be one of none, mid'
        self.mode = mode
        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self.enable_thinking = enable_thinking
        if tokenizer_only:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        else:
            self._load_model(path, model_kwargs)
            self.tokenizer = self.model.get_tokenizer()

        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))

    def mid_truncated(self, message, max_prompt_len):
        truncated_message = message
        half_max_prompt_len = max_prompt_len // 2
        tokens = self.tokenizer.encode(message)
        if len(tokens) > max_prompt_len:
            self.logger.warning('=' * 100)
            self.logger.warning("This prompt exceed the model's predefined maximum length.")
            self.logger.warning('=' * 100)
            # 避免边界情况
            front = tokens[:half_max_prompt_len - 1]
            back = tokens[-(half_max_prompt_len + 1):]
            truncated_tokens = front + back
            truncated_message = self.tokenizer.decode(truncated_tokens)
        return truncated_message

    def _load_model(self, path: str, added_model_kwargs: dict = dict()):
        import ray

        if ray.is_initialized():
            self.logger.info('shutdown ray instance to avoid "Calling ray.init() again" error.')
            ray.shutdown()

        DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        model_kwargs.update(added_model_kwargs)
        self.model = LLM(path, **model_kwargs)

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

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
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
        else:
            if self.enable_thinking:
                self.logger.info('[启用 深思模式] enable_thinking=True')
            else:
                self.logger.info('[启用 非深思模式] enable_thinking=False')
            # try:
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking) for m in messages]
            # except Exception as e:
            #     self.logger.error(f"Error get enable_thinking: {e}")
            #     messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
            # vLLM tokenize prompts by AutoTokenizer with its default parameter "add_special_token=True"
            # OC add bos_token in the prompt, which requires tokenizing prompts using "add_speicial_token=False"
            # But vLLM doesn't have "add_speicial_token" in the pipeline API. So, we remove bos_token
            # from messages as a workaround
            self.logger.info(f'do not removeprefix: {self.tokenizer.bos_token}')
            # if self.tokenizer.bos_token:
            #     bos_token = self.tokenizer.bos_token
            #     messages = [message.removeprefix(bos_token) if message.startswith(bos_token) else message for message in messages]

        if self.mode == 'mid':
            # Reserve space for the tokens to be generated in the future.
            max_prompt_len = self.max_seq_len - max_out_len
            self.logger.info('enter mid: max_out_len: {}, self.max_seq_len: {}, max_prompt_len: {}'.format(max_out_len, self.max_seq_len, max_prompt_len))

            # Retain the first 0.5 * max_prompt_len tokens and the last 0.5 * max_prompt_len tokens, discarding the middle ones,
            # because the prompts' questions are usually at the beginning or the end.
            # To avoid the warning:
            # This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length.
            # Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
            messages = [self.mid_truncated(m, max_prompt_len) for m in messages]

        DEFAULT_GENERATION_KWARGS = {
            'temperature': 0,
            'max_tokens': max_out_len,
            'stop': list(set(self.stop_words + stopping_criteria))
        }
        sampling_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        sampling_kwargs.update(self.generation_kwargs)
        sampling_kwargs.update(kwargs)
        sampling_kwargs = SamplingParams(**sampling_kwargs)
        self.logger.info(sampling_kwargs)
        self.logger.info(f'messages: {messages}')

        outputs = self.model.generate(messages, sampling_kwargs)

        prompt_list, output_strs = [], []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            prompt_list.append(prompt)
            output_strs.append(generated_text)
            self.logger.info(f'generated_text: {generated_text}')

        return output_strs
