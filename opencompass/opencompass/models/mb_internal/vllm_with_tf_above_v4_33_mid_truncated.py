# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional

from opencompass.models.huggingface_above_v4_33 import (
    _convert_chat_messages, _format_with_fast_chat_template,
    _get_meta_template, _get_possible_max_seq_len)
from opencompass.models.vllm_with_tf_above_v4_33 import VLLMwithChatTemplate
from opencompass.registry import MODELS

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None


@MODELS.register_module()
class VLLMwithChatTemplateLongTruncated(VLLMwithChatTemplate):
    """An Internal Model wrapper for HuggingFace models designed for chat."""

    def __init__(self,
                 *args,
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 enable_thinking: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['none', 'mid']
        self.mode = mode
        self.enable_thinking = enable_thinking
        self.logger.info(
            f'Use {mode} mode for truncation for long text prompts.')

    def mid_truncated(self, message, max_prompt_len):
        truncated_message = message
        half_max_prompt_len = max_prompt_len // 2
        tokens = self.tokenizer.encode(message)
        if len(tokens) > max_prompt_len:
            self.logger.warning('=' * 100)
            self.logger.warning(
                "This prompt exceed the model's predefined maximum length.")
            self.logger.warning('=' * 100)
            # 避免边界情况
            front = tokens[:half_max_prompt_len - 1]
            back = tokens[-(half_max_prompt_len + 1):]
            truncated_tokens = front + back
            truncated_message = self.tokenizer.decode(truncated_tokens)
        return truncated_message

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
            if self.enable_thinking:
                messages = [self.tokenizer.apply_chat_template(
                    m, add_generation_prompt=True, tokenize=False, enable_thinking=True) for m in messages]
            else:
                self.logger.info('[启用 Qwen3 非深思模式] enable_thinking=False')
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

        if self.mode == 'mid':
            # Reserve space for the tokens to be generated in the future.
            max_prompt_len = self.max_seq_len - max_out_len

            # Retain the first 0.5 * max_prompt_len tokens and the last 0.5 * max_prompt_len tokens, discarding the middle ones,
            # because the prompts' questions are usually at the beginning or the end.
            # To avoid the warning:
            # This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length.
            # Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
            messages = [self.mid_truncated(m, max_prompt_len)
                        for m in messages]

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
