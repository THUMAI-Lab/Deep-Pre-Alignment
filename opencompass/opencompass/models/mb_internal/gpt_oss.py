# flake8: noqa
# yapf: disable
import os
from typing import Dict, List, Literal, Optional

import numpy as np

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f'vllm not found {e}')
    LLM, SamplingParams = None, None

try:
    from openai_harmony import (Conversation, DeveloperContent,
                                HarmonyEncodingName, Message, ReasoningEffort,
                                Role, SystemContent, load_harmony_encoding)
except ImportError as e:
    print(f'openai_harmony not found {e}')
    HarmonyEncodingName = None
    load_harmony_encoding = None
    Conversation = None
    Message = None
    Role = None
    SystemContent = None
    DeveloperContent = None

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger


class GPTOssVLLM(BaseModel):
    """GPT-OSS model wrapper using openai_harmony for conversation formatting.

    This model uses openai_harmony to properly format conversations for GPT-OSS
    models, following the demo pattern while conforming to OpenCompass interfaces.

    Args:
        path (str): The path to the GPT-OSS model.
        model_kwargs (dict): Keyword arguments for the model. Defaults to {}.
        generation_kwargs (dict): The generation kwargs for the model.
            Defaults to {}.
        max_seq_len (int): The maximum sequence length of the model.
            Defaults to 131072.
        meta_template (Dict, optional): The model's meta prompt template.
        stop_words (List[str]): List of stop words. Defaults to [].
        reasoning_level (Literal['high', 'medium', 'low']): The reasoning level for the model.
            Defaults to 'high'.
        entries_start_token (str): The start token for entries.
            Defaults to '##<channel>##'.
        entries_end_token (str): The end token for entries.
            Defaults to '##</channel>##'.
        tiktoken_cache_dir (str): Directory for tiktoken cache.
            Defaults to '/tmp/tiktoken_cache'.
    """

    def __init__(
        self,
        path: str,
        model_kwargs: dict = dict(),
        generation_kwargs: dict = dict(),
        max_seq_len: int = 65536,
        meta_template: Optional[Dict] = None,
        stop_words: List[str] = [],
        reasoning_level: Literal['high', 'medium', 'low'] = 'high',
        entries_start_token: str = '##<channel>##',
        entries_end_token: str = '##</channel>##',
        tiktoken_cache_dir: str = '~/tiktoken_cache',
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            generation_kwargs=generation_kwargs
        )

        self.logger = get_logger()

        # Check required dependencies
        assert LLM, ('Please install VLLM with `pip install vllm`.')
        assert load_harmony_encoding, (
            'Please install openai_harmony for GPT-OSS conversation formatting.')

        self.path = path
        self.stop_words = stop_words
        self.reasoning_level = reasoning_level

        if reasoning_level == 'high':
            self.system_message = (
                SystemContent.new()
                .with_reasoning_effort(ReasoningEffort.HIGH)
            )
        elif reasoning_level == 'medium':
            self.system_message = (
                SystemContent.new()
                .with_reasoning_effort(ReasoningEffort.MEDIUM)
            )
        elif reasoning_level == 'low':
            self.system_message = (
                SystemContent.new()
                .with_reasoning_effort(ReasoningEffort.LOW)
            )
        self.logger.info(f'Using reasoning level: {reasoning_level}')

        # Customize the output format
        self.entries_start_token = entries_start_token
        self.entries_end_token = entries_end_token

        # Set up tiktoken cache environment
        os.environ['TIKTOKEN_CACHE_DIR'] = tiktoken_cache_dir
        os.environ['TIKTOKEN_RS_CACHE_DIR'] = tiktoken_cache_dir

        # Initialize harmony encoding
        self.encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
        self.logger.info('Successfully loaded Harmony GPT-OSS encoding')

        # Get Harmony stop tokens
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self._load_model(path, model_kwargs)

        self.generation_kwargs = generation_kwargs

    def _load_model(self, path: str, added_model_kwargs: dict = dict()):
        """Load the VLLM model."""
        import ray

        if ray.is_initialized():
            self.logger.info(
                'shutdown ray instance to avoid "Calling ray.init() again" error.')
            ray.shutdown()

        DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        model_kwargs.update(added_model_kwargs)
        self.model = LLM(path, **model_kwargs)
        self.logger.info(f'Successfully loaded model from {path}')

    def _create_harmony_conversation(self, input_text: str) -> Conversation:
        """Create a Harmony conversation from input text."""
        return Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, self.system_message),
            Message.from_role_and_content(Role.USER, input_text),
        ])

    def generate(self, inputs: List[str], max_out_len: int, stopping_criteria: List[str] = [], **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.
            stopping_criteria (List[str]): Additional stop criteria.

        Returns:
            List[str]: A list of generated strings.
        """
        # Prepare batch of conversations and prefill token IDs

        # assert len(inputs) == 1, 'Only support batch size 1 for now'

        prefill_ids_batch = []
        for input_text in inputs:
            convo = self._create_harmony_conversation(input_text)
            prefill_ids = self.encoding.render_conversation_for_completion(
                convo, Role.ASSISTANT)
            prefill_ids_batch.append(prefill_ids)

        # Prepare generation parameters
        DEFAULT_GENERATION_KWARGS = {
            'temperature': 1.0,
            'top_p': 1.0,
            'max_tokens': max_out_len,
            'stop_token_ids': self.stop_token_ids,
            'skip_special_tokens': False,
        }

        sampling_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        sampling_kwargs.update(self.generation_kwargs)
        sampling_kwargs.update(kwargs)

        # Add any additional stop criteria (though harmony primarily uses token IDs)
        if stopping_criteria:
            stop_words = sampling_kwargs.get('stop', [])
            stop_words.extend(stopping_criteria + self.stop_words)
            sampling_kwargs['stop'] = list(set(stop_words))

        sampling_params = SamplingParams(**sampling_kwargs)

        self.logger.info(
            f'Generating for {len(inputs)} inputs with params: {sampling_params}')

        # Generate using VLLM with token IDs
        outputs = self.model.generate(
            prompt_token_ids=prefill_ids_batch,
            sampling_params=sampling_params,
        )

        # Process outputs and parse with Harmony
        output_strs = []
        for output in outputs:
            gen = output.outputs[0]
            raw_text = gen.text
            self.logger.info(f'--------------------------------')
            self.logger.info(f'text:\n {raw_text}')
            self.logger.info(f'--------------------------------')
            output_tokens = gen.token_ids  # Completion token IDs (no prefill)

            # Parse the completion token IDs back into structured Harmony messages
            try:
                entries = self.encoding.parse_messages_from_completion_tokens(
                    output_tokens, Role.ASSISTANT)
                # except Exception as e:
                # self.logger.error(f'{e.traceback}')
                # self.logger.error(f'raw_text[-200:]: {raw_text[-200:]}')
                # self.logger.error(f'--------------------------------')
                # output_tokens = output_tokens +[200006]
                # entries = self.encoding.parse_messages_from_completion_tokens(
                #     output_tokens, Role.ASSISTANT)

                text_content = ''
                for message in entries:
                    message_dict = message.to_dict()
                    entry_str = message_dict['content'][0]['text']
                    self.logger.info(f'entry_str: {entry_str}')
                    text_content += self.entries_start_token + entry_str + self.entries_end_token
                output_strs.append(text_content)

            except Exception as e:
                self.logger.error(f'{e.__traceback__}')
                self.logger.error(f'raw_text[-200:]: {raw_text[-200:]}')
                self.logger.error(f'--------------------------------')

                if '<|channel|>final<|message|>' in raw_text:
                    final_answer = raw_text.split(
                        '<|channel|>final<|message|>')[-1]
                elif '<|message|>' in raw_text:
                    self.logger.debug(
                        f'<|channel|>final<|message|> not in raw_text:\n{raw_text}')
                    final_answer = raw_text.split('<|message|>')[-1]
                    if '<|end|>' in final_answer:
                        final_answer = final_answer.split('<|end|>')[0]
                else:
                    self.logger.debug(
                        f'<|message|> not in raw_text:\n{raw_text}')
                    final_answer = raw_text

                output_strs.append(final_answer)
        return output_strs

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        try:
            convo = self._create_harmony_conversation(prompt)
            prefill_ids = self.encoding.render_conversation_for_completion(
                convo, Role.ASSISTANT)
            return len(prefill_ids)
        except Exception as e:
            self.logger.warning(
                f'Failed to get token length with Harmony: {e}')
            # Fallback: rough estimation
            return len(prompt.split()) * 2

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support '
                                  'ppl-based evaluation yet, try gen-based instead.')

    def get_ppl_tokenwise(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get tokenwise perplexity scores given a list of inputs."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support '
                                  'ppl-based evaluation yet, try gen-based instead.')

    def encode(self, prompt: str):
        """Encode prompt to tokens."""
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement encode method.')

    def decode(self, tokens):
        """Decode tokens to text."""
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement decode method.')
