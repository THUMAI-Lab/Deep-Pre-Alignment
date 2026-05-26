# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional

import numpy as np
import torch

from opencompass.models.base import BaseModel
from opencompass.models.huggingface_above_v4_33 import (
    _convert_chat_messages, _format_with_fast_chat_template,
    _get_meta_template, _get_possible_max_seq_len)
from opencompass.utils import get_logger

try:
    from vllm import SamplingParams

    from cpmcu.common.display import display
    from cpmcu.common.utils import (apply_minicpm4_yarn_config, create_model,
                                    setup_frspec_vocab, setup_model_paths)
except ImportError:
    create_model, SamplingParams = None, None


class CPMCUwithChatTemplate(BaseModel):
    def __init__(
        self,
        path: str,
        model_kwargs: dict = dict(),
        tokenizer_only: bool = False,
        generation_kwargs: dict = dict(),
        max_seq_len: int = None,
        meta_template: Optional[Dict] = None,
        fastchat_template: Optional[str] = None,
        enable_thinking: bool = True,
        stop_words: List[str] = [],
    ):
        assert create_model, (
            'Please install CPMCU with `https://github.com/OpenBMB/CPM.cu.git`')
        self.logger = get_logger()
        self.path = path
        self.enable_thinking = enable_thinking
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        if tokenizer_only:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True)
        else:
            self._load_model(path, model_kwargs, generation_kwargs)
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True)
        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)
        self.fastchat_template = fastchat_template
        self.stop_words = list(
            set(stop_words + self._get_potential_stop_words(path)))
        self.logger.warning(
            'ignore stop_words, just use `self.tokenizer.eos_token_id` as terminators')
        self.terminators = [self.tokenizer.eos_token_id]

    def _load_model(self, path: str, added_model_kwargs: dict = dict(), added_generation_kwargs: dict = dict()):
        import ray
        if ray.is_initialized():
            self.logger.info(
                'shutdown ray instance to avoid "Calling ray.init() again" error.')
            ray.shutdown()
        DEFAULT_MODEL_KWARGS = {
            # Model Configuration
            'model_path': None,  # required=True, 需要手动设置
            'draft_model_path': None,
            'frspec_path': None,
            'model_type': 'minicpm4',
            'dtype': 'bfloat16',
            'minicpm4_yarn': False,

            # System Configuration
            'cuda_graph': True,
            'memory_limit': 0.9,
            'chunk_length': 2048,
            'plain_output': False,

            # Speculative Decoding
            'spec_window_size': 1024,
            'spec_num_iter': 2,
            'spec_topk_per_iter': 10,
            'spec_tree_size': 12,
            'frspec_vocab_size': 32768,

            # Sparse Attention
            'sink_window_size': 1,
            'block_window_size': 32,
            'sparse_topk_k': 64,
            'sparse_switch': 8192,
            'use_compress_lse': True,

            # Server Configuration (from server parser)
            'host': '0.0.0.0',
            'port': 8000,

            # Prompt Configuration (from CLI parser)
            'prompt_file': None,
            'prompt_text': None,
            'use_chat_template': True,

            # Generation Configuration (from CLI parser)
            'use_stream': True,
            'num_generate': 65536,
            'temperature': 0.0,
            'top_p': 1.0,
            'random_seed': None,
            'ignore_eos': False,
        }
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        model_kwargs.update(added_model_kwargs)
        model_kwargs.update(added_generation_kwargs)
        model_kwargs['model_path'] = path
        self.logger.info('CPM.cu Config:\n' + str(model_kwargs)+'\n')
        model_path, draft_model_path, frspec_path = setup_model_paths(
            model_kwargs)
        # LLM(path, **model_kwargs)
        self.model = create_model(model_path, draft_model_path, model_kwargs)
        self.model.init_storage()
        if getattr(model_kwargs, 'minicpm4_yarn', False):
            try:
                apply_minicpm4_yarn_config(self.model)
            except Exception as e:
                self.logger.warning(f'MiniCPM4 YARN configuration failed: {e}')
        # Load frequency speculative vocabulary if enabled (draft model exists)
        self.has_speculative = getattr(
            model_kwargs, 'draft_model_path', None) is not None
        if self.has_speculative and (frspec_path is not None) and (getattr(model_kwargs, 'frspec_vocab_size', 0) > 0):
            frspec_result = setup_frspec_vocab(
                self.model, frspec_path, getattr(model_kwargs, 'frspec_vocab_size', 0))
            if frspec_result is True:
                self.logger.info('Loaded frequency speculative vocabulary')
            else:
                self.logger.warning(
                    'Could not load frequency speculative vocabulary')

        # Load model weights
        self.logger.info('Loading model weights...')
        self.model.load_from_hf()
        self.logger.info('Model loading completed!')

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(
                    self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(
                        self.tokenizer.decode(token_id))
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
            messages = _format_with_fast_chat_template(
                messages, self.fastchat_template)
        else:
            if self.enable_thinking:
                self.logger.info('[启用 CPMCU 深思模式] enable_thinking=True')
                messages = [self.tokenizer.apply_chat_template(
                    m, add_generation_prompt=True, tokenize=False, enable_thinking=True) for m in messages]
            else:
                self.logger.info('[启用 CPMCU 非深思模式] enable_thinking=False')
                messages = [self.tokenizer.apply_chat_template(
                    m, add_generation_prompt=True, tokenize=False, enable_thinking=False) for m in messages]
            # vLLM tokenize prompts by AutoTokenizer with its default parameter "add_special_token=True"
            # OC add bos_token in the prompt, which requires tokenizing prompts using "add_speicial_token=False"
            # But vLLM doesn't have "add_speicial_token" in the pipeline API. So, we remove bos_token
            # from messages as a workaround
            if self.tokenizer.bos_token:
                bos_token = self.tokenizer.bos_token
                messages = [message.removeprefix(bos_token) if message.startswith(
                    bos_token) else message for message in messages]
        # DEFAULT_GENERATION_KWARGS = {
        #     'temperature': 0,
        #     'max_tokens': max_out_len,
        #     'stop': list(set(self.stop_words + stopping_criteria))
        # }
        # sampling_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        # sampling_kwargs.update(self.generation_kwargs)
        # sampling_kwargs.update(kwargs)
        # sampling_kwargs = SamplingParams(**sampling_kwargs)
        # self.logger.info(sampling_kwargs)
        self.logger.info(f'messages: {messages}')
        # outputs = self.model.generate(messages, sampling_kwargs)
        # prompt_list, output_strs = [], []
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     prompt_list.append(prompt)
        #     output_strs.append(generated_text)
        #     self.logger.info(f'generated_text: {generated_text}')
        output_strs = []
        for prompt in messages:
            input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(
                'cuda', dtype=torch.int32)  # ! 要检查一下有没有bos ->加了
            results = self.model.generate(
                input_ids=input_ids.view(-1),
                generation_length=max_out_len,
                teminators=self.terminators,
                use_stream=False,
                progress_callback=None
            )
            # Extract tokens and statistics from results
            input_length = len(input_ids.view(-1))

            if self.has_speculative:
                tokens, accept_lengths, decode_time, prefill_time = results
            else:
                tokens, decode_time, prefill_time = results
                accept_lengths = None

            # Decode tokens and handle edge cases
            generated_text = self.tokenizer.decode(
                tokens, skip_special_tokens=True) or ''
            output_strs.append(generated_text)
            self.logger.info(f'generated_text: {generated_text}')

        return output_strs

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.
        Returns:
            int: Length of the input tokens
        """
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(
            m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])
