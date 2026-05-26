from io import BytesIO
from typing import Dict, List, Optional
from urllib.request import urlopen

import numpy as np
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger


class Qwen2_5OmniModelHF(BaseModel):

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        model_kwargs: dict = None,
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        mode: str = 'none',
        use_fastchat_template: bool = False,
        stop_words: List[str] = [],
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)

        self.logger = get_logger()
        self._load_model(path, model_kwargs)
        # self.tokenizer = self.model.get_tokenizer()
        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)
        self.stop_words = stop_words

    def _load_model(self,
                    path: str,
                    add_model_kwargs: dict = None,
                    num_retry: int = 3):
        self.logger.info(f'Loading model from {path}')
        self.model = Qwen2_5OmniModel.from_pretrained(
            path,
            torch_dtype='auto',
            device_map='auto',
            attn_implementation='flash_attention_2',
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(path)

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """

        # if self.mode == 'mid':
        #     input_ids = self.tokenizer(inputs, truncation=False)['input_ids']
        #     inputs = []
        #     for input_id in input_ids:
        #         if len(input_id) > self.max_seq_len - max_out_len:
        #             half = int((self.max_seq_len - max_out_len) / 2)
        #             inputs.append(
        #                 self.tokenizer.decode(input_id[:half],
        #                                       skip_special_tokens=True) +
        #                 self.tokenizer.decode(input_id[-half:],
        #                                       skip_special_tokens=True))
        #         else:
        #             inputs.append(
        #                 self.tokenizer.decode(input_id,
        #                                       skip_special_tokens=True))

        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        generation_kwargs.update({'max_new_tokens': max_out_len})
        # _stop = list(set(self.stop_words + stopping_criteria))
        # generation_kwargs.update({'stop': _stop})
        # print(generation_kwargs)
        # generation_kwargs["max_new_tokens"] = generation_kwargs["max_length"]
        # del generation_kwargs["max_length"]

        assert isinstance(inputs, list) and len(inputs) == 1
        conversations = []
        for input_id in inputs:

            conversation = [
                # {"role": "system", "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."},
                {
                    'role': 'user',
                    'content': input_id
                },
            ]
            conversations.append(conversation)

        text = self.processor.apply_chat_template(conversation,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        self.logger.info(f'text: {text}')
        inputs_ = self.processor(text=text, return_tensors='pt', padding=True)
        # inputs_['input_ids'] = inputs_['input_ids'].to("cuda")
        # inputs_.input_ids = inputs_.input_ids.to("cuda")
        inputs_ = inputs_.to('cuda')
        # inputs_ = inputs_.to(self.model.device).to(self.model.dtype)
        # self.logger.info(f"inputs_: {inputs_}")

        # print(inputs_.input_ids.device)
        # print(self.model.device)

        generate_ids = self.model.generate(**inputs_,
                                           use_audio_in_video=False,
                                           **generation_kwargs,
                                           return_audio=False)
        # text = processor.batch_decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        generate_ids = generate_ids[:, inputs_.input_ids.size(1):]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        return response
