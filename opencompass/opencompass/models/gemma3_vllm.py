import os
from typing import Dict, List, Optional, Union

import torch
from transformers import (AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM,
                          pipeline)

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from .huggingface_above_v4_33 import (_convert_chat_messages,
                                      _format_with_fast_chat_template,
                                      _get_meta_template,
                                      _get_possible_max_seq_len)

PromptType = Union[PromptList, str]


class Gemma3(BaseModel):

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 model_kwargs: dict = None,
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 stop_words: List[str] = []):
        self.logger = get_logger()
        if os.environ.get('LOCAL_PATH', ''):
            path = os.environ.get('LOCAL_PATH', path)
        self.path = path
        self.template_parser = _get_meta_template(meta_template)
        self.logger.info(f'### LOAD MODEL FROM: ### {path}')

        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = Gemma3ForCausalLM.from_pretrained(
            path, quantization_config=quantization_config).eval()

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        prompt_tokens = []
        results = []
        for input in inputs:
            # input_ids = get_input_ids_TT(input, self.text_tokenizer)
            text_output = self._generate(input)
            # print(f"Input: {input}")
            # print(f"Output: {text_output}")
            results.append(text_output)

        return results

    def _generate(self, input):
        msgs = [{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': input
                },
            ]
        }]
        text = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
        ).to(self.model.device).to(torch.bfloat16)

        print('*' * 100)
        print(text)
        print('*' * 100)

        # 显式设置 eos_token_id（结束符的token ID）
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<end_of_turn>')

        with torch.inference_mode():
            outputs = self.model.generate(
                **text,
                max_new_tokens=1024,
                early_stopping=True,
                # eos_token_id=tokenizer.eos_token_id +  eos_token_id,
                eos_token_id=[self.tokenizer.eos_token_id, eos_token_id]
                # eos_token_id=eos_token_id
            )

        outputs = self.tokenizer.batch_decode(outputs)
        generated_text = outputs[0].split('<start_of_turn>model\n')[1].split(
            '<end_of_turn>')[0]
        print('generated_text', generated_text)
        return generated_text
