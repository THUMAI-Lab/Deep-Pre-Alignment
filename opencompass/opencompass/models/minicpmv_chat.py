from typing import Dict, List, Optional

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True)

import os

import torch
from transformers import AutoModel, AutoTokenizer


class MinicpmVChat(BaseModel):

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
        # self.generation_kwargs.pop('do_sample', None)

        # assert mode in ['none', 'mid']
        self.mode = mode
        self.use_fastchat_template = use_fastchat_template
        self.stop_words = stop_words

    def _load_model(self,
                    path: str,
                    add_model_kwargs: dict = None,
                    num_retry: int = 3):
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        if add_model_kwargs is not None:
            model_kwargs.update(add_model_kwargs)
        self.model = AutoModel.from_pretrained(path,
                                               trust_remote_code=True,
                                               torch_dtype=torch.float16)
        self.model = self.model.to(device='cuda')
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)

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
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        generation_kwargs.update({'max_tokens': max_out_len})
        _stop = list(set(self.stop_words + stopping_criteria))
        generation_kwargs.update({'stop': _stop})

        prompt_list, output_strs = [], []

        for input_id in inputs:
            # print(input_id)
            msgs = [{'role': 'user', 'content': f'{input_id}'}]
            answer = self.model.chat(
                image=None,
                msgs=msgs,
                context=None,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                **generation_kwargs,
            )
            # print(answer)
            output_strs.append(answer[0])
        return output_strs
