from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger


class MiniCPMOModel(BaseModel):

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        model_kwargs: dict = None,
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        mode: str = 'none',
        stop_words: List[str] = [],
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)

        self.logger = get_logger()
        self._load_model(path, model_kwargs)
        self.generation_kwargs = generation_kwargs
        self.mode = mode
        self.stop_words = stop_words

    def _load_model(self, path: str, add_model_kwargs: dict = None):
        """加载模型和tokenizer."""
        DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True,
                                    attn_implementation='sdpa',
                                    torch_dtype=torch.bfloat16,
                                    init_vision=False,
                                    init_audio=False,
                                    init_tts=False)
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        if add_model_kwargs is not None:
            model_kwargs.update(add_model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, **model_kwargs)
        self.model = self.model.eval().cuda()

    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        """生成回复."""
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        generation_kwargs.update({'max_new_tokens': max_out_len})

        if 'temperature' not in generation_kwargs:
            generation_kwargs['temperature'] = 0.5

        output_texts = []
        for input_text in inputs:
            # 构建系统提示词
            # sys_prompt = ('You are a helpful assistant. You can accept '
            #               'audio and text input and output voice and text.')
            # sys_msg = {"role": "user", "content": [sys_prompt]}

            # 构建用户输入消息
            msgs = [{'role': 'user', 'content': input_text}]

            # 生成回复
            res = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                # omni_input=True,
                omni_input=False,
                use_tts_template=False,
                generate_audio=False,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
                **generation_kwargs)

            output_texts.append(res.text)

        return output_texts
