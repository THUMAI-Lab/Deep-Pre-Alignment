from typing import Dict, List, Optional

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

# DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True)
DEFAULT_MODEL_KWARGS = dict()


class Llama4withChatTemplate(BaseModel):

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
        self.use_fastchat_template = use_fastchat_template
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

        self.processor = AutoProcessor.from_pretrained(path)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map='balanced',
            **model_kwargs)

        self.model = self.model.eval()

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
        generation_kwargs.update({'max_new_tokens': max_out_len})
        # self.logger.info(f'Generation kwargs: {generation_kwargs}')

        prompt_list, output_strs = [], []

        for input_id in inputs:
            # print(input_id)

            # self.logger.info(f'Input: {input_id}')
            if self.use_fastchat_template:

                msgs = [
                    {
                        'role': 'user',
                        'content': [{
                            'type': 'text',
                            'text': f'{input_id}'
                        }]
                    },
                ]
                self.logger.info(f'Msgs: {msgs}')

                inputs = self.processor.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors='pt',
                ).to(self.model.device)
            else:
                self.logger.info(f'Input: {input_id}')
                inputs = self.processor(
                    text=input_id,
                    return_tensors='pt',
                ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
                # max_new_tokens=max_out_len,
            )

            response = self.processor.batch_decode(
                outputs[:, inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True)[0]
            self.logger.info(f'Response: {response}')
            output_strs.append(response)
        return output_strs


class Llama4BaseModel(Llama4withChatTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fastchat_template = False
        self.stop_words = []
