# flake8: noqa
# yapf: disable
import base64
from io import BytesIO
from typing import Dict, List, Optional, Union

import numpy as np
import requests
import torch
from PIL import Image

from opencompass.models.base import BaseModel
from opencompass.models.huggingface_above_v4_33 import (
    _convert_chat_messages, _format_with_fast_chat_template,
    _get_meta_template, _get_possible_max_seq_len)
from opencompass.utils import get_logger

try:
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
except ImportError:
    AutoProcessor, Gemma3nForConditionalGeneration = None, None


class Gemma3n(BaseModel):

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
        device_map: str = 'auto',
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        assert AutoProcessor and Gemma3nForConditionalGeneration, (
            'Please install transformers with Gemma3n support: `pip install transformers>=4.53.0`'
        )

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        if tokenizer_only:
            self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            self.model = None
        else:
            self._load_model(path, model_kwargs)

        self.generation_kwargs = generation_kwargs
        self.generation_kwargs.pop('do_sample', None)
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))

    def _load_model(self, path: str, added_model_kwargs: dict = dict()):
        DEFAULT_MODEL_KWARGS = dict(
            trust_remote_code=True,
            device_map=self.device_map,
            # device_map='cuda',
            torch_dtype=self.torch_dtype,
        )
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        model_kwargs.update(added_model_kwargs)

        self.model = Gemma3nForConditionalGeneration.from_pretrained(path, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.processor.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.processor.tokenizer.decode(token_id))
        if self.processor.tokenizer.eos_token is not None:
            potential_stop_words.append(self.processor.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def _load_image(self, image_path_or_url: str) -> Image.Image:
        """Load image from path or URL."""
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
        elif image_path_or_url.startswith('data:image'):
            # Handle base64 encoded images
            image_data = image_path_or_url.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        else:
            image = Image.open(image_path_or_url)
        return image.convert('RGB')

    def _parse_multimodal_input(self, inputs: Union[str, List[Dict]]) -> List[Dict]:
        """Parse input to multimodal messages format."""
        if isinstance(inputs, str):
            # Simple text input
            return [
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': inputs}]
                }
            ]
        elif isinstance(inputs, list):
            # Already in message format
            return inputs
        else:
            raise ValueError(f'Unsupported input type: {type(inputs)}')

    def generate(self, inputs: Union[str, List[str], List[Dict]], max_out_len: int, stopping_criteria: List[str] = [], **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs: Input can be:
                - str: Simple text input
                - List[str]: List of text inputs
                - List[Dict]: List of multimodal messages in chat format
            max_out_len (int): The maximum length of the output.
            stopping_criteria (List[str]): Additional stopping criteria.

        Returns:
            List[str]: A list of generated strings.
        """

        self.logger.info(f'inputs: {inputs}')
        inputs = inputs[0][-1]['prompt']
        self.logger.info(f'inputs: {inputs}')

        # [{'role': 'user', 'content': [{'type': 'text', 'text': 'Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n\n'}]}]

        messages = [
            # {
            #     "role": "system",
            #     "content": [{"type": "text", "text": "You are a helpful assistant."}]
            # },
            {
                'role': 'user',
                'content': [
                    # {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                    # {"type": "text", "text": "Describe this image in detail."}
                    {'type': 'text', 'text': inputs}
                ]
            }
        ]
        self.logger.info(f'messages: {messages}')

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self.model.device)

        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                # max_new_tokens=max_out_len,
                max_new_tokens=1024,
                do_sample=False
                )
            # self.logger.info(f'generation: {generation}')
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        self.logger.info(f'decoded: {decoded}')
        return [decoded]
        # if isinstance(inputs, str):
        #     inputs = [inputs]

        # if isinstance(inputs[0], str):
        #     # Convert text inputs to multimodal format
        #     messages_list = [self._parse_multimodal_input(inp) for inp in inputs]
        # else:
        #     # Assume already in multimodal format
        #     messages_list = inputs if isinstance(inputs[0], list) else [inputs]

        # DEFAULT_GENERATION_KWARGS = {
        #     'max_new_tokens': max_out_len,
        #     'do_sample': False,
        #     'temperature': 0.0,
        # }
        # generation_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        # generation_kwargs.update(self.generation_kwargs)
        # generation_kwargs.update(kwargs)

        # output_strs = []

        # for messages in messages_list:
        #     try:
        #         # Process images in messages
        #         processed_messages = []
        #         for message in messages:
        #             processed_content = []
        #             for content in message.get('content', []):
        #                 if content['type'] == 'image':
        #                     if isinstance(content['image'], str):
        #                         # Load image from path/URL
        #                         image = self._load_image(content['image'])
        #                         processed_content.append({"type": "image", "image": image})
        #                     else:
        #                         processed_content.append(content)
        #                 else:
        #                     processed_content.append(content)
        #             processed_messages.append({
        #                 "role": message['role'],
        #                 "content": processed_content
        #             })

        #         # Apply chat template and tokenize
        #         model_inputs = self.processor.apply_chat_template(
        #             processed_messages,
        #             add_generation_prompt=True,
        #             tokenize=True,
        #             return_dict=True,
        #             return_tensors="pt",
        #         ).to(self.model.device)

        #         input_len = model_inputs["input_ids"].shape[-1]

        #         # Generate
        #         with torch.inference_mode():
        #             generation = self.model.generate(**model_inputs, **generation_kwargs)
        #             generation = generation[0][input_len:]

        #         # Decode
        #         decoded = self.processor.decode(generation, skip_special_tokens=True)
        #         output_strs.append(decoded)

        #         self.logger.info(f'Generated text: {decoded}')

        #     except Exception as e:
        #         self.logger.error(f'Error generating for input: {e}')
        #         output_strs.append("")

        # return output_strs

    def get_token_len(self, prompt: Union[str, List[Dict]]) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt: Input string or multimodal message.

        Returns:
            int: Length of the input tokens
        """
        if isinstance(prompt, str):
            messages = self._parse_multimodal_input(prompt)
        else:
            messages = prompt

        try:
            model_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                tokenize=True,
                return_tensors='pt'
            )
            return model_inputs['input_ids'].shape[-1]
        except Exception as e:
            self.logger.error(f'Error getting token length: {e}')
            return 0


class Gemma3nBase(Gemma3n):

    def generate(self, inputs: Union[str, List[str], List[Dict]], max_out_len: int, stopping_criteria: List[str] = [], **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs: Input can be:
                - str: Simple text input
                - List[str]: List of text inputs
                - List[Dict]: List of multimodal messages in chat format
            max_out_len (int): The maximum length of the output.
            stopping_criteria (List[str]): Additional stopping criteria.

        Returns:
            List[str]: A list of generated strings.
        """

        self.logger.info(f'inputs: {inputs}')
        inputs = inputs[-1]
        self.logger.info(f'inputs: {inputs}')

        inputs = self.processor(text=inputs, return_tensors='pt').to(self.model.device)

        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                # max_new_tokens=max_out_len,
                max_new_tokens=5,
                do_sample=False
                )
            # self.logger.info(f'generation: {generation}')
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        self.logger.info(f'decoded: {decoded}')
        return [decoded]


    def get_token_len(self, prompt: Union[str, List[Dict]]) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt: Input string or multimodal message.

        Returns:
            int: Length of the input tokens
        """
        if isinstance(prompt, str):
            messages = self._parse_multimodal_input(prompt)
        else:
            messages = prompt

        try:
            inputs = self.processor(text=prompt, return_tensors='pt').to(self.model.device)
            return inputs['input_ids'].shape[-1]
        except Exception as e:
            self.logger.error(f'Error getting token length: {e}')
            return 0
