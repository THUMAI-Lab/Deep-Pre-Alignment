import os
from typing import Dict, List, Optional

import torch
from ola.conversation import SeparatorStyle, conv_templates
from ola.mm_utils import KeywordsStoppingCriteria
from ola.model.language_model.ola_qwen import OlaQwenForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

# DEFAULT_MODEL_KWARGS = dict(trust_remote_code=True)
DEFAULT_MODEL_KWARGS = dict()

os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = '0x64'
os.environ['VIDEO_MAXRES'] = '480'
os.environ['VIDEO_MINRES'] = '288'
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

# Model Constants
IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = '<speech>'
IMAGE_TOKEN_INDEX = -300
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


def tokenizer_image_token(prompt,
                          tokenizer,
                          image_token_index=IMAGE_TOKEN_INDEX,
                          return_tensors=None):
    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split('<image>')
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X))
                for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[
            0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks,
                              [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def load_pretrained_model(model_path,
                          model_base,
                          is_lora=False,
                          s2s=False,
                          load_8bit=False,
                          load_4bit=False,
                          device='cuda',
                          use_flash_attn=False,
                          **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    model_cls = OlaQwenForCausalLM

    # Load Ola model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = model_cls.from_pretrained(model_path,
                                      low_cpu_mem_usage=False,
                                      **kwargs)
    model = model.to(device=device)

    image_processor = None
    model.resize_token_embeddings(len(tokenizer))
    print('Loading vision tower succeeded.')

    if hasattr(model.config, 'max_sequence_length'):
        context_len = model.config.max_sequence_length
    else:
        context_len = 16384

    return tokenizer, model, image_processor, context_len


class OlaModel(BaseModel):

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
        self.generation_kwargs = generation_kwargs

        self.mode = mode
        self.use_fastchat_template = use_fastchat_template
        self.stop_words = stop_words
        self.pad_token_ids = 151643

    def _load_model(self, path: str, add_model_kwargs: dict = None):
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        if add_model_kwargs is not None:
            model_kwargs.update(add_model_kwargs)

        self.tokenizer, self.model, self.image_processor, _ = \
            load_pretrained_model(path, None)
        self.model = self.model.to('cuda').eval()
        self.model = self.model.bfloat16()

    def _process_inputs(self, text):
        # 初始化多模态相关变量
        self.images = [
            torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16,
                                           device='cuda',
                                           non_blocking=True)
        ]
        self.images_highres = [
            torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16,
                                           device='cuda',
                                           non_blocking=True)
        ]
        self.image_sizes = [(224, 224)]

        self.speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
        self.speech_lengths = [torch.LongTensor([3000]).to('cuda')]
        self.speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
        self.speech_chunks = [torch.LongTensor([1]).to('cuda')]

        return text

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

        output_strs = []
        for input_text in inputs:
            # 处理输入
            text = self._process_inputs(input_text)

            # 准备对话模板
            conv_mode = 'qwen_1_5'
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # self.logger.info(f"prompt: {prompt}")

            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                return_tensors='pt').unsqueeze(0).to('cuda')

            # 准备生成参数
            attention_masks = input_ids.ne(
                self.pad_token_ids).long().to('cuda')
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2  # noqa: E501
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, self.tokenizer, input_ids)

            if 'temperature' not in generation_kwargs:
                generation_kwargs['temperature'] = 0.2
            if 'top_p' not in generation_kwargs:
                generation_kwargs['top_p'] = None
            if 'num_beams' not in generation_kwargs:
                generation_kwargs['num_beams'] = 1

            # 生成输出
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=self.images,
                    images_highres=self.images_highres,
                    image_sizes=self.image_sizes,
                    modalities=['text'],
                    speech=self.speechs,
                    speech_lengths=self.speech_lengths,
                    speech_chunks=self.speech_chunks,
                    speech_wav=self.speech_wavs,
                    attention_mask=attention_masks,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True
                    if generation_kwargs['temperature'] > 0 else False,
                    **generation_kwargs)

            # 处理输出
            outputs = self.tokenizer.batch_decode(output_ids,
                                                  skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            output_strs.append(outputs)

        return output_strs
