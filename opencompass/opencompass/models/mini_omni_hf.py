from typing import Dict, List, Optional, Union

import torch

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

import os
import sys
import time

import lightning as L
import torch
from snac import SNAC

# sys.path.append("/data/lby/opencompass/mini-omni")

# 将本地路径添加到 sys.path 的开头
litgpt_path = '/data/lby/opencompass/mini-omni'  # 替换为实际路径
sys.path.insert(0, litgpt_path)

import soundfile as sf
# sys.path.append("/data/lby/opencompass/mini-omni/litgpt/utils")
# from snac_utils import layershift, reconscruct_snac, reconstruct_tensors, get_time_str
# from utils.snac_utils import get_snac, generate_audio_data
import whisper
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from litgpt import Tokenizer
# from litgpt.utils import (
#     num_parameters,
# )
from litgpt.generate.base import \
    generate_TT  # generate_AA,; generate_ASR,; generate_TA,; generate_AT,; generate_TA_BATCH,; next_token_batch
from litgpt.model import GPT, Config
from utils.snac_utils import (get_time_str, layershift, reconscruct_snac,
                              reconstruct_tensors)

# sys.path.append("/data/lby/opencompass/mini-omni/litgpt/generate")
# from base import generate_TT

# from tqdm import tqdm
# from huggingface_hub import snapshot_download

# TODO
text_vocabsize = 151936
text_specialtokens = 64
audio_vocabsize = 4096
audio_specialtokens = 64

padded_text_vocabsize = text_vocabsize + text_specialtokens
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens

_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_asr = text_vocabsize + 4

_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_split = audio_vocabsize + 4


def get_input_ids_TT(text, text_tokenizer):
    input_ids_item = [[] for i in range(8)]
    text_tokens = text_tokenizer.encode(text).tolist()

    for i in range(7):
        input_ids_item[i] = torch.tensor([layershift(_pad_a, i)] *
                                         (len(text_tokens) + 3)).unsqueeze(0)
    input_ids_item[-1] = [_input_t] + text_tokens + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)

    return input_ids_item


class Omni(BaseModel):

    def __init__(
        self,
        path: str,
        meta_template: Optional[Dict] = None,
        num_gpus: int = 2,
    ):  # noqa
        # if tokenizer_only:
        #     self._load_tokenizer(tokenizer_path=tokenizer_path)
        # else:
        self.fabric, self.model, self.text_tokenizer, self.snacmodel, self.whispermodel = self._load_model(
            path=path)
        # self.max_seq_len = max_seq_len
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()

    def _load_model(self, path: str, device='cuda:0'):
        snacmodel = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval().to(
            device)
        whispermodel = whisper.load_model('small').to(device)
        text_tokenizer = Tokenizer(path)
        fabric = L.Fabric(devices=1, strategy='auto')
        config = Config.from_file(path + '/model_config.yaml')
        config.post_adapter = False

        with fabric.init_module(empty_init=False):
            model = GPT(config)

        model = fabric.setup(model)
        state_dict = lazy_load(path + '/lit_model.pth')
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()

        return fabric, model, text_tokenizer, snacmodel, whispermodel

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        prompt_tokens = []
        results = []
        with torch.no_grad():
            step = 0

            for input in inputs:
                input_ids = get_input_ids_TT(input, self.text_tokenizer)
                text_output = self.T1_T2(self.fabric, input_ids, self.model,
                                         self.text_tokenizer, step)
                # print(f"Input: {input}")
                # print(f"Output: {text_output}")
                results.append(text_output)

        return results

    def get_token_len(self, prompt: str) -> int:
        return len(self.text_tokenizer.encode(prompt, True, True))

    def T1_T2(self, fabric, input_ids, model, text_tokenizer, step):
        with fabric.init_tensor():
            model.set_kv_cache(batch_size=1)
        tokenlist = generate_TT(
            model,
            None,
            input_ids,
            None,
            ['T1T2'],
            max_returned_tokens=2048,
            temperature=0.9,
            top_k=1,
            eos_id_a=_eoa,
            eos_id_t=_eot,
            pad_id_t=_pad_t,
            shift=padded_text_vocabsize,
            include_prompt=True,
            generate_text=True,
        )
        model.clear_kv_cache()
        return text_tokenizer.decode(torch.tensor(tokenlist)).strip()
