########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import datetime
import json
import os
import random
import re
import sys

import numpy as np
import torch
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset, load_from_disk
from torch.nn import functional as F

from opencompass.models.base import BaseModel
from opencompass.utils import get_logger

# for HF_MODE, do these first:
# pip install flash_attn -U --force-reinstall
# pip install git+https://github.com/fla-org/flash-linear-attention -U --force-reinstall
#
# HF_MODE = True # currently HF_MODE only has 30% running speed. wait for optimizations

# HF_MODE = False # you will get 44.87% for RWKV-x070-World-1.5B-v3-20250127-ctx4096

########################################################################################################


class RWKV_World(BaseModel):

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        model_kwargs: dict = None,
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        mode: str = 'none',
        use_fastchat_template: bool = False,
        lora_path: str = None,
        stop_words: List[str] = [],
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        self.path = path

        os.environ['RWKV_V7_ON'] = '1'  # enable this for rwkv-7 models
        os.environ['RWKV_JIT_ON'] = '1'  #### set these before import RWKV
        os.environ[
            'RWKV_CUDA_ON'] = '1'  #### set to '1' to compile CUDA kernel (10x faster) - requires c++ compiler & cuda libraries

        from rwkv.model import RWKV  # ### pip install rwkv --upgrade
        from rwkv.utils import PIPELINE, PIPELINE_ARGS

        # if not HF_MODE:
        # download from https://huggingface.co/BlinkDL/rwkv-7-world
        # MODEL_NAME = "/mnt/e/RWKV-Runner/models/RWKV-x070-World-1.5B-v3-20250127-ctx4096"
        print(f'Loading model - {self.path}')

        self.model = RWKV(model=self.path, strategy='cuda fp16')
        self.pipeline = PIPELINE(self.model, 'rwkv_vocab_v20230424')
        self.tokenizer = self.pipeline.tokenizer
        # else:
        # MODEL_NAME = 'fla-hub/rwkv7-1.5B-world'
        # print(f"Loading model - {MODEL_NAME}")

        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda:0", trust_remote_code=True).eval()
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

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

        prompt_list, output_strs = [], []
        for input in inputs:
            chat_rounds = [
                # "User: hi",
                # "Assistant: Hi. I am your assistant and I will provide expert full response in full details.",
                'User: '
                # "User: You are a very talented expert. Answer this question:\n"
                # "Question: "
                + re.sub(r'\n{2,}', '\n', input).strip().replace(
                    '\r\n', '\n'),  #### replace all \n\n and \r\n by \n
                # "Assistant: The answer is",
                'Assistant: The answer is',
                # "Answer:",
            ]  #### dont add space after this final ":"

            print('*' * 100)
            print('\n\n'.join(chat_rounds[-2:]), end='')
            print('*' * 100)

            # print("input", input)

            # my_qa_generator("\n\n".join(chat_rounds))
            output_strs.append(self._generate('\n\n'.join(chat_rounds)))
            print('\n' + '=' * 80)
        return output_strs

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

        # generation_kwargs = kwargs.copy()
        # generation_kwargs.update(self.generation_kwargs)
        # generation_kwargs.update({'max_tokens': max_out_len})
        # _stop = list(set(self.stop_words + stopping_criteria))
        # generation_kwargs.update({'stop': _stop})
        # sampling_kwargs = SamplingParams(**generation_kwargs)
        # print("*" * 100)
        # print(inputs)
        # print("*" * 100)

        # outputs = self.model.generate(inputs, sampling_kwargs)

        # prompt_list, output_strs = [], []
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     prompt_list.append(prompt)
        #     output_strs.append(generated_text)

        # return output_strs
    def get_token_len(self,
                      prompt: str,
                      add_special_tokens: bool = True) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        # tokenizer = self.model.get_tokenizer()
        # token_ids = tokenizer.encode(prompt,
        #                              add_special_tokens=add_special_tokens)
        tokens = self.pipeline.encode(prompt)
        return len(tokens)

    def _generate(self, ctx):
        out_tokens = []
        out_len = 0
        out_str = ''
        occurrence = {}
        state = None
        for i in range(999):
            if i == 0:
                out, state = self.pipeline.model.forward(
                    self.pipeline.encode(ctx), state)
            else:
                out, state = self.pipeline.model.forward([token], state)

            for n in occurrence:
                out[n] -= (
                    0.4 + occurrence[n] * 0.4
                )  #### higher repetition penalty because of lower top_p here

            token = self.pipeline.sample_logits(
                out, temperature=1.0, top_p=0.2)  #### sample the next token

            if token == 0:
                break  #### exit at token [0] = <|endoftext|>

            out_tokens += [token]

            for n in occurrence:
                occurrence[n] *= 0.996  #### decay repetition penalty
            occurrence[token] = 1 + (occurrence[token]
                                     if token in occurrence else 0)

            tmp = self.pipeline.decode(out_tokens[out_len:])
            if ('\ufffd' not in tmp) and (
                    not tmp.endswith('\n')
            ):  #### print() only when out_str is valid utf-8 and not end with \n
                out_str += tmp
                print(tmp, end='', flush=True)
                out_len = i + 1
            elif '\n\n' in tmp:  #### exit at '\n\n'
                tmp = tmp.rstrip()
                out_str += tmp
                print(tmp, end='', flush=True)
                break
        return out_str.strip()
