import json
import os
import re
import sys
import traceback
from typing import Dict, List, Optional, Union

import torch
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.speech2unit.speech2unit import Speech2Unit

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

# import argparse
# import logging
# from tqdm import tqdm
# from speechgpt.utils.speech2unit.speech2unit import Speech2Unit
# import soundfile as sf

utils_path = '/data/lby/opencompass/SpeechGPT/speechgpt'
sys.path.insert(0, utils_path)

# import transformers

NAME = 'SpeechGPT'
META_INSTRUCTION = 'You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University.  SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n'  # noqa
DEFAULT_GEN_PARAMS = {
    'max_new_tokens': 20,
    'min_new_tokens': 5,
    'temperature': 0.8,
    'do_sample': True,
    'top_k': 60,
    'top_p': 0.8,
}
# device = torch.device('cuda')
device = torch.device('cuda:0')


def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ''
    return response


class SpeechGPT(BaseModel):

    def __init__(
        self,
        path: str,
        lora_weights: str = None,
        s2u_dir: str = 'speechgpt/utils/speech2unit/',
        vocoder_dir: str = 'speechgpt/utils/vocoder/',
        meta_template: Optional[Dict] = None,
        num_gpus: int = 2,
    ):  # noqa

        self.meta_instruction = META_INSTRUCTION
        self.template = '[Human]: {question} <eoh>. [SpeechGPT]: '

        # speech2unit
        self.s2u = Speech2Unit(ckpt_dir=s2u_dir)

        # model
        self.model = LlamaForCausalLM.from_pretrained(
            path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map='auto',
        )

        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()

        if lora_weights is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map='auto',
            )

        self.model.half()

        self.model.eval()
        if torch.__version__ >= '2' and sys.platform != 'win32':
            self.model = torch.compile(self.model)

        # tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = 'left'

        # generation
        self.generate_kwargs = DEFAULT_GEN_PARAMS

        # vocoder
        vocoder = os.path.join(vocoder_dir, 'vocoder.pt')
        vocoder_cfg = os.path.join(vocoder_dir, 'config.json')
        with open(vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(vocoder, vocoder_cfg).to(device)

        # self.output_dir = output_dir

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, True, True))

    def preprocess(
        self,
        raw_text: str,
    ):
        processed_parts = []
        for part in raw_text.split('is input:'):
            if os.path.isfile(part.strip()) and os.path.splitext(
                    part.strip())[-1] in ['.wav', '.flac', '.mp4']:
                processed_parts.append(self.s2u(part.strip(), merged=True))
            else:
                processed_parts.append(part)
        processed_text = 'is input:'.join(processed_parts)

        prompt_seq = self.meta_instruction + self.template.format(
            question=processed_text)
        return prompt_seq

    def postprocess(
        self,
        response: str,
    ):
        question = extract_text_between_tags(response,
                                             tag1='[Human]',
                                             tag2='<eoh>')
        answer = extract_text_between_tags(response + '<eoa>',
                                           tag1='[SpeechGPT] :',
                                           tag2='<eoa>')
        tq = extract_text_between_tags(
            response, tag1='[SpeechGPT] :',
            tag2='; [ta]') if '[ta]' in response else ''
        ta = extract_text_between_tags(
            response, tag1='[ta]', tag2='; [ua]') if '[ta]' in response else ''
        ua = extract_text_between_tags(
            response +
            '<eoa>', tag1='[ua]', tag2='<eoa>') if '[ua]' in response else ''

        return {
            'question': question,
            'answer': answer,
            'textQuestion': tq,
            'textAnswer': ta,
            'unitAnswer': ua
        }

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        with torch.no_grad():
            # preprocess
            preprocessed_prompts = []
            for prompt in inputs:
                preprocessed_prompts.append(self.preprocess(prompt))

            input_ids = self.tokenizer(preprocessed_prompts,
                                       return_tensors='pt',
                                       padding=True).input_ids
            for input_id in input_ids:
                if input_id[-1] == 2:
                    input_id = input_id[:, :-1]

            input_ids = input_ids.to(device)

            # generate
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                do_sample=True,
                max_new_tokens=100,
                min_new_tokens=10,
            )

            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                # max_new_tokens=1024,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(generated_ids.cpu(),
                                                    skip_special_tokens=True)

            # postprocess
            responses = [self.postprocess(x) for x in responses]

            # save responses
            for r in responses:
                if r['textAnswer'] != '':
                    # print("Transcript:", r["textQuestion"])
                    print('Text response:\n', r['textAnswer'])
                    return r['textAnswer']
                else:
                    print('Response:\n', r['answer'])
                    return r['answer'].strip()
                # json_line = json.dumps(r)
                # f.write(json_line + '\n')
            # return r["textAnswer"]
            # dump wav
            # wav = torch.tensor(0)
            # os.makedirs(f"{self.output_dir}/wav/", exist_ok=True)

    # def __call__(self, input):
    #     return self.forward(input)

    def interact(self):
        prompt = str(input(f'Please talk with {NAME}:\n'))
        while prompt != 'quit':
            try:
                self.forward([prompt])
            except Exception as e:
                traceback.print_exc()
                print(e)

            prompt = str(input(f'Please input prompts for {NAME}:\n'))
