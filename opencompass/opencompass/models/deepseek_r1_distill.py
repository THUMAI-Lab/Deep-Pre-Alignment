import json
import os
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None


def add_jsonl(js_content, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(js_content, ensure_ascii=False) + '\n')
    return


class DeepSeekR1Distill(BaseModel):

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        model_kwargs: dict = None,
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        mode: str = 'none',
        use_fastchat_template: bool = False,
        enable_thinking: bool = True,
        stop_words: List[str] = [],
    ):  # noqa

        self.logger = get_logger()
        if os.environ.get('LOCAL_PATH', ''):
            path = os.environ.get('LOCAL_PATH', path)
        self.path = path
        self.logger.info(f'### LOAD MODEL FROM: ### {path}')

        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            # repetition_penalty=1.05,
            max_tokens=32768)

        self.model = LLM(
            model=path,
            trust_remote_code=True,
            # tensor_parallel_size=4,
        )

        self.template_parser = APITemplateParser(meta_template)

        self.enable_thinking = enable_thinking

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        results = []
        for input in inputs:
            # input_ids = get_input_ids_TT(input, self.text_tokenizer)
            text_output = self._generate(input)
            # print(f"Input: {input}")
            # print(f"Output: {text_output}")
            results.append(text_output)

        return results

    def get_token_len(self, prompt: str) -> int:
        return len(self.text_tokenizer.encode(prompt, True, True))

    def _generate(self, input):
        msgs = [{
            'role': 'user',
            'content': input
            # + '\nPlease reason step by step, '
            # 'and put your final answer within \\boxed{}.'
        }]

        # print("msgs", input)

        text = self.tokenizer.apply_chat_template(msgs,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        if not self.enable_thinking:
            text = text + '<think>\n\n</think>'

        # print("*" * 100)
        # print(text)
        # print("*" * 100)

        outputs = self.model.generate([text], self.sampling_params)

        work_dir = os.environ['work_dir']
        reasoning_folder = os.path.join(work_dir, 'reasoning')
        os.makedirs(reasoning_folder, exist_ok=True)

        prompt = ''
        reasoning = ''
        generated_text = ''

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            self.logger.info(f'Prompt: {prompt!r}')
        if '</think>' in generated_text:
            reasoning = generated_text.split('</think>')[0].strip()
            generated_text = generated_text.split('</think>')[-1].strip()
            self.logger.info(f'think progress: {reasoning!r}')
            self.logger.info(f'Generated text: {generated_text!r}')
        else:
            self.logger.info(f'Generated text: {generated_text!r}')
            self.logger.info('*** NO SPECIAL TOKEN**')
        add_jsonl(
            {
                'prompt': prompt,
                'reasoning': reasoning,
                'generated_text': generated_text
            }, os.path.join(reasoning_folder, 'reasoning.jsonl'))
        return generated_text
