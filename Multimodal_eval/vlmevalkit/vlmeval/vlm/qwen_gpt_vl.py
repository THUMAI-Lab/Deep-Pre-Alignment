import os
import math
import torch
import torch.distributed as dist
import random
import numpy as np
from PIL import Image

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

from .qwen2_vl.prompt import Qwen2VLPromptMixin

# 导入所需的 transformers 和工具库
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
from io import BytesIO
from qwen_vl_utils import process_vision_info

# 定义模型特定的生成后缀
GENERATION_SUFFIX = "<|channel|>analysis<|message|>\n\n<|end|><|start|>assistant<|channel|>final<|message|>\n\n"

class QwenGPTVL(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='', use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.model_path = model_path

        print(f'Loading models from {self.model_path}')
        
        # 加载模型、处理器和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = self.model.eval()
        self.kwargs = kwargs

        # 根据推理脚本设置采样参数
        self.sampling_params = dict(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=2048,
            repetition_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def generate_inner(self, message, dataset=None):
        print(f"Generating response for message: {message}", flush=True)

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

        # --- 开始执行定制化的输入准备流程 ---

        # 1. 单独处理图像，获取元数据
        image_inputs, _ = process_vision_info(messages)
        image_processed_data = self.processor.image_processor(images=image_inputs, return_tensors="pt")
        image_grid_thw = image_processed_data["image_grid_thw"]
        pixel_values = image_processed_data["pixel_values"]

        # 2. 手动计算并生成图像占位符字符串
        merge_length = self.processor.image_processor.merge_size**2
        num_image_tokens = image_grid_thw[0].prod() // merge_length
        image_placeholder_string = self.processor.image_token * num_image_tokens
        image_placeholder_string = "<|vision_start|>" + image_placeholder_string + "<|vision_end|>"

        # 3. 使用带占位符的 chat template
        text_only_messages = [
            # 此处可以根据需要添加 system prompt
            {"role": "user", "content": [{"type": "text", "text": "IMAGE_PLACEHOLDER"}, {"type": "text", "text": prompt}]}
        ]
        raw_text = self.processor.apply_chat_template(text_only_messages, tokenize=False, add_generation_prompt=True)

        # 4. 替换占位符并添加生成后缀
        final_text = raw_text.replace("IMAGE_PLACEHOLDER", image_placeholder_string)
        final_text += GENERATION_SUFFIX

        # 5. Tokenize 文本并手动合并图像数据
        inputs = self.processor.tokenizer(
            [final_text],
            padding=True,
            return_tensors="pt",
        )
        inputs['pixel_values'] = pixel_values
        inputs['image_grid_thw'] = image_grid_thw
        inputs = inputs.to("cuda")

        # --- 输入准备结束 ---

        generated_ids = self.model.generate(**inputs, **self.sampling_params)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # 按照框架要求，返回字典格式
        # 这个模型似乎没有 <think> 标签，所以 prediction 和 detailed_prediction 相同
        res = {
            'prediction': output_text.strip(),
            'detailed_prediction': output_text.strip()
        }

        return res
    
    def generate_batch_inner(self, messages, dataset=None):
        print(f"Generating batch with {len(messages)} messages.", flush=True)

        prompts, image_paths = self.messages_to_promptimg(messages, dataset=dataset)

        # --- 开始执行定制化的批量输入准备流程 ---
        
        batch_messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": pmt}]} for pmt, img_path in zip(prompts, image_paths)]

        # 1. 批量处理图像
        image_inputs, _ = process_vision_info(batch_messages)
        image_processed_data = self.processor.image_processor(images=image_inputs, return_tensors="pt")
        image_grid_thw_list = image_processed_data["image_grid_thw"]
        pixel_values = image_processed_data["pixel_values"]

        final_texts = []
        merge_length = self.processor.image_processor.merge_size**2

        # 循环为批次中的每个样本生成其独特的文本 prompt
        for i, prompt in enumerate(prompts):
            # 2. 为当前样本计算图像占位符
            image_grid_thw = image_grid_thw_list[i]
            num_image_tokens = image_grid_thw.prod() // merge_length
            image_placeholder_string = self.processor.image_token * num_image_tokens
            image_placeholder_string = "<|vision_start|>" + image_placeholder_string + "<|vision_end|>"
            
            # 3. 构建带占位符的模板
            text_only_messages = [
                {"role": "user", "content": [{"type": "text", "text": "IMAGE_PLACEHOLDER"}, {"type": "text", "text": prompt}]}
            ]
            raw_text = self.processor.apply_chat_template(text_only_messages, tokenize=False, add_generation_prompt=True)
            
            # 4. 替换占位符并添加后缀
            final_text = raw_text.replace("IMAGE_PLACEHOLDER", image_placeholder_string)
            final_text += GENERATION_SUFFIX
            final_texts.append(final_text)

        # 5. 批量 Tokenize 文本并合并图像数据
        inputs = self.tokenizer(final_texts, padding=True, return_tensors="pt")
        inputs['pixel_values'] = pixel_values
        inputs['image_grid_thw'] = image_grid_thw_list
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # --- 输入准备结束 ---

        generated_ids = self.model.generate(**inputs, **self.sampling_params)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        results_list = []
        for text in output_texts:
            results_list.append({
                'prediction': text.strip(),
                'detailed_prediction': text.strip(),
            })

        return results_list