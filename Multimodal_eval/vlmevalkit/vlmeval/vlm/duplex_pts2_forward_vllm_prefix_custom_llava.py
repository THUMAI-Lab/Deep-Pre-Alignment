import math
import torch
import torch.distributed as dist
import random
import numpy as np
from PIL import Image

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

# from .vllm_jointmodel import VLLMJointModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin
from transformers import AutoModelForCausalLM

from vllm import SamplingParams

import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models import ModelRegistry
import transformers.models.qwen3.modeling_qwen3 as qwen3

from .duplex_thinker_forward import DuplexThinkerForCausalLMForward

ModelRegistry.register_model("Qwen3ForCausalLM", DuplexThinkerForCausalLMForward)

CUSTOM_PREFIX = os.environ.get("CUSTOM_PREFIX", "<think>\n\n</think>\n\n")
Temperature = float(os.getenv('TEMPERATURE', 0.01))
Prompt = os.getenv('PROMPT', '')
MaxImageResolution = os.getenv('MAX_IMAGE_RESOLUTION', None)
USE_2ND_LAYER = os.getenv("USE_2ND_LAYER", "false").lower() == "true"
USE_NORM_LAYER = os.getenv("USE_NORM_LAYER", "false").lower() == "true"
USE_CONTEXTUAL_ENCODING = os.getenv("USE_CONTEXTUAL_ENCODING", "false").lower() == "true"
SKIP_PERCEIVER_LM = os.getenv("SKIP_PERCEIVER_LM", "false").lower() == "true"
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.").strip()

if MaxImageResolution is not None:
    try:
        MaxImageResolution = int(MaxImageResolution)
    except ValueError:
        print(f"Warning: MAX_IMAGE_RESOLUTION={MaxImageResolution} is not a valid integer, ignoring it")
        MaxImageResolution = None

def get_visual_message_tokens(num_tokens: int) -> List[str]:
    return [f'<im_msg-{i}>' for i in range(num_tokens)]

@torch.inference_mode()
def extract_batch_image_features(batch: List[Dict[str, Any]], perceiver_model, perceiver_processor, alignment_layer, device, norm_layer=None) -> List[torch.Tensor]:
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    p_prompt_template = "{question}"
    
    p_texts = []
    for item in batch:
        p_prompt = p_prompt_template.format(question=item["question"])
        if not USE_CONTEXTUAL_ENCODING:
            p_message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': p_prompt}]}]
                        #  {'role': 'assistant', 'content': [{'type': 'text', 'text': visual_message_str}]}]
        else:
            p_message = [{'role': 'user', 'content': [{'type': 'text', 'text': p_prompt}, {'type': 'image'}]}]
        if SYSTEM_PROMPT:
            p_message.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
        p_texts.append(perceiver_processor.apply_chat_template(p_message, tokenize=False, add_generation_prompt=False))
        print(f"Prepared perceiver input text for sample: {p_texts[-1]}")

    inputs = perceiver_processor(text=p_texts, images=images, return_tensors="pt", padding=True).to(device)
    outputs = perceiver_model(**inputs, output_hidden_states=True)
    if SKIP_PERCEIVER_LM:
        print("SKIP_PERCEIVER_LM: Using embedding layer (hidden_states[0]) for feature extraction.")
        middle_hidden_states = outputs.hidden_states[0]
    elif USE_2ND_LAYER:
        print("Using second to last hidden states for feature extraction.")
        middle_hidden_states = outputs.hidden_states[-2]
    else:
        print("Using last hidden states for feature extraction.")
        middle_hidden_states = outputs.hidden_states[-1]
    if USE_NORM_LAYER:
        print("Applying normalization layer to extracted features.")
        middle_hidden_states = norm_layer(middle_hidden_states)
    # p_msg_token_ids = perceiver_processor.tokenizer.convert_tokens_to_ids(get_visual_message_tokens(VISUAL_BANDWIDTH))
    # p_msg_start_id, p_msg_end_id = p_msg_token_ids[0], p_msg_token_ids[-1]
    p_msg_start_id, p_msg_end_id = 151652, 151653
    
    batch_features = []
    for i in range(len(batch)):
        input_ids_sample = inputs.input_ids[i]
        start_indices = (input_ids_sample == p_msg_start_id).nonzero(as_tuple=True)[0]
        end_indices = (input_ids_sample == p_msg_end_id).nonzero(as_tuple=True)[0]
        if len(start_indices) == 0 or len(end_indices) == 0:
            raise ValueError(f"Visual message tokens not found in sample {i}")
        features = middle_hidden_states[i, start_indices[0]:end_indices[0] + 1, :]
        if SKIP_PERCEIVER_LM:
            batch_features.append(features)
        else:
            aligned_features = alignment_layer(features)
            batch_features.append(aligned_features)
    return batch_features

def batch_inference(batch_data: List[Dict[str, Any]], llm: LLM, thinker_tokenizer, perceiver_model, perceiver_processor, alignment_layer, device, norm_layer=None):
    print(f"\n--- Starting batch inference for {len(batch_data)} samples ---")
    
    print("Step 1: Extracting visual features...")
    extracted_features = extract_batch_image_features(batch_data, perceiver_model, perceiver_processor, alignment_layer, device, norm_layer=norm_layer)
    
    print("Step 2: Preparing prompts for VLLM...")
    prompts_for_vllm = []
    # visual_message_str = "".join(get_visual_message_tokens(VISUAL_BANDWIDTH))
    # t_prompt_template = '{question} Image: ' + visual_message_str
    # t_prompt_template = '{question} {prompt}Image: ' + visual_message_str
    if not USE_CONTEXTUAL_ENCODING:
        t_prompt_template = '{image}{question}'
    else:
        t_prompt_template = '{question}{image}'
    # t_prompt_template = '{question} Your response must be a single letter: A, B, C, or D. Image: ' + visual_message_str
    for idx, item in enumerate(batch_data):
        # t_prompt = t_prompt_template.format(question=item["question"])
        # t_prompt = t_prompt_template.format(question=item["question"], prompt=Prompt)
        extracted_feature_shape = extracted_features[idx].shape
        image_placeholder = "<|vision_start|>" + "<|image_pad|>" * (extracted_feature_shape[0] - 2) + "<|vision_end|>\n"
        t_prompt = t_prompt_template.format(image=image_placeholder, question=item["question"])
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": t_prompt}
        ]
        final_prompt = thinker_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        ####### difference between _prefix
        final_prompt = final_prompt + CUSTOM_PREFIX
        #######
        prompts_for_vllm.append(final_prompt)
        
    print(f"shape of extracted_features: {[feat.shape for feat in extracted_features]}")

    print("Step 3: Assembling structured requests for VLLM...")
    requests = []
    for i in range(len(prompts_for_vllm)):
        extracted_features[i] = extracted_features[i].unsqueeze(1).to("cpu").to(torch.float32)
        print(f"Feature shape for sample {i}: {extracted_features[i].shape}")
        request = {
            "prompt": prompts_for_vllm[i],
            "multi_modal_data": {
                "image": extracted_features[i]
            }
        }
        requests.append(request)

    print("Step 4: Running batched generation with VLLM...")
    sampling_params = SamplingParams(temperature=Temperature, top_p=0.9, max_tokens=4096)
    print(f"using sampling params: {sampling_params}")
    # assert False
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=30)
    
    outputs = llm.generate(requests, sampling_params)
    
    res = []
    for i, output in enumerate(outputs):
        if output.outputs:
            res.append(output.outputs[0].text.strip())
        else:
            res.append("No output generated.")
    
    print(f"Generated {len(res)} responses.")
    return res

class DuplexThinkerS2ForwardvLLMPrefixCustomLLaVA(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="", use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.model_path = model_path
        self.perceiver_device = "cuda:0"
        
        SEPARATED_MODELS_DIR = Path(model_path)
        PERCEIVER_PATH = SEPARATED_MODELS_DIR / "perceiver"
        THINKER_PATH = SEPARATED_MODELS_DIR / "thinker"
        ALIGNMENT_LAYER_PATH = SEPARATED_MODELS_DIR / "linear_align_dim.pth"
        NORM_LAYER_PATH = SEPARATED_MODELS_DIR / "norm_weights.pth"
        VISUAL_BANDWIDTH = 64

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("DualThinker with VLLM requires at least one CUDA GPU.")

        self.perceiver_processor = AutoProcessor.from_pretrained(PERCEIVER_PATH)
        # 从环境变量读取最大分辨率，如果指定了则设置 image_processor 的 max_pixels
        if MaxImageResolution is not None:
            if hasattr(self.perceiver_processor, 'image_processor'):
                self.perceiver_processor.image_processor.max_pixels = MaxImageResolution
                print(f"Set image_processor.max_pixels to {MaxImageResolution} (from MAX_IMAGE_RESOLUTION env var)")
            else:
                print(f"Warning: processor does not have image_processor attribute, MAX_IMAGE_RESOLUTION will be ignored")
        
        from .modeling_llava_baseline import LLaVABaselineModelForConditionalGeneration
        
        # 加载基础模型以获取配置
        self.perceiver_model = LLaVABaselineModelForConditionalGeneration.from_pretrained(
            PERCEIVER_PATH,
            torch_dtype=torch.bfloat16,
        ).to(self.perceiver_device).eval()
        print("Perceiver model and processor loaded.")

        self.thinker_config = AutoConfig.from_pretrained(THINKER_PATH, trust_remote_code=True)

        if not SKIP_PERCEIVER_LM:
            self.alignment_layer = torch.nn.Sequential(
                torch.nn.Linear(self.perceiver_model.model.vlm.language_model.config.hidden_size, self.perceiver_model.model.vlm.language_model.config.hidden_size, dtype=torch.bfloat16),
                torch.nn.ReLU(),
                torch.nn.Linear(self.perceiver_model.model.vlm.language_model.config.hidden_size, self.thinker_config.hidden_size, dtype=torch.bfloat16),
            ).to(self.perceiver_device).eval()
            self.alignment_layer.load_state_dict(torch.load(ALIGNMENT_LAYER_PATH))
            print("Alignment layer loaded.")
        else:
            self.alignment_layer = None
            print("SKIP_PERCEIVER_LM: Skipping alignment layer (hidden_size already matches thinker).")

        if USE_NORM_LAYER:
            if os.path.exists(NORM_LAYER_PATH):
                self.norm_layer = qwen3.Qwen3RMSNorm(self.perceiver_model.model.vlm.language_model.config.hidden_size, eps=1e-6).to(self.perceiver_device).eval()
                self.norm_layer.load_state_dict(torch.load(NORM_LAYER_PATH))
                print("Normalization layer loaded.")
            else:
                # clone perceiver.model.vlm.language_model.norm as norm_layer
                self.norm_layer = qwen3.Qwen3RMSNorm(self.perceiver_model.model.vlm.language_model.config.hidden_size, eps=1e-6).to(self.perceiver_device).eval()
                self.norm_layer.load_state_dict(self.perceiver_model.model.vlm.language_model.norm.state_dict())
                print("Normalization layer not found on disk, cloned from perceiver model.")
            self.norm_layer.to(self.perceiver_device).to(torch.bfloat16).eval()
        else:
            self.norm_layer = None

        if not SKIP_PERCEIVER_LM:
            print(f"weight of alignment layer: {self.alignment_layer[0].weight}, bias: {self.alignment_layer[0].bias}")
            print(f"wieght of alignment layer: {self.alignment_layer[2].weight}, bias: {self.alignment_layer[2].bias}")

        self.thinker_tokenizer = AutoTokenizer.from_pretrained(THINKER_PATH, padding_side="left")

        # get gpu count
        gpu_count = torch.cuda.device_count()

        self.llm = LLM(
            model=str(THINKER_PATH),
            trust_remote_code=True,
            tensor_parallel_size=gpu_count,
            gpu_memory_utilization=0.6,
            # enforce_eager=True,
            limit_mm_per_prompt={"image": 20480},
            max_num_seqs=8
        )

        # print(f'Loading vllm model from {self.model_path}')
        
        # self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        # self.model = self.model.eval().cuda()

        # self.kwargs = kwargs

        # self.p_sampling_params = dict(do_sample=True, temperature=0.01, top_p=0.001, top_k=1, max_new_tokens=2048, repetition_penalty=1.0)
        # self.t_sampling_params = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=32768)


    def generate_inner(self, message, dataset=None):
        print(f"Generating response for message: {message}", flush=True)
        print("----------------------------------------------")
        # assert False

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        
        # image = Image.open(image_path).convert('RGB')

        # image = self._resize_image_if_needed(image)

        # msgs = [{'role': 'user', 'content': prompt}]

        # print(f"Prepared message for generation: {msgs}", flush=True)

        # res = self.model.chat(
        #     [image_path],
        #     [msgs],
        #     perceiver_generation_params=self.p_sampling_params,
        #     thinker_generation_params=self.t_sampling_params,
        # )

        batch = [
            {"image_path": image_path, "question": prompt}
        ]

        return batch_inference(
            batch,
            self.llm,
            self.thinker_tokenizer,
            self.perceiver_model,
            self.perceiver_processor,
            self.alignment_layer,
            self.perceiver_device,
            self.norm_layer
        )

    def generate_batch_inner(self, messages, dataset=None):
        print(f"Generating batch with {len(messages)} messages.", flush=True)
        print("----------------------------------------------")

        prompts, image_paths = self.messages_to_promptimgs(messages, dataset=dataset)
        
        # images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        # images = [self._resize_image_if_needed(image) for image in images]

        batch = []
        for prompt, image_path in zip(prompts, image_paths):
            batch.append({"image_path": image_path, "question": prompt})
        print(f"Prepared {len(batch)} messages for batch generation.", flush=True)
        return batch_inference(
            batch,
            self.llm,
            self.thinker_tokenizer,
            self.perceiver_model,
            self.perceiver_processor,
            self.alignment_layer,
            self.perceiver_device,
            self.norm_layer
        )

        # msgs = [{'role': 'user', 'content': prompt} for prompt in prompts]

        # print(f"Prepared {len(msgs)} messages for batch generation.", flush=True)

        # res = self.model.chat(
        #     image_paths,
        #     msgs,
        #     perceiver_generation_params=self.p_sampling_params,
        #     thinker_generation_params=self.t_sampling_params,
        # )

        # return res
