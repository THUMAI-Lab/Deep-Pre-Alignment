# 文件名: duplex_thinker.py (与您的主脚本放在同一目录下)

import torch
from typing import Optional, List, Union

####### copy from qwen_vl
import regex as re
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import (BatchFeature, PretrainedConfig, PreTrainedTokenizer,
                          TensorType)
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.resampler import Resampler2, get_abs_pos
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from vllm.model_executor.models.qwen import QWenBaseModel, QWenModel
from vllm.model_executor.models.utils import flatten_bn, merge_multimodal_embeddings
####### end copy


####### copy from qwen2_vl
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, Literal, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoConfig, BatchFeature
from transformers.models.qwen2_vl import (Qwen2VLImageProcessor,
                                          Qwen2VLProcessor)
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig, Qwen2VLVisionConfig)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen2_vl.video_processing_qwen2_vl import (
    Qwen2VLVideoProcessor)

from vllm.config import VllmConfig
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, VideoItem)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import _Backend, current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope
from vllm.transformers_utils.processor import (
    cached_image_processor_from_config)
from vllm.transformers_utils.tokenizer import AnyTokenizer

from vllm.model_executor.models.interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)
from vllm.model_executor.models.vision import get_vit_attn_backend
####### end copy

# from vllm.model_executor.models.interfaces import SupportsMultiModalWithRawInput

# 从 VLLM 的 qwen3 实现中直接导入正确的父类
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, VideoItem)
from vllm.multimodal import MULTIMODAL_REGISTRY
from collections.abc import Iterable, Mapping, Sequence
from vllm.multimodal.profiling import BaseDummyInputsBuilder
import os

# HIDDEN_SIZE = 2560
# HIDDEN_SIZE = 5120
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE_OF_MODEL", 2560))
VISUAL_BANDWIDTH = 64

def get_visual_message_tokens():
    size = VISUAL_BANDWIDTH
    tokens = [f'<im_msg-{i}>' for i in range(size)]
    return tokens

# def get_visual_message_token_ids(tokenizer: PreTrainedTokenizer) -> List[int]:
def get_visual_message_token_ids() -> List[int]:
    """
    获取视觉消息的 token ID 列表。
    """
    size = VISUAL_BANDWIDTH
    begin_token_id = 151669
    return [begin_token_id + i for i in range(size)]
    # visual_message_tokens = get_visual_message_tokens()
    # return tokenizer.convert_tokens_to_ids(visual_message_tokens)

class DuplexThinkerProcessor:
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # print(f"DuplexThinkerProcessor initialized with config: {config}, tokenizer: {tokenizer}")

    @property
    def visual_message_start(self) -> str:
        return get_visual_message_tokens()[0]

    @property
    def visual_message_end(self) -> str:
        return get_visual_message_tokens()[-1]
    
    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        image_embeds: Optional[list[torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:

        # print(f"DuplexThinkerProcessor called with text: {text}, image_embeds: {image_embeds}, return_tensors: {return_tensors}")

        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]

        # if visual_message is None:
        #     visual_inputs = {}
        # else:
        #     visual_inputs = {
        #         "image_embeds": visual_message,
        #     }


        if image_embeds is None:
            visual_inputs = {}
        else:
            visual_inputs = {
                "image_embeds": image_embeds,
            }

        text_inputs = self.tokenizer(text)
    
        # print(f"processor return features: {text_inputs}, visual inputs: {visual_inputs}")
    
        return BatchFeature(
            {
                **text_inputs,
                **visual_inputs,
            },
            tensor_type=return_tensors,
        )



class DuplexThinkerProcessingInfo(BaseProcessingInfo):

    # def get_tokenizer(self) -> PretrainedTokenizer:
    #     tokenizer = self.ctx.tokenizer
    #     assert isinstance(tokenizer, PretrainedTokenizer)

    #     return _get_tokenizer_withu

    def get_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = self.ctx.tokenizer

        # TODO: check tokenizer

        return tokenizer


    def get_hf_processor(self, **kwargs: object) -> DuplexThinkerProcessor:
        return self.ctx.init_processor(
            DuplexThinkerProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {
            # "image": VISUAL_BANDWIDTH,  # 图像嵌入的最大数量
            "image": 20480,  # 图像嵌入的最大数量
            # "audio": None,  # None 表示没有限制
        }

# TODO
class DuplexThinkerDummyInputsBuilder(BaseDummyInputsBuilder[DuplexThinkerProcessingInfo]):

    # size_emb=1024

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # num_img_embs = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        # assert num_images == VISUAL_BANDWIDTH

        # visual_message_tokens = get_visual_message_tokens()
        # return "".join(visual_message_tokens)
        return "<|vision_start|>" + "<|image_pad|>" * (num_images - 2) + "<|vision_end|>"
    
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        
        # return random tensor with shape (num_images, 1, HIDDEN_SIZE)
        return {
            "image": torch.randn(num_images, 1, HIDDEN_SIZE),
        }

        # num_img_embs = mm_counts.get("audio", 0)
        # print(f"num_img_embs: {num_img_embs}")
        # return {
        #     # "audio": self._get_dummy_audios(length=self.size_emb, num_audios=num_img_embs),
        # }


class DuplexThinkerMultiModalProcessor(BaseMultiModalProcessor[DuplexThinkerProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:

        # print(f"in processor _call_hf_processor, prompt: {prompt}, mm_data: {mm_data}, mm_kwargs: {mm_kwargs}, tok_kwargs: {tok_kwargs}")

        image_embeds = mm_data.get("image_embeds")
        # if image_embeds is not None:
        #     print(f"type of visual_message: {type(visual_message)}")
        #     # TODO: assert

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )


    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            image_embeds=MultiModalFieldConfig.batched("image")
        )
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:

        tokenizer = self.info.get_tokenizer()
        visual_message_tokens = get_visual_message_tokens()
        visual_message_token_ids = tokenizer.convert_tokens_to_ids(visual_message_tokens)
        
        # print(f"key in mm_items: {mm_items.keys()}")
        # for key, value in mm_items.items():
        #     print(f"key: {key}, type of value: {type(value)}")
        #     if isinstance(value, torch.Tensor):
        #         print(f"shape of tensor: {value.shape}")
        # assert False
        # print(f"out_mm_kwargs: {out_mm_kwargs}")
        image = mm_items.get("image")
        len_image = len(image)

        image_tokens = [151652] + [151655] * (len_image - 2) + [151653]

        # assert False, f"len_image: {len_image}, visual_message_token_ids: {visual_message_token_ids}, tokenizer: {tokenizer}"
        replace = [PromptReplacement(
            modality="image",
            target=[151652],
            replacement=[151652]
        ), PromptReplacement(
            modality="image",
            target=[151653],
            replacement=[151653]
        ), PromptReplacement(
            modality="image",
            target=[151655],
            replacement=[151655]
        )]
        return replace
        # return [
        #     PromptReplacement(
        #         modality="image",
        #         target=[token_id],
        #         replacement=[token_id],
        #     ) for token_id in visual_message_token_ids
        # ]

        # visual_message_tokens = get_visual_message_tokens()
        # visual_message_token_ids = tokenizer.convert_tokens_to_ids(visual_message_tokens)

        # visual_message = mm_items.get("image_embeds")


        # return None
        # return []

@MULTIMODAL_REGISTRY.register_processor(DuplexThinkerMultiModalProcessor,
                                        info=DuplexThinkerProcessingInfo,
                                        dummy_inputs=DuplexThinkerDummyInputsBuilder)
class DuplexThinkerForCausalLMForward(Qwen3ForCausalLM, SupportsMultiModal):
# class DuplexThinkerForCausalLM(Qwen3ForCausalLM, SupportsMultiModalWithRawInput):
    """
    精确继承自 vllm.model_executor.models.qwen3.Qwen3ForCausalLM。
    重写顶层的 forward 方法，实现视觉特征的注入。
    """

    # def __init__(self, vllm_config, model_config, **kwargs):
    #     """
    #     初始化方法，调用父类的初始化。
    #     """
    #     print(f"Initializing DuplexThinkerForCausalLM with vllm_config: {vllm_config}, model_config: {model_config}, kwargs: {kwargs}")
    #     super().__init__(vllm_config, model_config, **kwargs)

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        print(f"get_multimodal_embeddings keys in kwargs: {kwargs.keys()}")

    # def forward(
    #     self,
    #     *args,
    #     **kwargs
    # ):
    #     print(f"keys in kwargs: {kwargs.keys()}")
    #     return super().forward(
    #         *args,
    #         **kwargs
    #     )

    def get_multimodal_embeddings(
        self,
        **kwargs: object,
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        # print(f"get_multimodal_embeddings called with image_input: {image_input}")
        if image_input is None:
            return None
        return image_input

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        return inputs_embeds
        # print(f"get_input_embeddings called with input_ids: {input_ids}, multimodal_embeddings: {multimodal_embeddings}")
        # print(f"shape of inputs_embeds: {inputs_embeds.shape}, shape of multimodal_embeddings: {multimodal_embeddings.shape if multimodal_embeddings is not None else None}")
        # # assert False
        # visual_message_tokens = get_visual_message_tokens()
        # # visual_message_token_ids = self.tokenizer.convert_tokens_to_ids(visual_message_tokens)
        # visual_message_token_ids = get_visual_message_token_ids()

        # visual_message_start_id, visual_message_end_id = visual_message_token_ids[0], visual_message_token_ids[-1]
        # print(f"visual_message_start_id: {visual_message_start_id}, visual_message_end_id: {visual_message_end_id}")
        # visual_message_mask = (input_ids >= visual_message_start_id) & (input_ids <= visual_message_end_id)

        # visual_message_mask_count = visual_message_mask.sum(dim=-1)
        # assert visual_message_mask_count.max() == VISUAL_BANDWIDTH

        # mask = visual_message_mask.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
        # final_embeds = inputs_embeds.clone()


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        # 1. 从 kwargs 中获取我们传入的视觉特征
        # precomputed_features_list = kwargs.get("inputs_embeds_list")
        image_embeds = kwargs.pop("image_embeds", None)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        # print(f"shape of inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else None}")
        # print(f"shape of input_ids: {input_ids.shape if input_ids is not None else None}")

        if input_ids is not None and image_embeds is not None:
            visual_message_start_id = 151652
            visual_message_end_id = 151653
            visual_message_pad_id = 151655
            
            # === 关键修复：处理一维input_ids的情况 ===
            # vLLM在某些情况下会将整个batch展平成一维tensor
            # 例如：torch.Size([3604]) 而不是 torch.Size([4, 901])
            
            if len(input_ids.shape) == 1:
                # 一维情况：整个batch被展平了
                # 这种情况下，我们无法区分不同的样本
                # 但我们可以直接处理所有的视觉token
                
                print(f"Input_ids is 1D: shape={input_ids.shape}, treating as flattened batch")
                
                if isinstance(image_embeds, list):
                    # 多个image_embeds：需要按顺序注入
                    # 找到所有视觉token的位置
                    visual_message_mask = (input_ids == visual_message_start_id) | \
                                         (input_ids == visual_message_end_id) | \
                                         (input_ids == visual_message_pad_id)
                    
                    if visual_message_mask.sum() > 0:
                        print(f"Found {visual_message_mask.sum()} visual tokens in flattened batch")
                        print(f"Have {len(image_embeds)} image_embeds to inject")
                        
                        # 将所有image_embeds连接起来
                        all_image_embeds = torch.cat([
                            emb.reshape(-1, HIDDEN_SIZE) for emb in image_embeds
                        ], dim=0)  # [total_visual_tokens, HIDDEN_SIZE]
                        
                        all_image_embeds = all_image_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
                        
                        # 注入到一维tensor
                        image_mask = visual_message_mask.unsqueeze(-1).expand_as(inputs_embeds)
                        inputs_embeds = inputs_embeds.masked_scatter(
                            image_mask,
                            all_image_embeds.flatten()
                        )
                        print(f"✓ Injected visual features into flattened batch")
                else:
                    # 单个image_embeds
                    visual_message_mask = (input_ids == visual_message_start_id) | \
                                         (input_ids == visual_message_end_id) | \
                                         (input_ids == visual_message_pad_id)
                    
                    if visual_message_mask.sum() > 0:
                        image_embeds = image_embeds.reshape(-1, HIDDEN_SIZE)
                        image_embeds = image_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
                        
                        image_mask = visual_message_mask.unsqueeze(-1).expand_as(inputs_embeds)
                        inputs_embeds = inputs_embeds.masked_scatter(
                            image_mask,
                            image_embeds.flatten()
                        )
                        print(f"✓ Injected single visual features into flattened batch")
            
            else:
                # 二维情况：[batch_size, seq_len]
                
                if isinstance(image_embeds, list):
                    # 混合batch：维护独立索引
                    image_idx = 0
                    
                    for batch_idx in range(input_ids.shape[0]):
                        visual_message_mask = (input_ids[batch_idx] == visual_message_start_id) | \
                                             (input_ids[batch_idx] == visual_message_end_id) | \
                                             (input_ids[batch_idx] == visual_message_pad_id)
                        
                        if visual_message_mask.sum() > 0:
                            if image_idx < len(image_embeds):
                                print(f"✓ Batch[{batch_idx}]: Injecting visual features from image_embeds[{image_idx}]")
                                
                                image_embeds_i = image_embeds[image_idx].reshape(-1, HIDDEN_SIZE)
                                image_embeds_i = image_embeds_i.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
                                
                                image_mask = visual_message_mask.unsqueeze(-1).expand_as(inputs_embeds[batch_idx])
                                inputs_embeds[batch_idx] = inputs_embeds[batch_idx].masked_scatter(
                                    image_mask,
                                    image_embeds_i.flatten()
                                )
                                
                                image_idx += 1
                            else:
                                print(f"⚠️ Batch[{batch_idx}]: Needs visual features but image_embeds exhausted!")
                                
                else:
                    # 单个tensor
                    image_embeds = image_embeds.reshape(-1, HIDDEN_SIZE)
                    image_embeds = image_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
                    
                    for batch_idx in range(input_ids.shape[0]):
                        visual_message_mask = (input_ids[batch_idx] == visual_message_start_id) | \
                                             (input_ids[batch_idx] == visual_message_end_id) | \
                                             (input_ids[batch_idx] == visual_message_pad_id)
                        
                        if visual_message_mask.sum() > 0:
                            print(f"✓ Batch[{batch_idx}]: Injecting single visual features")
                            
                            image_mask = visual_message_mask.unsqueeze(-1).expand_as(inputs_embeds[batch_idx])
                            inputs_embeds[batch_idx] = inputs_embeds[batch_idx].masked_scatter(
                                image_mask,
                                image_embeds.flatten()
                            )
                            break

        #     # visual_message_tokens = get_visual_message_tokens()
        #     # visual_message_token_ids = get_visual_message_token_ids()

        #     # print(f"keys in kwargs: {kwargs.keys()}")
        #     # print(f"input_ids: {input_ids}, positions: {positions}, intermediate_tensors: {intermediate_tensors}, inputs_embeds: {inputs_embeds}")
        #     # print(f"input_ids list: {input_ids.tolist()}, positions: {positions}, intermediate_tensors: {intermediate_tensors}, inputs_embeds: {inputs_embeds}")
        #     # print(f"image_embeds: {image_embeds}") # [b, VISUAL_BANDWIDTH, 1, HIDDEN_SIZE]
        #     # print(f"shape of image_embeds: {image_embeds.shape}, shape of inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else None}")
            
        #     # if image_embeds is a list
        #     if isinstance(image_embeds, list):
        #         for i in range(len(image_embeds)):
        #             print(f"image_embeds[i]: {image_embeds[i]}")
        #             assert isinstance(image_embeds[i], torch.Tensor), f"image_embeds[i] should be a torch.Tensor, got {type(image_embeds[i])}"
        #             image_embeds_i = image_embeds[i].reshape(-1, HIDDEN_SIZE)
        #             visual_message_start_id = 151652
        #             visual_message_end_id = 151653
        #             visual_message_pad_id = 151655
        #             visual_message_mask = (input_ids[i] == visual_message_start_id) | (input_ids[i] == visual_message_end_id) | (input_ids[i] == visual_message_pad_id)
        #             visual_message_mask_count = visual_message_mask.sum(dim=-1)
        #             if visual_message_mask_count.max() != 0:
        #                 print(f"yes!!!!")
        #                 image_mask = visual_message_mask.unsqueeze(-1)
        #                 image_embeds_i = image_embeds_i.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
        #                 inputs_embeds[i] = inputs_embeds[i].masked_scatter(image_mask, image_embeds_i)
        #             else:
        #                 print(f"no!!!!")
        #     else:
        #         image_embeds = image_embeds.reshape(-1, HIDDEN_SIZE)
        #         # visual_message_start_id, visual_message_end_id = visual_message_token_ids[0], visual_message_token_ids[-1]
        #         # print(f"visual_message_start_id: {visual_message_start_id}, visual_message_end_id: {visual_message_end_id}")
        #         visual_message_start_id = 151652
        #         visual_message_end_id = 151653
        #         visual_message_pad_id = 151655
        #         visual_message_mask = (input_ids == visual_message_start_id) | (input_ids == visual_message_end_id) | (input_ids == visual_message_pad_id)
        #         # visual_message_mask = (input_ids >= visual_message_start_id) & (input_ids <= visual_message_end_id)
        #         # print(f"visual_message_mask: {visual_message_mask}")
        #         # print(f"shape of visual_message_mask: {visual_message_mask.shape}")
        #         visual_message_mask_count = visual_message_mask.sum(dim=-1)
        #         # print(f"visual_message_mask_count: {visual_message_mask_count}")
        #         # print(f"shape of input_embeds: {inputs_embeds.shape if inputs_embeds is not None else None}")
        #         visual_message_mask_unsqueezed = visual_message_mask.unsqueeze(-1)

        #         visual_message_mask_expanded = visual_message_mask_unsqueezed.expand_as(inputs_embeds)

        #         if visual_message_mask_count.max() != 0:
        #             print(f"yes!!!!")
        #             # assert visual_message_mask_count.max() % VISUAL_BANDWIDTH == 0, "visual_message_mask_count.max() should be a multiple of VISUAL_BANDWIDTH"
        #             # assert visual_message_mask_count.max() == image_embeds.shape[0], f"visual_message_mask_count.max() should be equal to image_embeds.shape[0], got {visual_message_mask_count.max()} and {image_embeds.shape[0]}"
        #             image_mask = visual_message_mask_expanded.to(inputs_embeds.device)
        #             image_embeds = image_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
        #             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        #         else:
        #             print(f"no!!!!")
        
            # print(f"to model, inputs_embeds: {inputs_embeds}, positions: {positions}, intermediate_tensors: {intermediate_tensors}")

        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states


        # 2. 如果没有视觉特征，执行原始的标准流程
        # assert precomputed_features_list is not None, "precomputed_features_list should not be None"
        # if precomputed_features_list is None:
        #     return super().forward(
        #         input_ids=input_ids,
        #         positions=positions,
        #         intermediate_tensors=intermediate_tensors,
        #         inputs_embeds=inputs_embeds,
        #         # 注意：这里不传递 **kwargs，因为父类的 forward 不接收它
        #     )



        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,  # 传递其他参数
        )

        # # 3. 如果有视觉特征，执行注入逻辑
        
        # # self.get_input_embeddings 是从父类继承来的方法，非常方便
        # word_embeddings = self.get_input_embeddings(input_ids)

        # # 确认占位符 ID
        # placeholder_id = 151668  # 请再次确认这个 ID
        # final_embeds = word_embeddings.clone()

        # for i in range(final_embeds.shape[0]):
        #     placeholder_mask = (input_ids[i] == placeholder_id)
        #     if placeholder_mask.any():
        #         image_features = precomputed_features_list[i]
        #         num_placeholders = placeholder_mask.sum()
        #         num_features = image_features.shape[0]

        #         if num_placeholders != num_features:
        #             raise ValueError(f"Sample {i}: Mismatch")
                
        #         final_embeds[i, placeholder_mask] = image_features.to(final_embeds.device, dtype=final_embeds.dtype)

        # # 4. 调用父类的 forward 方法，并传入我们构造好的 final_embeds
        # #    这是最关键的一步：我们利用了父类已有的 inputs_embeds 参数入口
        # return super().forward(
        #     input_ids=None,  # 必须设为 None，让父类使用我们提供的 inputs_embeds
        #     positions=positions,
        #     intermediate_tensors=intermediate_tensors,
        #     inputs_embeds=final_embeds, # 传入我们修改后的 embeddings
        # )