from .accessory import LLaMA2AccessoryModel  # noqa: F401
from .ai360_api import AI360GPT  # noqa: F401
from .alaya import AlayaLM  # noqa: F401
from .baichuan_api import BaiChuan  # noqa: F401
from .baidu_api import ERNIEBot  # noqa: F401
from .bailing_api_oc import BailingAPI  # noqa: F401
from .base import BaseModel, LMTemplateParser  # noqa: F401
from .base_api import APITemplateParser, BaseAPIModel  # noqa: F401
from .bluelm_api import BlueLMAPI  # noqa: F401
from .bytedance_api import ByteDance  # noqa: F401
from .claude_allesapin import ClaudeAllesAPIN  # noqa: F401
from .claude_api import Claude  # noqa: F401
from .claude_sdk_api import ClaudeSDK  # noqa: F401
from .deepseek_api import DeepseekAPI  # noqa: F401
from .deepseek_r1_distill import DeepSeekR1Distill  # noqa: F401
from .doubao_api import Doubao  # noqa: F401
from .gemini_api import Gemini  # noqa: F401
from .glm import GLM130B  # noqa: F401
from .huggingface import HuggingFace  # noqa: F401
from .huggingface import HuggingFaceCausalLM  # noqa: F401
from .huggingface import HuggingFaceChatGLM3  # noqa: F401
from .huggingface_above_v4_33 import HuggingFaceBaseModel  # noqa: F401
from .huggingface_above_v4_33 import HuggingFacewithChatTemplate  # noqa: F401
from .hunyuan_api import Hunyuan  # noqa: F401
from .intern_model import InternLM  # noqa: F401
from .interntrain import InternTrain  # noqa: F401
from .krgpt_api import KrGPT  # noqa: F401
from .lightllm_api import LightllmAPI, LightllmChatAPI  # noqa: F401
from .llama2 import Llama2, Llama2Chat  # noqa: F401
from .mb_internal import *  # noqa: F401, F403
from .minimax_api import MiniMax, MiniMaxChatCompletionV2  # noqa: F401
from .mistral_api import Mistral  # noqa: F401
from .mixtral import Mixtral  # noqa: F401
from .modelscope import ModelScope, ModelScopeCausalLM  # noqa: F401
from .moonshot_api import MoonShot  # noqa: F401
from .nanbeige_api import Nanbeige  # noqa: F401
from .openai_api import OpenAI  # noqa: F401
from .openai_api import OpenAISDK  # noqa: F401
from .openai_streaming import OpenAISDKStreaming  # noqa: F401
from .pangu_api import PanGu  # noqa: F401
from .qwen3_vllm import Qwen3VLLM  # noqa: F401
from .qwen_api import Qwen  # noqa: F401
from .rendu_api import Rendu  # noqa: F401
from .sensetime_api import SenseTime  # noqa: F401
from .stepfun_api import StepFun  # noqa: F401
from .turbomind import TurboMindModel  # noqa: F401
from .turbomind_with_tf_above_v4_33 import (  # noqa: F401
    TurboMindModelwithChatTemplate,
)
from .unigpt_api import UniGPT  # noqa: F401
from .vllm import VLLM  # noqa: F401
from .vllm_with_tf_above_v4_33 import VLLMwithChatTemplate  # noqa: F401
from .xunfei_api import XunFei, XunFeiSpark  # noqa: F401
from .yayi_api import Yayi  # noqa: F401
from .yi_api import YiAPI  # noqa: F401
from .zhipuai_api import ZhiPuAI  # noqa: F401
from .zhipuai_v2_api import ZhiPuV2AI  # noqa: F401

# ModelBest Internal Model Configs
# from .qwen2_5_omni import Qwen2_5OmniModelHF  # noqa: F401
# from .cpm_offline import CPMOffline  # noqa: F401
# from .vllm_offline import VllmOffline  # noqa: F401
# from .minicpmv_chat import MinicpmVChat  # noqa: F401
# from .llmcenter_api import LLMCenter  # noqa: F401
# from .tianshu_api import TianShuAPI  # noqa: F401
# from .luca_llmcenter_api import LucaLLMCenterAPI  # noqa: F401
# from .qwen2_audio_model import Qwen2AudioModel  # noqa: F401
# from .nvidia_llama import NvidiaLlama  # noqa: F401
# from .llamacpp import LlamaCpp  # noqa: F401
# from .llamacpp_deepseek import LlamaCppDeepSeek  # noqa: F401
# from .rwkv import RWKV
# from .modelbest_api import ModelBestApi  # noqa: F401
# from .gpt4o_realtime_api import GPT4RealTime  # noqa: F401
# from .mini_omni_hf import Omni  # noqa: F401
# from .speechgpt_hf import SpeechGPT  # noqa: F401
# from .minicpm3o import Minicpm3o  # noqa: F401
# from .modelbest_api import ModelBestApi  # noqa: F401
# from .siliconflow_api import SiliconFlowAPI  # noqa: F401
# from .ark_api import ArkAPI  # noqa: F401
# from .baiduyun_api import BaiduyunAPI  # noqa: F401
# from .rwkv_world import RWKV_World  # noqa: F401
# from .llama4 import Llama4BaseModel, Llama4withChatTemplate  # noqa: F401
# from .internal.ola_model import OlaModel  # noqa: F401
# from .internal.minicpm_o_model import MiniCPMOModel  # noqa: F401
# from .huggingface_above_v4_33_openlm import \
#     HuggingFaceBaseModelOpenlm  # noqa: F401
# from .gemma3_vllm import Gemma3 # noqa: F401
