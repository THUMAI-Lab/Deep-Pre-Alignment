import os
# from opencompass.models import VLLMwithChatTemplate
from opencompass.models import Qwen3VLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content_hunyuan 

models = [
    dict(
        type=Qwen3VLLM,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-hunyuan-think-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.8),
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_out_len=32768,
        max_seq_len=65536,
        enable_thinking=True,
        generation_kwargs=dict(
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.05,
            temperature=0.7,
            stop_token_ids=[127960]
            ),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content_hunyuan)
    )
]
