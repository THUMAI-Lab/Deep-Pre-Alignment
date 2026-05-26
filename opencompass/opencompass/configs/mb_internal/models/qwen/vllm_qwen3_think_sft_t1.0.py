import os
from opencompass.models import Qwen3VLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=Qwen3VLLM,
        abbr=os.path.basename(os.environ['LOCAL_PATH']) + '-qwen3-think-sft-vllm-t1.0',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.9),
        max_out_len=65536,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(
            temperature=os.environ.get('TEMPERATURE', 1.0),
            top_p=0.95,
            top_k=20,
            min_p=0,
            # presence_penalty=0.2,
        ),  # qwen3 官方对深思模式推荐配置
        run_cfg=dict(num_gpus=1),
        enable_thinking=True,
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
