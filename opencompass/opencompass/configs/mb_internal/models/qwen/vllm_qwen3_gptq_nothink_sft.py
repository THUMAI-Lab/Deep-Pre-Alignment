import os
from opencompass.models import Qwen3VLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=Qwen3VLLM,
        abbr=os.path.basename(os.environ['LOCAL_PATH']) + '-gptq-qwen3-nothink-sft-vllm',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            quantization='gptq',
        ),
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(
            temperature=os.environ.get('TEMPERATURE', 0.7),
            top_p=0.8,
            top_k=20,
            min_p=0,
        ),  # qwen3 官方对非深思模式推荐配置
        run_cfg=dict(num_gpus=1),
        enable_thinking=False,
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
