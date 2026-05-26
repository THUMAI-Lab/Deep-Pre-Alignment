from opencompass.models import VLLMwithChatTemplate

import os

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-3b-instruct-vllm',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
        max_out_len=2048,
        batch_size=16,
        generation_kwargs=dict(temperature=0.000001,repetition_penalty=1.05),
        run_cfg=dict(num_gpus=1),
    )
]
