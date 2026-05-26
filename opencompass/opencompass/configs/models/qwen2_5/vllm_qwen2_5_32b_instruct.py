from opencompass.models import VLLMwithChatTemplate

import os

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-32b-instruct-vllm',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=2),
        max_out_len=4096,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
    )
]
