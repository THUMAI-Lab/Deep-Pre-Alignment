from opencompass.models import VLLM

import os

models = [
    dict(
        type=VLLM,
        abbr='qwen2.5-7b-vllm',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
