import os
from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-base-vllm-4gpu",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=4, gpu_memory_utilization=0.8),
        max_out_len=1024,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
    )
]
