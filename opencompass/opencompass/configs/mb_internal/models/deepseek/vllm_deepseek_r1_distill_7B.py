import os
from opencompass.models import DeepSeekR1Distill


models = [
    dict(
        type=DeepSeekR1Distill,
        # abbr=os.path.basename(os.environ['LOCAL_PATH']) + "-vllm",
        abbr='DeepSeek-R1-Distill-Qwen-7B-vllm',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        # path=os.environ['LOCAL_PATH'],
        # model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.8),
        max_out_len=32768,
        batch_size=1,
        # # generation_kwargs=dict(temperature=0),
        # generation_kwargs=dict(temperature=0.6, top_p=0.95),
        run_cfg=dict(num_gpus=1),
    )
]
