import os
from opencompass.models import VLLMSpeedwithChatTemplate

models = [
    dict(
        type=VLLMSpeedwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-speed-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            dtype='bfloat16',
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.8
            ),
        max_out_len=16384,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=32768,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]