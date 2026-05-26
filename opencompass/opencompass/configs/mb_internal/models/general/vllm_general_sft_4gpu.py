import os
from opencompass.models import VLLMwithChatTemplate


models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=4, gpu_memory_utilization=0.8),
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=65536,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
    )
]
