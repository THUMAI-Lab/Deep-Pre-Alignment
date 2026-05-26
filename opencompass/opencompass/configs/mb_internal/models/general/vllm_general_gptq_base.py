import os
from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-gptq-base-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            quantization='gptq',
            dtype='float16', 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=4096,
            trust_remote_code=True,
            ),
        max_out_len=1024,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        # batch_size=4,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
