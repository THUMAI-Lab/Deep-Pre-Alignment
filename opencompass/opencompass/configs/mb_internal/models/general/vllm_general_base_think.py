import os
from opencompass.models import VLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLM,
        abbr=os.path.basename(os.environ['LOCAL_PATH']) + '-general-base-think-vllm',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.8),
        max_out_len=int(os.environ.get('MAX_OUT_LEN', 65536)),
        max_seq_len=65536,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
