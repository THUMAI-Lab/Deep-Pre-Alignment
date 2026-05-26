import os
from opencompass.models import Qwen3VLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=Qwen3VLLM,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-qwen3-nothink-long-mid",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            tensor_parallel_size=1, gpu_memory_utilization=0.9,
            rope_scaling={
                'rope_type': 'yarn',
                'factor': 4.0,
                'original_max_position_embeddings': 32768
            },
            max_model_len=131072
            ),
        max_out_len=4096,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=131072,
        generation_kwargs=dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0),
        run_cfg=dict(num_gpus=1),
        enable_thinking=False,
        mode='mid',
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
