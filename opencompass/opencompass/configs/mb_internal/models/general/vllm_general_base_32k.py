import os
from opencompass.models import VLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLM,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-base-vllm-32k",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            # dtype='bfloat16',
            tensor_parallel_size=2, 
            gpu_memory_utilization=0.9, 
            max_model_len=65536,
            # rope_scaling={'factor': 8.0, 'rope_type': 'linear'},
        ),
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=65536,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
        mode='mid',
        # stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
    )
]
