import os
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-pangu-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            tensor_parallel_size=2, 
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        ),
        max_out_len=65536,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=131072,
        generation_kwargs=dict(
            temperature=0.7,
            top_p=1,
            ),
        run_cfg=dict(num_gpus=2),
        pred_postprocessor=dict(
            type=extract_non_reasoning_content,
            think_start_token='[unused16]',
            think_end_token='[unused17]'
            )
    )
]
