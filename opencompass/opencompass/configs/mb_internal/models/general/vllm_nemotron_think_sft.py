import os
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-nemotron-think-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.8,
            mamba_ssm_cache_dtype='float32',
            max_num_seqs=64,
            max_model_len=131072,
            enforce_eager=True,
            ),
        max_out_len=81920,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        # max_seq_len=65536,
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            ),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
