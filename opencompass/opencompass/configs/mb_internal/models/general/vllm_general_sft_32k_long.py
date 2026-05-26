import os
from opencompass.models import VLLMwithChatTemplate
# from opencompass.utils.text_postprocessors import extract_non_reasoning_content_v2  

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-sft-vllm-32k",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            # dtype='bfloat16',
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            max_model_len=65536
        ),
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 1)),
        max_seq_len=65536,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
        # pred_postprocessor=dict(type=extract_non_reasoning_content_v2)
    )
]
