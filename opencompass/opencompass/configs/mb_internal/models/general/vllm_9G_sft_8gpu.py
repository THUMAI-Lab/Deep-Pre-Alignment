import os
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-9G-sft-vllm-8gpu",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=8, gpu_memory_utilization=0.8),
        max_out_len=1024,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        # max_seq_len=32768,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=8),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
