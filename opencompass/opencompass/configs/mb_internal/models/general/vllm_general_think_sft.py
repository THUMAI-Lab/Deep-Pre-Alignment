import os
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content


models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-think-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.9),
        max_out_len=65536,
        max_seq_len=65536,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            # top_k=20,
            # min_p=0,
            # presence_penalty=0.2
        ),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
