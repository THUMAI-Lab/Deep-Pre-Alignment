import os
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content_v2  

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-custom-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.8),
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=65536,
        generation_kwargs=dict(
            temperature=float(os.environ.get('TEMPERATURE', 0)),
            top_p=float(os.environ.get('TOP_P', 1)),
            top_k=int(os.environ.get('TOP_K', 20)),
            min_p=float(os.environ.get('MIN_P', 0)),
            presence_penalty=float(os.environ.get('PRESENCE_PENALTY', 0)),
            frequency_penalty=float(os.environ.get('FREQUENCY_PENALTY', 0)),
            repetition_penalty=float(os.environ.get('REPETITION_PENALTY', 1)),
            ),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content_v2)
    )
]
