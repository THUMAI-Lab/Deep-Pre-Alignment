import os
from opencompass.models import DeepSeekR1Distill
from opencompass.utils.text_postprocessors import extract_non_reasoning_content


models = [
    dict(
        type=DeepSeekR1Distill,
        abbr=os.path.basename(os.environ['LOCAL_PATH']) + '-think-vllm',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.8),
        enable_thinking=True,
        max_out_len=32768,
        batch_size=1,
        generation_kwargs=dict(temperature=0.6, top_p=0.95),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
