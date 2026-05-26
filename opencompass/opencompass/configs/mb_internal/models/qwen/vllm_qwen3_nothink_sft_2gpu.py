import os
from opencompass.models import Qwen3VLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=Qwen3VLLM,
        abbr=os.path.basename(os.environ['LOCAL_PATH']) + '-qwen3-nothink-sft-vllm-2gpu',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=2),
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
        ),  # qwen3 官方对非深思模式推荐配置
        run_cfg=dict(num_gpus=2),
        enable_thinking=False,
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
