import os
from opencompass.models import Qwen3VLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=Qwen3VLLM,
        abbr=os.path.basename(os.environ['LOCAL_PATH']) + '-qwen3_vl-nothink-sft-vllm',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, limit_mm_per_prompt=dict(image=0, video=0)),
        max_out_len=32768,
        max_seq_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(
            temperature=1.0,
            top_p=1.0,
            top_k=40,
        ),  # qwen3-vl text 官方对非深思模式推荐配置
        run_cfg=dict(num_gpus=1),
        enable_thinking=False,
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
