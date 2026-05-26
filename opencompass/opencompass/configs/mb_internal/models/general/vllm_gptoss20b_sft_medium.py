import os
from opencompass.models import GPTOssVLLM
from opencompass.utils.text_postprocessors import extract_non_reasoning_content_gptoss

entries_start_token = '##<channel>##'
entries_end_token = '##</channel>##'

models = [
    dict(
        type=GPTOssVLLM,
        abbr='gpt-oss-20b-medium-gptoss-sft-vllm',
        path='openai/gpt-oss-20b',
        model_kwargs=dict(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        ),
        max_out_len=262144,
        max_seq_len=262144,
        batch_size=32,
        generation_kwargs=dict(temperature=1.0, top_p=1.0),
        run_cfg=dict(num_gpus=1),
        reasoning_level='medium',
        entries_start_token=entries_start_token,
        entries_end_token=entries_end_token,
        pred_postprocessor=dict(
            type=extract_non_reasoning_content_gptoss,
            entries_start_token=entries_start_token,
            entries_end_token=entries_end_token,
        ),
    )
]