import os
from opencompass.models import VLLMwithChatTemplateLongTruncated
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=VLLMwithChatTemplateLongTruncated,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-sft-vllm-32k-8gpu",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=8, gpu_memory_utilization=0.8, max_model_len=32768),
        max_out_len=256,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=31500,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=8),
        mode='mid',
        enable_thinking=False,
        # stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
    )
]
