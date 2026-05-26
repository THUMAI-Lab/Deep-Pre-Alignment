import os
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-qwen3-sft-hf-2gpu",
        path=os.environ['LOCAL_PATH'],
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=65536,
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0,
        ),
        run_cfg=dict(num_gpus=2),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
