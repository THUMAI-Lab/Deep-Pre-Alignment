import os
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-sft-hf-t0.2",
        path=os.environ['LOCAL_PATH'],
        max_out_len=16384,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=32768,
        generation_kwargs=dict(temperature=0.2),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
