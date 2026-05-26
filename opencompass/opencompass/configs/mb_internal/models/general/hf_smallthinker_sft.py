import os
from opencompass.models import HuggingFacewithChatTemplate
import torch

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-smallthinker-sft-hf",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            torch_dtype=torch.bfloat16,
            device_map='cuda',
        ),
        max_out_len=16384,
        batch_size=1,
        max_seq_len=32768,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
