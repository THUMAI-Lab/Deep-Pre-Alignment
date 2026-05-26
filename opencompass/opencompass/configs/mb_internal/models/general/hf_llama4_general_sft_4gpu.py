import os
from opencompass.models import Llama4withChatTemplate
import torch

models = [
    dict(
        type=Llama4withChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-llama4-sft-hf-4gpu",
        model_kwargs=dict(
            attn_implementation='eager',
        ),
        path=os.environ['LOCAL_PATH'],
        use_fastchat_template=True,
        # max_seq_len=16384,
        max_out_len=4096,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        run_cfg=dict(num_gpus=4),
    )
]
