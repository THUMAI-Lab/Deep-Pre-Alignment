import os
from opencompass.models import Llama4BaseModel
import torch

models = [
    dict(
        type=Llama4BaseModel,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-llama4-base-hf-4gpu",
        model_kwargs=dict(
            attn_implementation='eager',
        ),
        path=os.environ['LOCAL_PATH'],
        use_fastchat_template=False,
        # max_seq_len=16384,
        max_out_len=4096,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        run_cfg=dict(num_gpus=4),
    )
]
