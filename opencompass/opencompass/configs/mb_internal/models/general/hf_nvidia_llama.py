from opencompass.models import NvidiaLlama
import os

models = [
    dict(
        type=NvidiaLlama,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-nvidia-base-hf",
        path=os.environ['LOCAL_PATH'],
        max_out_len=1024,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        # generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
