import os
from opencompass.models import Gemma3nBase


models = [
    dict(
        type=Gemma3nBase,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-gemma3n-base-hf",
        path=os.environ['LOCAL_PATH'],
        torch_dtype='bfloat16',
        model_kwargs=dict(),
        max_out_len=16384,
        max_seq_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
