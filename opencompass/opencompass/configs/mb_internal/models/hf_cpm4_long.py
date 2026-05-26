import os
from opencompass.models import CPMModel

models = [
    dict(
        type=CPMModel,
        abbr=os.path.basename(os.environ['LOCAL_PATH']) + '-cpm-hf',
        path=os.environ['LOCAL_PATH'],
        max_out_len=4096,
        max_seq_len=128*1024,
        torch_dtype='bfloat16',
        generation_kwargs=dict(logits_to_keep=1,do_sample=False),
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

