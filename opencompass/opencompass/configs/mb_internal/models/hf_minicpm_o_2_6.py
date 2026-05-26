import os
from opencompass.models import MiniCPMOModel

models = [
    dict(
        type=MiniCPMOModel,
        abbr=os.path.basename(os.environ['LOCAL_PATH']) + '-sft-hf',
        model_kwargs=dict(),
        path=os.environ['LOCAL_PATH'],
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
