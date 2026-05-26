import os
from opencompass.models import Qwen2_5OmniModelHF

models = [
    dict(
        type=Qwen2_5OmniModelHF,
        abbr='qwen2.5-omni-7b',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=4096,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
