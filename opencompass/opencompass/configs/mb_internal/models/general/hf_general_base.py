import os
from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-base-hf",
        path=os.environ['LOCAL_PATH'],
        max_out_len=1024,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        generation_kwargs=dict(temperature=0),
        model_kwargs=dict(torch_dtype='auto'),  # 自动从模型配置读取
        run_cfg=dict(num_gpus=1),
    )
]
