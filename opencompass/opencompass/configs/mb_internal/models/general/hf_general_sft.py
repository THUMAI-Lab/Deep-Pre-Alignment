import os
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-sft-hf",
        path=os.environ['LOCAL_PATH'],
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=65536,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1)
    )
]
