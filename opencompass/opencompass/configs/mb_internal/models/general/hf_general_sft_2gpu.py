import os
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-sft-hf-2gpu",
        path=os.environ['LOCAL_PATH'],
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        run_cfg=dict(num_gpus=8),
    )
]
