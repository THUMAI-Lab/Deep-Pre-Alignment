import os
from opencompass.models import HuggingFaceBaseModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-base-hf-32k",
        path=os.environ['LOCAL_PATH'],
        max_out_len=256,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=31500,
        # generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
        mode='mid',
        # stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
    )
]
