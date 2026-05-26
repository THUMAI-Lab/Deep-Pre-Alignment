import os
from opencompass.models import OpenAI
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        type=OpenAI,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-general-sft-oneapi",
        path=os.environ['LOCAL_PATH'],
        key='ENV',
        meta_template=api_meta_template,
        query_per_second=1,
        batch_size=1,
        temperature=float(os.environ.get('TEMPERATURE', 0.6)),
        max_seq_len=int(os.environ.get('MAX_SEQ_LEN', 65536)),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
