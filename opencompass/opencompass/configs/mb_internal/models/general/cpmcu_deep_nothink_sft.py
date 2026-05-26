import os
from opencompass.models import CPMCUwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content


models = [
    dict(
        type=CPMCUwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-nothink-sft-cpmcu",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            ),
        max_out_len=65536,
        batch_size=1,
        # max_seq_len=65536,
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            # top_k=20,
            # min_p=0,
            # presence_penalty=0.2,
        ),
        run_cfg=dict(num_gpus=1),
        enable_thinking=False,
        # pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
