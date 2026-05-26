import os
from opencompass.models import VLLMwithChatTemplate
# from opencompass.utils.text_postprocessors import extract_non_reasoning_content_v2  

# os.environ['BATCH_SIZE'] = '1'
# model meta template
_meta_template = dict(  
            round=[
                    dict(role='SYSTEM', api_role='SYSTEM', begin='/no_think'),  # begin and end can be a list of strings or integers.
                    dict(role='HUMAN', api_role='HUMAN'),  # begin and end can be a list of strings or integers.
                    dict(role='BOT', api_role='BOT', generate=True),
            ],
            # reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM', begin='/no_think', end='\n'),],
         )

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-nemotron-nothink-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.8,
            mamba_ssm_cache_dtype='float32',
            enforce_eager=True
            ),
        max_out_len=65536,
        meta_template=_meta_template,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        # max_seq_len=65536,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
        # pred_postprocessor=dict(type=extract_non_reasoning_content_v2)
    )
]
