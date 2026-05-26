from mmengine.config import read_base

from opencompass.models import HuggingFacewithChatTemplateInternal, VLLM

with read_base():
    from ..cluster.local_cluster import eval, infer_num_worker as infer
    from ..summarizers.mb_example import summarizer


_meta_template = dict(
    begin='<|im_start|>user\n',
    end='<|im_end|>\n<|im_start|>assistant\n',
    round=[
          dict(role='HUMAN', begin='', end=''),
          dict(role='BOT', begin='\n', end='\n'),
    ],
)


internal_meta_template = dict(
    round=[
          dict(role='HUMAN', begin='', end=''),
          dict(role='BOT', begin='\n', end='\n'),
    ],
)


hf_model = dict(
    type=HuggingFacewithChatTemplateInternal,
    abbr=None,
    path=None,
    meta_template=internal_meta_template,
    max_out_len=1024,
    batch_size=8,
    max_seq_len=4096,
    run_cfg=dict(num_gpus=1),
    model_kwargs=dict(
        trust_remote_code=True,
    ),
    tokenizer_kwargs=dict(
        trust_remote_code=True,
    ),
)


vllm_model = dict(
    type=VLLM,
    abbr=None,
    path=None,
    model_kwargs=dict(tensor_parallel_size=1),
    max_out_len=1024,
    max_seq_len=4096,
    batch_size=1,
    generation_kwargs=dict(temperature=0),
    run_cfg=dict(num_gpus=1),
    meta_template=_meta_template,
)
