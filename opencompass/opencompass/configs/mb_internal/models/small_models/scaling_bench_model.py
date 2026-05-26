
from opencompass.models import HuggingFaceBaseModelInternal

base_scaling_model = dict(
    type=HuggingFaceBaseModelInternal,
    abbr=None,
    path=None,
    # max_out_len=1024,
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
    ),
    model_kwargs=dict(
        # default is float16, which may cause overflow
        torch_dtype='torch.bfloat16',
    ),
    batch_size=1,
    # max_seq_len=4096,
    run_cfg=dict(num_gpus=1),
    # NOTE: answer_token_len is not used in the `get_ppl_internal`
    # answer_token_len=1,
    # user can choose `byte` or `token`
    ppl_type='byte',
)
