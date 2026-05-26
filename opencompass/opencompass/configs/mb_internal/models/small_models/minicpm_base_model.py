
from opencompass.models import HuggingFaceBaseModel, VLLM

base_hf_model = dict(
    type=HuggingFaceBaseModel,
    abbr=None,
    path=None,
    max_out_len=1024,
    batch_size=1,
    max_seq_len=4096,
    run_cfg=dict(num_gpus=1),
)


base_vllm_model = dict(
    type=VLLM,
    abbr=None,
    path=None,
    model_kwargs=dict(tensor_parallel_size=1),
    max_out_len=1024,
    max_seq_len=4096,
    batch_size=1,
    generation_kwargs=dict(temperature=0),
    run_cfg=dict(num_gpus=1),
)

base_vllm_model_less_gpu_mem = dict(
    type=VLLM,
    abbr=None,
    path=None,
    model_kwargs=dict(tensor_parallel_size=1, gpu_memory_util=0.85),
    max_out_len=1024,
    max_seq_len=4096,
    batch_size=1,
    generation_kwargs=dict(temperature=0),
    run_cfg=dict(num_gpus=1),
)