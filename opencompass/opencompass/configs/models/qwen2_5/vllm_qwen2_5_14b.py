from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='qwen2.5-14b-vllm',
        path='Qwen/Qwen2.5-14B',
        # model_kwargs=dict(tensor_parallel_size=2),
        # max_out_len=4096,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]