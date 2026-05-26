from opencompass.models import VLLMwithChatTemplate
import os

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2-1.5b-instruct-vllm',
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=2048,
        batch_size=16,
        generation_kwargs=dict(temperature=0.01,top_p=0.001,top_k=1),
        run_cfg=dict(num_gpus=1),
    )
]
