import os
from opencompass.models.mb_internal.megrez_vllm import MegrezVLLM

models = [
    dict(
        type=MegrezVLLM,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-megrez-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.8),
        max_out_len=32768,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=65536,
        generation_kwargs=dict(temperature=0.7, top_p=0.9),
        run_cfg=dict(num_gpus=1),
        # pred_postprocessor=dict(type=extract_non_reasoning_content_v2)
    )
]
