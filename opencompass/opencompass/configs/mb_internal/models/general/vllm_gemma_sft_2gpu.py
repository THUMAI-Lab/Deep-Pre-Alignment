import os
# from opencompass.models import Gemma3
from opencompass.models import VLLMwithChatTemplate


models = [
    dict(
        type=VLLMwithChatTemplate,
        # type=Gemma3,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-gemma-sft-vllm-2gpu",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            dtype='bfloat16',
            tensor_parallel_size=2, 
            gpu_memory_utilization=0.9,
            # for long context
            # rope_scaling={'factor': 8.0, 'rope_type': 'linear'}
            ),
        max_out_len=16384,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=32768,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
        stop_words=['<end_of_turn>'],
    )
]
