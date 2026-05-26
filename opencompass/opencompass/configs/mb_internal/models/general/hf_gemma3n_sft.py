import os
from opencompass.models import Gemma3n
# from opencompass.models import VLLMwithChatTemplate


models = [
    dict(
        # type=VLLMwithChatTemplate,
        type=Gemma3n,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-gemma3n-sft-hf",
        path=os.environ['LOCAL_PATH'],
        torch_dtype='bfloat16',
        model_kwargs=dict(
            # dtype='bfloat16',
            # tensor_parallel_size=1, gpu_memory_utilization=0.75,
            # for long context
            # rope_scaling={'factor': 8.0, 'rope_type': 'linear'}
            ),
        max_out_len=16384,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        max_seq_len=32768,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
        stop_words=['<end_of_turn>'],
    )
]
