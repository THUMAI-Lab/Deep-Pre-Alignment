from opencompass.models import Qwen2AudioModel

models = [
    dict(
        type=Qwen2AudioModel,
        abbr='qwen2_audio_7b_instruct-hf',
        path='Qwen/Qwen2-7B-Instruct',
        max_out_len=1024,
        # max_seq_len=8192,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
