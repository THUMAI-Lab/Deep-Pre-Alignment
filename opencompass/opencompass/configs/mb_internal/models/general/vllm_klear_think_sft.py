import os
from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=f"{os.path.basename(os.path.normpath(os.environ['LOCAL_PATH']))}-klear-think-sft-vllm",
        path=os.environ['LOCAL_PATH'],
        model_kwargs=dict(
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            max_num_seqs=128,
            hf_overrides={
                'rope_scaling': {
                    'rope_type': 'yarn', 
                    'factor': 2.5, 
                    'original_max_position_embeddings': 32768
                    },
                'max_model_len': 81920
            },
            enforce_eager=False,
            seed=0,
            ),
        max_out_len=65536,
        batch_size=int(os.environ.get('BATCH_SIZE', 32)),
        # max_seq_len=65536,
        generation_kwargs=dict(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.05
            ),
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
