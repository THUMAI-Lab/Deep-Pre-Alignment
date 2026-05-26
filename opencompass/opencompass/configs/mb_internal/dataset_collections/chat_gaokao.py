from mmengine.config import read_base

with read_base():
    # gaokao
    from ..datasets.mb_gaokao.mb_gaokao_gen_0shot_20241122 import mb_gaokao_datasets

HF_INFER_DATASET_NAMES = [
    # 'mmlu_datasets',
]

# base datasets
base_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k not in HF_INFER_DATASET_NAMES), [])
# mmlu may got OOM error when using vllm, so we need to split it and use HF Wrapper
hf_infer_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in HF_INFER_DATASET_NAMES), [])
