from mmengine.config import read_base

with read_base():
    # cmmlu
    from ...datasets.cmmlu.cmmlu_ppl_041cbf import cmmlu_datasets

    # ceval
    from ...datasets.ceval.ceval_internal_ppl_93e5ce import ceval_datasets

    # commonsenseqa cn
    from ...datasets.commonsenseqa_cn.commonsenseqacn_ppl_971f48 import commonsenseqacn_datasets

HF_INFER_DATASET_NAMES = [
    'mmlu_datasets',
]

# base datasets
base_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k not in HF_INFER_DATASET_NAMES), [])
# mmlu may got OOM error when using vllm, so we need to split it and use HF Wrapper
hf_infer_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in HF_INFER_DATASET_NAMES), [])
