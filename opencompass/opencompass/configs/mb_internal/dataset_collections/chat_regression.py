from mmengine.config import read_base

with read_base():
    # mmlu
    from ...datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets

    # cmmlu
    from ...datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets

    # arc
    from ...datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets

    # gsm8k
    from ...datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

    # sanitized mbpp
    from ...datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import sanitized_mbpp_datasets

    # humaneval
    from ...datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets

HF_INFER_DATASET_NAMES = [
    # 'mmlu_datasets',
]

# base datasets
base_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k not in HF_INFER_DATASET_NAMES), [])
# mmlu may got OOM error when using vllm, so we need to split it and use HF Wrapper
hf_infer_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in HF_INFER_DATASET_NAMES), [])
