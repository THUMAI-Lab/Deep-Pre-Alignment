from mmengine.config import read_base

with read_base():
    # mmlu
    from ...datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets

    # cmmlu
    from ...datasets.cmmlu.cmmlu_ppl_041cbf import cmmlu_datasets

    # ceval
    from ...datasets.ceval.ceval_internal_ppl_93e5ce import ceval_datasets

    # arc
    from ...datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from ...datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets

    # bbh
    # different with v3
    from ...datasets.bbh.bbh_gen_98fba6 import bbh_datasets

    # gpqa
    from ...datasets.gpqa.gpqa_ppl_6bf57a import gpqa_datasets

    # math
    from ...datasets.math.math_4shot_base_gen_db136b import math_datasets

    # gsm8k
    from ...datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets

    # sanitized mbpp
    from ...datasets.mbpp.sanitized_mbpp_gen_742f0c import sanitized_mbpp_datasets

    # humaneval
    from ...datasets.humaneval.deprecated_humaneval_gen_d2537e import humaneval_datasets

HF_INFER_DATASET_NAMES = [
    'mmlu_datasets',
]

# base datasets
base_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k not in HF_INFER_DATASET_NAMES), [])
# mmlu may got OOM error when using vllm, so we need to split it and use HF Wrapper
hf_infer_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in HF_INFER_DATASET_NAMES), [])
