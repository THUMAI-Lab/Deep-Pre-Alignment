from mmengine.config import read_base

with read_base():
    # mmlu
    from ...datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from ..datasets.mmlu.mmlu_zero_shot_cot_gen_len512_47e2c0 import mmlu_cot_datasets

    # cmmlu
    from ...datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets

    # ceval
    from ...datasets.ceval.ceval_gen_5f30c7 import ceval_datasets

    # arc
    from ...datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    from ...datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets

    # bbh
    from ...datasets.bbh.bbh_gen_5b92b0 import bbh_datasets

    # gpqa
    from ...datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets

    # math
    from ...datasets.math.math_0shot_gen_393424 import math_datasets

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
