from mmengine.config import read_base

with read_base():
    # mmlu
    from ...datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets

    # commonsenseqa
    # from ...datasets.commonsenseqa.commonsenseqa_ppl_5545e2 import commonsenseqa_datasets
    from ...datasets.commonsenseqa.commonsenseqa_ppl_e51e32 import commonsenseqa_datasets

    # hellaswag
    from ...datasets.hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets

    # arc
    from ...datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from ...datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets

    # piqa
    from ...datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets

    # siqa
    from ...datasets.siqa.siqa_ppl_ced5f6 import siqa_datasets

    # winogrande
    from ...datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets

    # openbook qa
    from ...datasets.obqa.obqa_ppl_c7c154 import obqa_datasets

    # bbh
    from ...datasets.bbh.bbh_gen_5b92b0 import bbh_datasets

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
