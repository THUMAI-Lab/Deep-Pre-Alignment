from mmengine.config import read_base

with read_base():
    # mmlu
    from ...datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets

    # cmmlu
    from ...datasets.cmmlu.cmmlu_ppl_041cbf import cmmlu_datasets

    # ceval
    from ...datasets.ceval.ceval_internal_ppl_93e5ce import ceval_datasets

    # bbh
    from ...datasets.bbh.bbh_gen_98fba6 import bbh_datasets

    # math
    from ...datasets.math.math_4shot_base_gen_db136b import math_datasets

    # sanitized mbpp
    from ...datasets.mbpp.sanitized_mbpp_gen_742f0c import sanitized_mbpp_datasets

    # humaneval
    from ...datasets.humaneval.deprecated_humaneval_gen_d2537e import humaneval_datasets

    # mb-gaokao2024
    from ..datasets.mb_gaokao.mb_gaokao_ppl_0shot_20241122 import mb_gaokao_0shot_datasets

    # scaling bench
    from ..datasets.scaling_bench.scaling_bench_all import scaling_bench_datasets

SCLAING_DATASET_NAMES = [
    'scaling_bench_datasets',
]

# base datasets
base_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k not in SCLAING_DATASET_NAMES), [])
# mmlu may got OOM error when using vllm, so we need to split it and use HF Wrapper
scaling_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in SCLAING_DATASET_NAMES), [])
