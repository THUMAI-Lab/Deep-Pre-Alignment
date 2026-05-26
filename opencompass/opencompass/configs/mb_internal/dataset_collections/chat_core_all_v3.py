from mmengine.config import read_base

with read_base():
    # mmlu
    from ..datasets.chat_core_v3.mmlu_gen_4d595a import mmlu_datasets
    from ..datasets.chat_core_v3.mmlu_zero_shot_cot_gen_len512_47e2c0 import mmlu_cot_datasets

    # cmmlu
    from ..datasets.chat_core_v3.cmmlu_gen_c13365 import cmmlu_datasets
    from ..datasets.chat_core_v3.cmmlu_0shot_cot_gen_len512_305931 import cmmlu_cot_datasets

    # ceval
    from ..datasets.chat_core_v3.ceval_gen_5f30c7 import ceval_datasets
    from ..datasets.chat_core_v3.ceval_zero_shot_gen_len512_bd40ef import ceval_cot_datasets

    # arc
    from ..datasets.chat_core_v3.ARC_e_gen_1e0de5 import ARC_e_datasets
    from ..datasets.chat_core_v3.ARC_c_gen_1e0de5 import ARC_c_datasets

    # bbh
    from ..datasets.chat_core_v3.bbh_gen_5b92b0 import bbh_datasets

    # gpqa
    from ..datasets.chat_core_v3.gpqa_gen_4baadb import gpqa_datasets

    # math
    # from ..datasets.math.math_0shot_gen_393424 import math_datasets
    # math 500
    from ..datasets.chat_core_v3.math_prm800k_500_0shot_cot_gen_11c4b5 import math_datasets

    # gsm8k
    from ..datasets.chat_core_v3.gsm8k_gen_1d7fe4 import gsm8k_datasets

    # sanitized mbpp
    from ..datasets.chat_core_v3.sanitized_mbpp_mdblock_gen_a447ff import sanitized_mbpp_datasets

    # humaneval
    from ..datasets.chat_core_v3.humaneval_gen_8e312c import humaneval_datasets

    # mb-gaokao
    from ..datasets.chat_core_v3.mb_gaokao_gen_0shot_20241122 import mb_gaokao_0shot_datasets
    from ..datasets.chat_core_v3.mb_gaokao_gen_5shot_20241122 import mb_gaokao_5shot_datasets

    # livecodebench
    from ..datasets.chat_core_v3.livecodebench_o1_gen_f0ed6c import LCB_datasets

    # aime2024
    from ..datasets.chat_core_v3.aime2024_gen_6e39a4 import aime2024_datasets


HF_INFER_DATASET_NAMES = [
    # 'mmlu_datasets',
]

# base datasets
base_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k not in HF_INFER_DATASET_NAMES), [])
# mmlu may got OOM error when using vllm, so we need to split it and use HF Wrapper
hf_infer_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets') and k in HF_INFER_DATASET_NAMES), [])
