from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from .datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from .datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from .datasets.humaneval.humaneval_gen_a82cae import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.math.math_gen_265cce import math_datasets
    from .models.openbmb.hf_minicpm_2b_sft_bf16 import models

# _base_ = [
#     'secrets.py',
# ]
  
# datasets = [*siqa_datasets, *winograd_datasets]
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
# models = [hf_minicpm_2b_sft_bf16]
# models = [vllm_minicpm_2b_sft_bf16]
