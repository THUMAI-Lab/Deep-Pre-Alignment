from mmengine.config import read_base

with read_base():
    from .bigcodebench_full_complete_gen_2888d3 import bigcodebench_full_complete_datasets
    from .bigcodebench_full_instruct_gen_c3d5ad import bigcodebench_full_instruct_datasets

bigcodebench_full_internal_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])