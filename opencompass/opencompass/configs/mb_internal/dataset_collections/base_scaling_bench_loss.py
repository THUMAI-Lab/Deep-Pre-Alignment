from mmengine.config import read_base

with read_base():
    # scaling bench
    from ..datasets.scaling_bench.scaling_bench_all import scaling_bench_datasets

base_datasets_total = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
