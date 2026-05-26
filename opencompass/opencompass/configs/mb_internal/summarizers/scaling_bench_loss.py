
from mmengine.config import read_base

with read_base():
    from .groups.scaling_bench import scaling_bench_summary_groups

summarizer = dict(
    dataset_abbrs=[
        ['scaling_bench', 'naive_average'],
        # details
        ['scaling_bench_mmlu', 'average_ppl'],
        ['scaling_bench_cmmlu', 'average_ppl'],
        ['scaling_bench_ceval', 'average_ppl'],
        ['scaling_bench_bbh', 'average_ppl'],
        ['scaling_bench_math', 'average_ppl'],
        ['scaling_bench_mbpp', 'average_ppl'],
        ['scaling_bench_human_eval', 'average_ppl'],
        ['scaling_bench_gaokao2024', 'average_ppl'],

    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
