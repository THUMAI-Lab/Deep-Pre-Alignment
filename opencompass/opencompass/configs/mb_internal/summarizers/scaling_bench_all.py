
from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.mmlu import mmlu_summary_groups
    from ...summarizers.groups.cmmlu import cmmlu_summary_groups
    from ...summarizers.groups.ceval import ceval_summary_groups
    from ...summarizers.groups.bbh import bbh_summary_groups
    from ..summarizers.groups.mb_gaokao import mb_gaokao_summary_groups
    from .groups.scaling_bench import scaling_bench_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # mmlu
        ['mmlu', 'naive_average'],

        # cmmlu
        ['cmmlu', 'naive_average'],

        # ceval
        ['ceval', 'naive_average'],

        # bbh
        ['bbh', 'naive_average'],

        # math
        ['math', 'accuracy'],

        # sanitized mbpp
        ['sanitized_mbpp', 'score'],

        # humaneval
        ['openai_humaneval', 'humaneval_pass@1'],

        # mb-gaokao
        'mb_gaokao-0shot-weighted',

        # scaling bench
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
