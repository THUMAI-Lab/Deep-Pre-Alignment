scaling_bench_summary_groups = []

_scaling_bench_all_sets = [
    'mmlu',
    'cmmlu',
    'ceval',
    'bbh',
    'math',
    'mbpp',
    'human_eval',
    'gaokao2024',
]
_scaling_bench_all = [f'scaling_bench_{c}' for c in _scaling_bench_all_sets]

scaling_bench_summary_groups.append({'name': 'scaling_bench', 'subsets': _scaling_bench_all})
