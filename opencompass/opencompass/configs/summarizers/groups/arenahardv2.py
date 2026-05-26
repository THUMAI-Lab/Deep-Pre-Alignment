arenahardv2_summary_groups = []

_arenahardv2_all = {
    # 'hard_prompt',
    'arena_hard_v2-coding': 253,
    'arena_hard_v2-math': 247,
    'arena_hard_v2-creative_writing': 250,
}

arenahardv2_summary_groups.append(
    {'name': 'arenahardv2-hard_prompt', 'subsets': [c for c in ['arena_hard_v2-coding', 'arena_hard_v2-math']]})

arenahardv2_summary_groups.append(
    {'name': 'arenahardv2-creative_writing', 'subsets': [c for c in ['arena_hard_v2-creative_writing']]})

arenahardv2_summary_groups.append(
    {'name': 'arenahardv2', 'subsets': [c for c in _arenahardv2_all]})

arenahardv2_summary_groups.append(
    {'name': 'arenahardv2-weighted', 'subsets': [c for c in _arenahardv2_all], 'weights': _arenahardv2_all})
