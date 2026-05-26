mb_gaokao2025_summary_groups = []

_mb_gaokao2025_all = [
        'mb_gaokao2025-生物_选择题_mocks-0shot',
        'mb_gaokao2025-政治_选择题_mocks-0shot',
        'mb_gaokao2025-物理_选择题_mocks-0shot',
        'mb_gaokao2025-历史_选择题_mocks-0shot',
        'mb_gaokao2025-数学_选择题_mocks-0shot',
    ]

_mb_gaokao2025_weights_0shot = {'mb_gaokao2025-历史_选择题_mocks-0shot': 15, 'mb_gaokao2025-生物_选择题_mocks-0shot': 26, 'mb_gaokao2025-数学_选择题_mocks-0shot': 10, 'mb_gaokao2025-物理_选择题_mocks-0shot': 3, 'mb_gaokao2025-政治_选择题_mocks-0shot': 4}

mb_gaokao2025_summary_groups.append({'name': 'mb_gaokao2025-0shot', 'subsets': [c for c in _mb_gaokao2025_all]})

mb_gaokao2025_summary_groups.append({'name': 'mb_gaokao2025-0shot-weighted', 'subsets': list(_mb_gaokao2025_weights_0shot.keys()), 'weights': _mb_gaokao2025_weights_0shot})

# # 新增的标准差groups
# mb_gaokao2025_summary_groups.append({
#     'name': 'mb_gaokao2025-0shot-std',
#     'subsets': [c for c in _mb_gaokao2025_all],
#     'std': True
# })


