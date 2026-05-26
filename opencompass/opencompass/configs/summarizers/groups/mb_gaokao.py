mb_gaokao_summary_groups = []

_mb_gaokao_all = [
        # 'gaokao2024_1122',
        'mb_gaokao-gaokaotagging2024_1123',
        'mb_gaokao-gaokaotagging2024_1126',
        'mb_gaokao-gaokaotagging2024_1127',
        'mb_gaokao-gaokaotagging2024_1128',
        'mb_gaokao-gaokaotagging2024_1129',
    ]

# total
_mb_gaokao_weights_0shot = {'gaokaotagging2024_1123': 239, 'gaokaotagging2024_1126': 458, 'gaokaotagging2024_1127': 379, 'gaokaotagging2024_1128': 253, 'gaokaotagging2024_1129': 451}
# hard split 5 for 5-shot setting
_mb_gaokao_weights_5shot = {'gaokaotagging2024_1123': 234, 'gaokaotagging2024_1126': 453, 'gaokaotagging2024_1127': 374, 'gaokaotagging2024_1128': 249, 'gaokaotagging2024_1129': 446}

_mb_gaokao_weights_0shot = {'mb_gaokao-' + k + '-0shot': v for k, v in _mb_gaokao_weights_0shot.items()}
_mb_gaokao_weights_5shot = {'mb_gaokao-' + k + '-5shot': v for k, v in _mb_gaokao_weights_5shot.items()}

mb_gaokao_summary_groups.append({'name': 'mb_gaokao-0shot', 'subsets': [c + '-0shot' for c in _mb_gaokao_all]})
mb_gaokao_summary_groups.append({'name': 'mb_gaokao-5shot', 'subsets': [c + '-5shot' for c in _mb_gaokao_all]})

mb_gaokao_summary_groups.append({'name': 'mb_gaokao-0shot-weighted', 'subsets': list(_mb_gaokao_weights_0shot.keys()), 'weights': _mb_gaokao_weights_0shot})
mb_gaokao_summary_groups.append({'name': 'mb_gaokao-5shot-weighted', 'subsets': list(_mb_gaokao_weights_5shot.keys()), 'weights': _mb_gaokao_weights_5shot})

# 新增的标准差groups
mb_gaokao_summary_groups.append({
    'name': 'mb_gaokao-0shot-std',
    'subsets': [c + '-0shot' for c in _mb_gaokao_all],
    'std': True
})

mb_gaokao_summary_groups.append({
    'name': 'mb_gaokao-5shot-std',
    'subsets': [c + '-5shot' for c in _mb_gaokao_all],
    'std': True
})
