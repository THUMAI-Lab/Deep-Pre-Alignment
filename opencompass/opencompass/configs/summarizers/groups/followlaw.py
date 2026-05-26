followlaw_summary_groups = []

_follwlaw_all = [
    'followlaw_1合并上诉状-1',
    'followlaw_2一审民事审理要点-1',
    'followlaw_3争议焦点（带开庭笔录）-1',
    'followlaw_4执行标的分类-0',
    'followlaw_5核心法条适用要点-1',
    'followlaw_6执行驳回裁定说理-1',
    'followlaw_7执行和解摘要-0',
    'followlaw_8劳动争议证据评价-1',
    'followlaw_9二审行政审理要点-0',
    'followlaw_10金融案件抗辩分类-1',
    'followlaw_11二审民事说理-0',
    'followlaw_12民事事实认定-1',
    'followlaw_13最终诉请-0',
    'followlaw_14一审民事说理-1',
]

followlaw_summary_groups.append({'name': 'followlaw_accuracy', 'subsets': _follwlaw_all})

# followlaw_summary_groups.append(
#     {'name': 'followlaw_Inst-level-strict-accuracy', 'subsets': [c for c in _follwlaw_all]})

# mb_gaokao_summary_groups.append({'name': 'mb_gaokao_5shot-weighted', 'subsets': list(_mb_gaokao_weights_5shot.keys()),
#                                  'weights': _mb_gaokao_weights_5shot})
