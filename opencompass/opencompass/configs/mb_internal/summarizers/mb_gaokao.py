from mmengine.config import read_base

with read_base():
    from ..summarizers.groups.mb_gaokao import mb_gaokao_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # mb-gaokao
        'mb_gaokao-0shot-weighted',
        'mb_gaokao-5shot-weighted',

        # 'mb_gaokao-0shot',
        ['mb_gaokao-gaokaotagging2024_1123-0shot', 'accuracy'],
        ['mb_gaokao-gaokaotagging2024_1126-0shot', 'accuracy'],
        ['mb_gaokao-gaokaotagging2024_1127-0shot', 'accuracy'],
        ['mb_gaokao-gaokaotagging2024_1128-0shot', 'accuracy'],
        ['mb_gaokao-gaokaotagging2024_1129-0shot', 'accuracy'],
        # 'mb_gaokao-5shot',
        ['mb_gaokao-gaokaotagging2024_1123-5shot', 'accuracy'],
        ['mb_gaokao-gaokaotagging2024_1126-5shot', 'accuracy'],
        ['mb_gaokao-gaokaotagging2024_1127-5shot', 'accuracy'],
        ['mb_gaokao-gaokaotagging2024_1128-5shot', 'accuracy'],
        ['mb_gaokao-gaokaotagging2024_1129-5shot', 'accuracy'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
