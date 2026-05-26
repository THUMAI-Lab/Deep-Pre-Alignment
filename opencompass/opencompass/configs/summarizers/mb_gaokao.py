from mmengine.config import read_base

with read_base():
    from .groups.mb_gaokao import mb_gaokao_summary_groups

summarizer = dict(
    dataset_abbrs=[
        'mb_gaokao',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
