from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.mmlu import mmlu_summary_groups
    from ...summarizers.groups.cmmlu import cmmlu_summary_groups
    from ...summarizers.groups.ceval import ceval_summary_groups
    from ...summarizers.groups.bbh import bbh_summary_groups
    from ..summarizers.groups.mmlu_cot import mmlu_cot_summary_groups
    from ..summarizers.groups.cmmlu_cot import cmmlu_cot_summary_groups
    from ..summarizers.groups.ceval_cot import ceval_cot_summary_groups
    from ..summarizers.groups.mb_gaokao import mb_gaokao_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # mmlu
        ['mmlu', 'naive_average'],
        ['mmlu_cot', 'naive_average'],

        # cmmlu
        ['cmmlu', 'naive_average'],
        ['cmmlu_cot', 'naive_average'],

        # ceval
        ['ceval', 'naive_average'],
        ['ceval_cot', 'naive_average'],

        # arc
        ['ARC-c', 'accuracy'],
        ['ARC-e', 'accuracy'],

        # bbh
        ['bbh', 'naive_average'],

        # gpqa
        ['GPQA_diamond', 'accuracy'],

        # math
        ['math', 'accuracy'],

        # gsm8k
        ['gsm8k', 'accuracy'],

        # sanitized mbpp
        ['sanitized_mbpp', 'score'],

        # humaneval
        ['openai_humaneval', 'humaneval_pass@1'],

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
