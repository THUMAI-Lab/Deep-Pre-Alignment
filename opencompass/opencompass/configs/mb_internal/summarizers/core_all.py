from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.mmlu import mmlu_summary_groups
    from ...summarizers.groups.cmmlu import cmmlu_summary_groups
    from ...summarizers.groups.ceval import ceval_summary_groups
    from ...summarizers.groups.bbh import bbh_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # mmlu
        ['mmlu', 'naive_average'],

        # cmmlu
        ['cmmlu', 'naive_average'],

        # ceval
        ['ceval', 'naive_average'],

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
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
