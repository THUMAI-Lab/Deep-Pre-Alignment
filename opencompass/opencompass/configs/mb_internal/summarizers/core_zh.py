from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.cmmlu import cmmlu_summary_groups
    from ...summarizers.groups.ceval import ceval_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # mmlu
        ['cmmlu', 'naive_average'],

        # ceval
        ['ceval', 'naive_average'],

        # commonsenseqa cn
        ['commonsenseqa_cn', 'accuracy'],

        # csl
        ['csl_dev', 'accuracy'],
        ['csl_test', 'accuracy'],

        # nq_cn
        ['nq_cn', 'score'],

        # ocnli
        ['ocnli', 'accuracy'],

        # cmnli
        ['cmnli', 'accuracy'],

        # chid
        ['chid-dev', 'accuracy'],
        ['chid-test', 'accuracy'],

        # C3
        ['C3', 'accuracy'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
