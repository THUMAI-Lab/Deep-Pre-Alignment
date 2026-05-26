from mmengine.config import read_base

with read_base():
    from ...summarizers.groups.agieval import agieval_summary_groups
    from ...summarizers.groups.mmlu import mmlu_summary_groups
    from ...summarizers.groups.mmlu_pro import mmlu_pro_summary_groups
    from ...summarizers.groups.cmmlu import cmmlu_summary_groups
    from ...summarizers.groups.ceval import ceval_summary_groups
    from ...summarizers.groups.bbh import bbh_summary_groups
    from ...summarizers.groups.GaokaoBench import GaokaoBench_summary_groups
    from ...summarizers.groups.flores import flores_summary_groups
    from ...summarizers.groups.tydiqa import tydiqa_summary_groups
    from ...summarizers.groups.xiezhi import xiezhi_summary_groups
    from ...summarizers.groups.scibench import scibench_summary_groups
    from ...summarizers.groups.mgsm import mgsm_summary_groups
    from ...summarizers.groups.longbench import longbench_summary_groups
    from ..summarizers.groups.mmlu_cot import mmlu_cot_summary_groups
    from ..summarizers.groups.cmmlu_cot import cmmlu_cot_summary_groups
    from ..summarizers.groups.ceval_cot import ceval_cot_summary_groups
    from ..summarizers.groups.mb_gaokao import mb_gaokao_summary_groups

summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
