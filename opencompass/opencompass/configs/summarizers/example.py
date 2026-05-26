from mmengine.config import read_base

with read_base():
    from .groups.agieval import agieval_summary_groups
    from .groups.babilong import babilong_summary_groups
    from .groups.bbeh import bbeh_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.calm import calm_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.ceval_cot import ceval_cot_summary_groups
    from .groups.charm_reason import charm_reason_summary_groups
    from .groups.cibench import cibench_summary_groups
    from .groups.cmmlu import cmmlu_summary_groups
    from .groups.ds1000 import ds1000_summary_groups
    from .groups.flores import flores_summary_groups
    from .groups.gsm8k_contamination import gsm8k_contamination_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups
    from .groups.humanevalx import humanevalx_summary_groups
    from .groups.infinitebench import infinitebench_summary_groups
    from .groups.jigsaw_multilingual import jigsaw_multilingual_summary_groups
    from .groups.korbench import korbench_summary_groups
    from .groups.lawbench import lawbench_summary_groups
    from .groups.lcbench import lcbench_summary_groups
    from .groups.leval import leval_summary_groups
    from .groups.longbench import longbench_summary_groups
    from .groups.lveval import lveval_summary_groups
    # from .groups.mathbench_2024 import mathbench_2024_
    # from .groups.mathbench_agent import mathbench_agent_summary_groups
    # from .groups.mathbench_v1_2024_lang import mathbench_2024_summary_groups
    # from .groups.mathbench_v1_2024 import mathbench_2024_summary_groups
    # from .groups.mathbench_v1 import mathbench_v1_summary_groups
    from .groups.mathbench import mathbench_summary_groups
    from .groups.mb_gaokao import mb_gaokao_summary_groups
    from .groups.mgsm import mgsm_summary_groups
    from .groups.mmlu_cf import mmlu_cf_summary_groups
    from .groups.mmlu_pro import mmlu_pro_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.MMLUArabic import MMLUArabic_summary_groups
    from .groups.mmmlu import mmmlu_summary_groups
    from .groups.multipl_e import multiple_summary_groups
    from .groups.minibench import ceval_minibench_summary_groups, mmlu_minibench_summary_groups, cmmlu_minibench_summary_groups
    from .groups.OlympiadBench import OlympiadBench_summary_groups
    from .groups.PHYSICS import physics_summary_groups
    from .groups.plugineval import plugineval_summary_groups
    from .groups.PMMEval import PMMEval_summary_groups
    from .groups.ruler import ruler_summary_groups
    from .groups.scibench import scibench_summary_groups
    from .groups.scicode import scicode_summary_groups
    from .groups.supergpqa import supergpqa_summary_groups
    from .groups.teval import teval_summary_groups
    from .groups.tydiqa import tydiqa_summary_groups
    from .groups.xiezhi import xiezhi_summary_groups
    from .groups.benbench import benbench_summary_groups
    from ..mb_internal.summarizers.groups.mmlu_cot import mmlu_cot_summary_groups
    from .groups.mathbench import mathbench_summary_groups
    from .groups.FinanceIQ import FinanceIQ_summary_groups, FinanceIQ_0shot_summary_groups
    from .groups.mb_gaokao2025mock import mb_gaokao2025_summary_groups
    from .groups.repeat_samples import repeat_samples_summary_groups
    from .groups.mmlu_redux import mmlu_redux_summary_groups
    from .groups.mb_safety import mb_safety_summary_groups
    from .groups.bigcodebench import bigcodebench_summary_groups
    from .groups.rbench import rbench_summary_groups
    from .groups.arenahardv2 import arenahardv2_summary_groups
    from .groups.mb_general import openai_mmmlu_lite_summary_groups

summarizer = dict(
    summary_groups=sum([v for k, v in locals().items()
                       if k.endswith('_summary_groups')], []),
)
