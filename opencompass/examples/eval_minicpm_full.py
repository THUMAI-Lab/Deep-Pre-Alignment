
from mmengine.config import read_base

with read_base():
    from .models.openbmb.hf_minicpm_2b_sft_bf16 import models
    # from .models.llama.llama2_7b import models
    
    # --------- 语言 Language ---------
    from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from .datasets.summedits.summedits_gen_315438 import summedits_datasets
    from .datasets.FewCLUE_chid.FewCLUE_chid_gen_0a29a2 import chid_datasets
    from .datasets.CLUE_afqmc.CLUE_afqmc_gen_901306 import afqmc_datasets
    from .datasets.FewCLUE_bustm.FewCLUE_bustm_gen_634f41 import bustm_datasets
    from .datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_gen_c68933 import cluewsc_datasets
    from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import WSC_datasets
    # from .datasets.winogrande.winogrande_gen_a9ede5 import winogrande_datasets
    from .datasets.winogrande.winogrande_gen_458220 import winogrande_datasets
    # from .datasets.flores.flores_gen_806ede import flores_datasets
    from .datasets.tydiqa.tydiqa_gen_978d2a import tydiqa_datasets
    from .datasets.CLUE_C3.CLUE_C3_gen_8c358f import C3_datasets
    from .datasets.CLUE_CMRC.CLUE_CMRC_gen_1bd3c8 import CMRC_datasets
    from .datasets.CLUE_DRCD.CLUE_DRCD_gen_1bd3c8 import DRCD_datasets
    from .datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_gen_27071f import MultiRC_datasets
    from .datasets.race.race_gen_69ee4f import race_datasets
    from .datasets.obqa.obqa_gen_9069e4 import obqa_datasets
    from .datasets.drop.drop_gen_8a9ed9 import drop_datasets
    from .datasets.FewCLUE_csl.FewCLUE_csl_gen_28b223 import csl_datasets
    from .datasets.lcsts.lcsts_gen_8ee1fe import lcsts_datasets
    from .datasets.Xsum.Xsum_gen_31397e import Xsum_datasets
    from .datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_gen_740ea0 import eprstmt_datasets
    from .datasets.lambada.lambada_gen_217e11 import lambada_datasets
    from .datasets.FewCLUE_tnews.FewCLUE_tnews_gen_b90e4a import tnews_datasets
    # --------- 知识 Knowledge ---------
    from .datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_883d50 import BoolQ_datasets
    # from .datasets.commonsenseqa.commonsenseqa_gen_c946f2 import commonsenseqa_datasets
    from .datasets.nq.nq_gen_c788f6 import nq_datasets
    from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    # different with v052
    # from .datasets.ceval.ceval_internal_gen_2daf24 import ceval_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from .datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from .datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    from .datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets
    # from .datasets.wikibench.wikibench_gen_f96ece import wikibench_datasets  # TODO: check data
    from .datasets.commonsenseqa_cn.commonsenseqacn_gen_d380d0 import commonsenseqacn_datasets
    from .datasets.nq_cn.nqcn_gen_141737 import nqcn_datasets
    # --------- 推理 Reasoning ---------
    from .datasets.CLUE_cmnli.CLUE_cmnli_gen_1abf97 import cmnli_datasets
    from .datasets.CLUE_ocnli.CLUE_ocnli_gen_c4cb6c import ocnli_datasets
    from .datasets.FewCLUE_ocnli_fc.FewCLUE_ocnli_fc_gen_f97a97 import ocnli_fc_datasets
    from .datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_gen_4dfefa import AX_b_datasets
    from .datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_gen_68aac7 import AX_g_datasets
    from .datasets.SuperGLUE_RTE.SuperGLUE_RTE_gen_68aac7 import RTE_datasets
    from .datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets
    from .datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from .datasets.piqa.piqa_gen_1194eb import piqa_datasets
    from .datasets.siqa.siqa_gen_e78df3 import siqa_datasets
    # from .datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets  # TODO: check data
    from .datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from .datasets.strategyqa.strategyqa_gen_1180a7 import strategyqa_datasets
    # --------- 数学 Mathematics ---------
    # math-evaluator v1
    # from .datasets.math.math_gen_265cce import math_datasets
    # math-evaluator v2
    from .datasets.math.deprecated_math_evaluatorv2_gen_265cce import math_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # NOTE: the setting of mathbench_gen_ad37c1 dataset is not exists in the common dataset, add this in sft_cfg
    # from .datasets.MathBench.mathbench_gen_ad37c1 import mathbench_datasets
    # use CoT in MathBench, and update prompt
    # from .datasets.MathBench.mathbench_cot_gen_66f329 import mathbench_datasets
    # from .datasets.MathBench.mathbench_2024_gen_1dc21d import mathbench_datasets
    # from .datasets.MathBench.mathbench_arith_gen_ccd638 import mathbench_datasets as arithmath_datasets
    # from .datasets.gsm_hard.gsmhard_gen_8a1400 import gsmhard_datasets  ### TODO: Check data
    # NOTE: the setting of gsm8k_option_gen_108724 dataset is not exists in the common dataset, add this in sft_cfg
    # from .datasets.gsm8k_extra.gsm8k_option_gen_108724 import gsm8k_option_datasets as gsm8k_option_datasets
    # --------- 代码 Coding ---------
    # the new prompt can get better performance than the old one
    # from .datasets.mbpp.deprecated_mbpp_gen_caa7ab import mbpp_datasets  # mbpp_gen_830460

    from .datasets.mbpp_cn.mbpp_cn_gen_9114d5 import mbpp_cn_datasets
    from .datasets.mbpp.mbpp_gen_830460 import mbpp_datasets
    from .datasets.humaneval.humaneval_gen_6d1cc2 import humaneval_datasets
    #  TODO: CodeBench

    # old prompt setting
    # from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets
    # from .datasets.ds1000.ds1000_service_eval_gen_cbc84f import ds1000_datasets

    # from .datasets.py150.py150_gen_38b13d import py150_datasets
    # from .datasets.clozeTest_maxmin.clozeTest_maxmin_gen_c205fb import maxmin_datasets

    #  agent
    #  cibench t-eval/plugineval
    #  TODO: agent模型配置需要重新确认


    # --------- summary ---------
    from .summarizers.groups.agieval import agieval_summary_groups
    from .summarizers.groups.mmlu import mmlu_summary_groups
    from .summarizers.groups.ceval import ceval_summary_groups
    from .summarizers.groups.bbh import bbh_summary_groups
    from .summarizers.groups.GaokaoBench import GaokaoBench_summary_groups
    from .summarizers.groups.flores import flores_summary_groups
    from .summarizers.groups.jigsaw_multilingual import jigsaw_multilingual_summary_groups
    from .summarizers.groups.cmmlu import cmmlu_summary_groups
    from .summarizers.groups.xiezhi import xiezhi_summary_groups
    from .summarizers.groups.tydiqa import tydiqa_summary_groups
    from .summarizers.groups.ds1000 import ds1000_summary_groups
    from .summarizers.groups.mathbench import mathbench_summary_groups
    # NOTE: this is the old version of CIBench summarizer groups, not exist any more, add this in the sft_cfg
    from .summarizers.groups.plugineval import plugineval_summary_groups


other_summary_groups = [
    {
        'name': 'Language',
        'subsets': [['WiC', 'accuracy'], ['summedits', 'accuracy'], ['chid-dev', 'accuracy'], ['afqmc-dev', 'accuracy'],
                    ['bustm-dev', 'accuracy'], ['cluewsc-dev', 'accuracy'], ['WSC', 'accuracy'],
                    ['winogrande', 'accuracy'], ['flores_100', 'naive_average'], ['tydiqa-goldp', 'f1'],
                    ['C3', 'accuracy'], ['CMRC_dev', 'score'], ['DRCD_dev', 'score'], ['MultiRC', 'accuracy'],
                    ['race-middle', 'accuracy'], ['race-high', 'accuracy'], ['openbookqa_fact', 'accuracy'],
                    ['drop', 'score'], ['csl_dev', 'accuracy'], ['lcsts', 'rouge1'], ['Xsum', 'rouge1'],
                    ['eprstmt-dev', 'accuracy'], ['lambada', 'accuracy'], ['tnews-dev', 'accuracy']],
    },
    {
        'name': 'Knowledge',
        'subsets': [['BoolQ', 'accuracy'], ['commonsense_qa', 'accuracy'], ['nq', 'score'], ['triviaqa', 'score'],
                    ['tydiqa-goldp_english', 'f1'], ['ceval', 'naive_average'], ['ceval-test', 'naive_average'],
                    ['agieval', 'naive_average'], ['mmlu', 'naive_average'], ['GaokaoBench', 'weighted_average'],
                    ['ARC-c', 'accuracy'], ['ARC-e', 'accuracy'], ['cmmlu', 'naive_average'],
                    ['wikibench-wiki-single_choice_cncircular', 'perf_4'], ['commonsenseqa_cn', 'accuracy'],
                    ['nq_cn', 'score']],
    },
    {
        'name': 'Reasoning',
        'subsets': [['cmnli', 'accuracy'], ['ocnli', 'accuracy'], ['ocnli_fc-dev', 'accuracy'], ['AX_b', 'accuracy'],
                    ['AX_g', 'accuracy'], ['RTE', 'accuracy'], ['ReCoRD', 'score'], ['hellaswag', 'accuracy'],
                    ['piqa', 'accuracy'], ['siqa', 'accuracy'], ['strategyqa', 'accuracy'], ['TheoremQA', 'accuracy'],
                    ['bbh', 'naive_average']],
    },
    {
        'name': 'Mathematics',
        'subsets': [['math', 'accuracy'], ['gsm8k', 'accuracy'], ['mathbench-circular-and-cloze', 'naive_average'],
                    ['mathbench-arithmeticarithmetic-cloze_arith_en', 'accuracy'], ['gsm-hard', 'accuracy'],
                    ['gsm8k-extra-options', 'perf_4']],
    },
    {
        'name': 'Coding',
        'subsets': [['openai_humaneval', 'humaneval_pass@1'], ['mbpp', 'score'], ['ds1000', 'naive_average'],
                    ['py150', 'score'], ['maxmin', 'accuracy']],
    },
    {
        'name': 'Agent',
        'subsets': [['math-agent', 'follow_acc'], ['math-agent', 'reasoning_acc'], ['gsm8k-agent', 'follow_acc'],
                    ['gsm8k-agent', 'reasoning_acc'], ['mathbench-circular-and-cloze-agent', 'naive_average'],
                    ['cibench_generation', 'executable'], ['plugin_eval', 'naive_average']],
    },
    {
        'name': 'Overall',
        'subsets': ['Language', 'Knowledge', 'Reasoning', 'Mathematics', 'Coding', 'Agent'],
    }
]

summarizer = dict(
    dataset_abbrs=[
        # 'Overall',
        # 'Language',
        # 'Knowledge',
        # 'Reasoning',
        # 'Mathematics',
        # 'Coding',
        # 'Agent',
        # '--------- 语言 Language ---------',
        ['WiC', 'accuracy'],
        ['summedits', 'accuracy'],
        ['chid-dev', 'accuracy'],
        ['afqmc-dev', 'accuracy'],
        ['bustm-dev', 'accuracy'],
        ['cluewsc-dev', 'accuracy'],
        ['WSC', 'accuracy'],
        ['winogrande', 'accuracy'],
        ['flores_100', 'naive_average'],
        ['tydiqa-goldp', 'f1'],
        ['C3', 'accuracy'],
        ['CMRC_dev', 'score'],
        ['DRCD_dev', 'score'],
        ['MultiRC', 'accuracy'],
        ['race-middle', 'accuracy'],
        ['race-high', 'accuracy'],
        ['openbookqa_fact', 'accuracy'],
        ['drop', 'score'],
        ['csl_dev', 'accuracy'],
        ['lcsts', 'rouge1'],
        ['Xsum', 'rouge1'],
        ['eprstmt-dev', 'accuracy'],
        ['lambada', 'accuracy'],
        ['tnews-dev', 'accuracy'],
        # '--------- 知识 Knowledge ---------',
        ['BoolQ', 'accuracy'],
        ['commonsense_qa', 'accuracy'],
        ['nq', 'score'],
        ['triviaqa', 'score'],
        ['tydiqa-goldp_english', 'f1'],
        ['ceval', 'naive_average'],
        ['ceval-test', 'naive_average'],
        ['agieval', 'naive_average'],
        ['mmlu', 'naive_average'],
        ['GaokaoBench', 'weighted_average'],
        ['ARC-c', 'accuracy'],
        ['ARC-e', 'accuracy'],
        ['cmmlu', 'naive_average'],
        ['wikibench-wiki-single_choice_cncircular', 'perf_4'],
        ['commonsenseqa_cn', 'accuracy'],
        ['nq_cn', 'score'],
        # '--------- 推理 Reasoning ---------',
        ['cmnli', 'accuracy'],
        ['ocnli', 'accuracy'],
        ['ocnli_fc-dev', 'accuracy'],
        ['AX_b', 'accuracy'],
        ['AX_g', 'accuracy'],
        ['RTE', 'accuracy'],
        ['ReCoRD', 'score'],
        ['hellaswag', 'accuracy'],
        ['piqa', 'accuracy'],
        ['siqa', 'accuracy'],
        ['strategyqa', 'accuracy'],
        ['TheoremQA', 'accuracy'],
        ['bbh', 'naive_average'],
        # '--------- 数学 Mathematics ---------',
        ['math', 'accuracy'],
        ['gsm8k', 'accuracy'],
        ['mathbench', 'naive_average'],
        ['mathbench-circular', 'naive_average'],
        ['mathbench-circular-and-cloze', 'naive_average'],
        ['mathbench-arithmeticarithmetic-cloze_arith_en', 'accuracy'],
        ['gsm-hard', 'accuracy'],
        ['gsm8k-extra-options', 'perf_4'],
        # '--------- 代码 Coding ---------',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['mbpp', 'score'],
        ['ds1000', 'naive_average'],
        ['py150', 'score'],
        ['maxmin', 'accuracy'],
        # '--------- 智能体 Agent ---------',
        ['math-agent', 'follow_acc'],
        ['math-agent', 'reasoning_acc'],
        ['gsm8k-agent', 'follow_acc'],
        ['gsm8k-agent', 'reasoning_acc'],
        ['mathbench-agent', 'naive_average'],
        ['mathbench-circular-agent', 'naive_average'],
        ['mathbench-circular-and-cloze-agent', 'naive_average'],
        # '--------- cibench ---------',
        ['cibench_generation', 'executable'],
        ['cibench_generation_Pandas', 'executable'],
        ['cibench_generation_Matplotlib', 'executable'],
        ['cibench_generation_Opencv', 'executable'],
        ['cibench_generation_SciPy', 'executable'],
        ['cibench_generation_Seaborn', 'executable'],
        ['cibench_generation_PyTorch', 'executable'],
        # ['cibench_generation', 'vis_sim'],
        # ['cibench_generation_Pandas', 'vis_sim'],
        # ['cibench_generation_Matplotlib', 'vis_sim'],
        # ['cibench_generation_Opencv', 'vis_sim'],
        # ['cibench_generation_SciPy', 'vis_sim'],
        # ['cibench_generation_Seaborn', 'vis_sim'],
        # ['cibench_generation_PyTorch', 'vis_sim'],
        # ['cibench_generation', 'general_correct'],
        # ['cibench_generation_Pandas', 'general_correct'],
        # ['cibench_generation_Matplotlib', 'general_correct'],
        # ['cibench_generation_Opencv', 'general_correct'],
        # ['cibench_generation_SciPy', 'general_correct'],
        # ['cibench_generation_Seaborn', 'general_correct'],
        # ['cibench_generation_PyTorch', 'general_correct'],
        # '--------- plugin_eval ---------',
        ['plugin_eval', 'naive_average'],
        ['plugin_eval-instruct_v1', 'format_metric'],
        ['plugin_eval-instruct_v1', 'args_em_metric'],
        ['plugin_eval-plan_str_v1', 'f1_score'],
        ['plugin_eval-plan_json_v1', 'f1_score'],
        ['plugin_eval-reason_str_v1', 'thought'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'thought'],
        ['plugin_eval-retrieve_str_v1', 'name'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'name'],
        ['plugin_eval-understand_str_v1', 'args'],
        ['plugin_eval-reason_retrieve_understand_json_v1', 'args'],
        ['plugin_eval-review_str_v1', 'review_quality'],
        # '--------- longeval ---------',
        ['longeval', 'naive_average'],
        ['longeval_2k', 'naive_average'],
        ['longeval_4k', 'naive_average'],
        ['longeval_8k', 'naive_average'],
        ['longeval_15k', 'naive_average'],
        ['longeval_30k', 'naive_average'],
        ['2wikimqa_e_4k', 'score'],
        ['2wikimqa_e_8k', 'score'],
        ['2wikimqa_e_15k', 'score'],
        ['gov_report_4k', 'score'],
        ['gov_report_8k', 'score'],
        ['hotpotqa_e_15k', 'score'],
        ['lines_2k', 'score'],
        ['lines_4k', 'score'],
        ['lines_8k', 'score'],
        ['lines_15k', 'score'],
        ['lines_30k', 'score'],
        ['multifieldqa_zh_4k', 'score'],
        ['passage_retrieval_zh_8k', 'score'],
        ['stackselect_2k', 'score'],
        ['stackselect_4k', 'score'],
        ['stackselect_8k', 'score'],
        ['stackselect_15k', 'score'],
        ['stackselect_30k', 'score'],
        ['textsort_2k', 'score'],
        ['textsort_4k', 'score'],
        ['textsort_8k', 'score'],
        ['textsort_15k', 'score'],
        ['textsort_30k', 'score'],
        ['trec_e_2k', 'score'],
        ['trec_e_4k', 'score'],
        ['trec_e_8k', 'score'],
        ['trec_e_15k', 'score'],
        # '--------- ds1000 细节 ---------',
        ['ds1000_Pandas', 'accuracy'],
        ['ds1000_Numpy', 'accuracy'],
        ['ds1000_Tensorflow', 'accuracy'],
        ['ds1000_Scipy', 'accuracy'],
        ['ds1000_Sklearn', 'accuracy'],
        ['ds1000_Pytorch', 'accuracy'],
        ['ds1000_Matplotlib', 'accuracy'],
        # '--------- ceval 细节 ---------',
        ['ceval-stem', 'naive_average'],
        ['ceval-social-science', 'naive_average'],
        ['ceval-humanities', 'naive_average'],
        ['ceval-other', 'naive_average'],
        ['ceval-hard', 'naive_average'],
        # category
        ['ceval-advanced_mathematics', 'accuracy'],
        ['ceval-college_chemistry', 'accuracy'],
        ['ceval-college_physics', 'accuracy'],
        ['ceval-college_programming', 'accuracy'],
        ['ceval-computer_architecture', 'accuracy'],
        ['ceval-computer_network', 'accuracy'],
        ['ceval-discrete_mathematics', 'accuracy'],
        ['ceval-electrical_engineer', 'accuracy'],
        ['ceval-high_school_biology', 'accuracy'],
        ['ceval-high_school_chemistry', 'accuracy'],
        ['ceval-high_school_mathematics', 'accuracy'],
        ['ceval-high_school_physics', 'accuracy'],
        ['ceval-metrology_engineer', 'accuracy'],
        ['ceval-middle_school_biology', 'accuracy'],
        ['ceval-middle_school_chemistry', 'accuracy'],
        ['ceval-middle_school_mathematics', 'accuracy'],
        ['ceval-middle_school_physics', 'accuracy'],
        ['ceval-operating_system', 'accuracy'],
        ['ceval-probability_and_statistics', 'accuracy'],
        ['ceval-veterinary_medicine', 'accuracy'],
        ['ceval-business_administration', 'accuracy'],
        ['ceval-college_economics', 'accuracy'],
        ['ceval-education_science', 'accuracy'],
        ['ceval-high_school_geography', 'accuracy'],
        ['ceval-high_school_politics', 'accuracy'],
        ['ceval-mao_zedong_thought', 'accuracy'],
        ['ceval-marxism', 'accuracy'],
        ['ceval-middle_school_geography', 'accuracy'],
        ['ceval-middle_school_politics', 'accuracy'],
        ['ceval-teacher_qualification', 'accuracy'],
        ['ceval-art_studies', 'accuracy'],
        ['ceval-chinese_language_and_literature', 'accuracy'],
        ['ceval-high_school_chinese', 'accuracy'],
        ['ceval-high_school_history', 'accuracy'],
        ['ceval-ideological_and_moral_cultivation', 'accuracy'],
        ['ceval-law', 'accuracy'],
        ['ceval-legal_professional', 'accuracy'],
        ['ceval-logic', 'accuracy'],
        ['ceval-middle_school_history', 'accuracy'],
        ['ceval-modern_chinese_history', 'accuracy'],
        ['ceval-professional_tour_guide', 'accuracy'],
        ['ceval-accountant', 'accuracy'],
        ['ceval-basic_medicine', 'accuracy'],
        ['ceval-civil_servant', 'accuracy'],
        ['ceval-clinical_medicine', 'accuracy'],
        ['ceval-environmental_impact_assessment_engineer', 'accuracy'],
        ['ceval-fire_engineer', 'accuracy'],
        ['ceval-physician', 'accuracy'],
        ['ceval-plant_protection', 'accuracy'],
        ['ceval-sports_science', 'accuracy'],
        ['ceval-tax_accountant', 'accuracy'],
        ['ceval-urban_and_rural_planner', 'accuracy'],
        # '--------- agieval 细节 ---------',
        ['agieval-chinese', 'naive_average'],
        ['agieval-english', 'naive_average'],
        ['agieval-gaokao', 'naive_average'],
        # category
        ['agieval-aqua-rat', 'accuracy'],
        ['agieval-math', 'accuracy'],
        ['agieval-logiqa-en', 'accuracy'],
        ['agieval-logiqa-zh', 'accuracy'],
        ['agieval-jec-qa-kd', 'accuracy'],
        ['agieval-jec-qa-ca', 'accuracy'],
        ['agieval-lsat-ar', 'accuracy'],
        ['agieval-lsat-lr', 'accuracy'],
        ['agieval-lsat-rc', 'accuracy'],
        ['agieval-sat-math', 'accuracy'],
        ['agieval-sat-en', 'accuracy'],
        ['agieval-sat-en-without-passage', 'accuracy'],
        ['agieval-gaokao-chinese', 'accuracy'],
        ['agieval-gaokao-english', 'accuracy'],
        ['agieval-gaokao-geography', 'accuracy'],
        ['agieval-gaokao-history', 'accuracy'],
        ['agieval-gaokao-biology', 'accuracy'],
        ['agieval-gaokao-chemistry', 'accuracy'],
        ['agieval-gaokao-physics', 'accuracy'],
        ['agieval-gaokao-mathqa', 'accuracy'],
        ['agieval-gaokao-mathcloze', 'accuracy'],
        # '--------- mmlu 细节 ---------',
        ['mmlu-humanities', 'naive_average'],
        ['mmlu-stem', 'naive_average'],
        ['mmlu-social-science', 'naive_average'],
        ['mmlu-other', 'naive_average'],
        # category
        ['lukaemon_mmlu_abstract_algebra', 'accuracy'],
        ['lukaemon_mmlu_anatomy', 'accuracy'],
        ['lukaemon_mmlu_astronomy', 'accuracy'],
        ['lukaemon_mmlu_business_ethics', 'accuracy'],
        ['lukaemon_mmlu_clinical_knowledge', 'accuracy'],
        ['lukaemon_mmlu_college_biology', 'accuracy'],
        ['lukaemon_mmlu_college_chemistry', 'accuracy'],
        ['lukaemon_mmlu_college_computer_science', 'accuracy'],
        ['lukaemon_mmlu_college_mathematics', 'accuracy'],
        ['lukaemon_mmlu_college_medicine', 'accuracy'],
        ['lukaemon_mmlu_college_physics', 'accuracy'],
        ['lukaemon_mmlu_computer_security', 'accuracy'],
        ['lukaemon_mmlu_conceptual_physics', 'accuracy'],
        ['lukaemon_mmlu_econometrics', 'accuracy'],
        ['lukaemon_mmlu_electrical_engineering', 'accuracy'],
        ['lukaemon_mmlu_elementary_mathematics', 'accuracy'],
        ['lukaemon_mmlu_formal_logic', 'accuracy'],
        ['lukaemon_mmlu_global_facts', 'accuracy'],
        ['lukaemon_mmlu_high_school_biology', 'accuracy'],
        ['lukaemon_mmlu_high_school_chemistry', 'accuracy'],
        ['lukaemon_mmlu_high_school_computer_science', 'accuracy'],
        ['lukaemon_mmlu_high_school_european_history', 'accuracy'],
        ['lukaemon_mmlu_high_school_geography', 'accuracy'],
        ['lukaemon_mmlu_high_school_government_and_politics', 'accuracy'],
        ['lukaemon_mmlu_high_school_macroeconomics', 'accuracy'],
        ['lukaemon_mmlu_high_school_mathematics', 'accuracy'],
        ['lukaemon_mmlu_high_school_microeconomics', 'accuracy'],
        ['lukaemon_mmlu_high_school_physics', 'accuracy'],
        ['lukaemon_mmlu_high_school_psychology', 'accuracy'],
        ['lukaemon_mmlu_high_school_statistics', 'accuracy'],
        ['lukaemon_mmlu_high_school_us_history', 'accuracy'],
        ['lukaemon_mmlu_high_school_world_history', 'accuracy'],
        ['lukaemon_mmlu_human_aging', 'accuracy'],
        ['lukaemon_mmlu_human_sexuality', 'accuracy'],
        ['lukaemon_mmlu_international_law', 'accuracy'],
        ['lukaemon_mmlu_jurisprudence', 'accuracy'],
        ['lukaemon_mmlu_logical_fallacies', 'accuracy'],
        ['lukaemon_mmlu_machine_learning', 'accuracy'],
        ['lukaemon_mmlu_management', 'accuracy'],
        ['lukaemon_mmlu_marketing', 'accuracy'],
        ['lukaemon_mmlu_medical_genetics', 'accuracy'],
        ['lukaemon_mmlu_miscellaneous', 'accuracy'],
        ['lukaemon_mmlu_moral_disputes', 'accuracy'],
        ['lukaemon_mmlu_moral_scenarios', 'accuracy'],
        ['lukaemon_mmlu_nutrition', 'accuracy'],
        ['lukaemon_mmlu_philosophy', 'accuracy'],
        ['lukaemon_mmlu_prehistory', 'accuracy'],
        ['lukaemon_mmlu_professional_accounting', 'accuracy'],
        ['lukaemon_mmlu_professional_law', 'accuracy'],
        ['lukaemon_mmlu_professional_medicine', 'accuracy'],
        ['lukaemon_mmlu_professional_psychology', 'accuracy'],
        ['lukaemon_mmlu_public_relations', 'accuracy'],
        ['lukaemon_mmlu_security_studies', 'accuracy'],
        ['lukaemon_mmlu_sociology', 'accuracy'],
        ['lukaemon_mmlu_us_foreign_policy', 'accuracy'],
        ['lukaemon_mmlu_virology', 'accuracy'],
        ['lukaemon_mmlu_world_religions', 'accuracy'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
