from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import CMMLUDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

cmmlu_subject_mapping = {
    'agronomy': '农学',
    'anatomy': '解剖学',
    'ancient_chinese': '古汉语',
    'arts': '艺术学',
    'astronomy': '天文学',
    'business_ethics': '商业伦理',
    'chinese_civil_service_exam': '中国公务员考试',
    'chinese_driving_rule': '中国驾驶规则',
    'chinese_food_culture': '中国饮食文化',
    'chinese_foreign_policy': '中国外交政策',
    'chinese_history': '中国历史',
    'chinese_literature': '中国文学',
    'chinese_teacher_qualification': '中国教师资格',
    'clinical_knowledge': '临床知识',
    'college_actuarial_science': '大学精算学',
    'college_education': '大学教育学',
    'college_engineering_hydrology': '大学工程水文学',
    'college_law': '大学法律',
    'college_mathematics': '大学数学',
    'college_medical_statistics': '大学医学统计',
    'college_medicine': '大学医学',
    'computer_science': '计算机科学',
    'computer_security': '计算机安全',
    'conceptual_physics': '概念物理学',
    'construction_project_management': '建设工程管理',
    'economics': '经济学',
    'education': '教育学',
    'electrical_engineering': '电气工程',
    'elementary_chinese': '小学语文',
    'elementary_commonsense': '小学常识',
    'elementary_information_and_technology': '小学信息技术',
    'elementary_mathematics': '初等数学',
    'ethnology': '民族学',
    'food_science': '食品科学',
    'genetics': '遗传学',
    'global_facts': '全球事实',
    'high_school_biology': '高中生物',
    'high_school_chemistry': '高中化学',
    'high_school_geography': '高中地理',
    'high_school_mathematics': '高中数学',
    'high_school_physics': '高中物理学',
    'high_school_politics': '高中政治',
    'human_sexuality': '人类性行为',
    'international_law': '国际法学',
    'journalism': '新闻学',
    'jurisprudence': '法理学',
    'legal_and_moral_basis': '法律与道德基础',
    'logical': '逻辑学',
    'machine_learning': '机器学习',
    'management': '管理学',
    'marketing': '市场营销',
    'marxist_theory': '马克思主义理论',
    'modern_chinese': '现代汉语',
    'nutrition': '营养学',
    'philosophy': '哲学',
    'professional_accounting': '专业会计',
    'professional_law': '专业法学',
    'professional_medicine': '专业医学',
    'professional_psychology': '专业心理学',
    'public_relations': '公共关系',
    'security_study': '安全研究',
    'sociology': '社会学',
    'sports_science': '体育学',
    'traditional_chinese_medicine': '中医中药',
    'virology': '病毒学',
    'world_history': '世界历史',
    'world_religions': '世界宗教'
}


cmmlu_all_sets = list(cmmlu_subject_mapping.keys())

cmmlu_datasets = []
for _name in cmmlu_all_sets:
    _ch_name = cmmlu_subject_mapping[_name]
    cmmlu_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        # f'以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}'
                        # 以下是关于农学的单项选择题，请直接给出正确答案的选项。\n\n问题：\n精液保存的技术关键是\n要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n选项：\n(A) 减少精子能量损失\n(B) 扩大利用率\n(C) 延长保存时间\n(D) 便于运输\n答案：\n(A)\n\n问题：\n产绒的羊种是\n要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n选项：\n(A) 细毛羊\n(B) 半毛羊\n(C) 山羊\n(D) 肉羊\n答案：\n(C)\n\n问题：\n可用于空气、物体表面消毒及组织表面感染治疗的光线是\n要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n选项：\n(A) 光辐射\n(B) 可见光线\n(C) 紫外线\n(D) 红外线\n答案：\n(C)\n\n问题：\n下列作物属于四碳作物的是\n要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n选项：\n(A) 小麦\n(B) 甘蔗\n(C) 棉花\n(D) 烟草\n答案：\n(B)\n\n问题：\n农业生态系统养分循环的三个主要养分库是\n要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n选项：\n(A) 水体库、土壤库、生物库\n(B) 土壤有效养分库、土壤矿物库、土壤有机物库\n(C) 植物库、动物库、土壤库\n(D) 土壤库、岩石库、生物库\n答案：\n(C)\n\n问题：\n只有在一个（）的市场条件下，生产要素的价格才能充分反映要素的稀缺程度，产品价格才能真正体现供求关系\n要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n选项：\n(A) 完全垄断\n(B) 完全竞争\n(C) 竞争垄断\n(D) 垄断竞争\n答案：\n
                        # f'以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n\n问题：\n{{question}}\n要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n选项：\n\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n'
                        # f'以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n\n问题：\n{{question}}\n要求：\n从以下选项中选择并回答正确答案的选项字母，包括左右括号。\n选项：\n(A) {{A}}\n(B) {{B}}\n(C) {{C}}\n(D) {{D}}'
                        f'以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: '
                    ),
                    dict(role='BOT', prompt='{answer}'),
                ]),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer, max_out_len=50),
    )

    cmmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_capital_postprocess))

    cmmlu_datasets.append(
        dict(
            type=CMMLUDataset,
            path='./data/cmmlu/',
            name=_name,
            abbr=f'cmmlu-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'),
            infer_cfg=cmmlu_infer_cfg,
            eval_cfg=cmmlu_eval_cfg,
        ))

del _name, _ch_name
