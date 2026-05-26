from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import BenBenchDataset, BenBenchEvaluator
import os


bbh_sets = [
    'temporal_sequences',
    'disambiguation_qa',
    'date_understanding',
    'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes',
    'snarks',
    'ruin_names',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'movie_recommendation',
    'salient_translation_error_detection',
    'reasoning_about_colored_objects',
    'multistep_arithmetic_two',
    'navigate',
    'dyck_languages',
    'word_sorting',
    'sports_understanding',
    'boolean_expressions',
    'object_counting',
    'formal_fallacies',
    'causal_judgement',
    'web_of_lies',
]

_ceval_subject_mapping = {
    'computer_network': ['Computer Network', '计算机网络', 'STEM'],
    'operating_system': ['Operating System', '操作系统', 'STEM'],
    'computer_architecture': ['Computer Architecture', '计算机组成', 'STEM'],
    'college_programming': ['College Programming', '大学编程', 'STEM'],
    'college_physics': ['College Physics', '大学物理', 'STEM'],
    'college_chemistry': ['College Chemistry', '大学化学', 'STEM'],
    'advanced_mathematics': ['Advanced Mathematics', '高等数学', 'STEM'],
    'probability_and_statistics': ['Probability and Statistics', '概率统计', 'STEM'],
    'discrete_mathematics': ['Discrete Mathematics', '离散数学', 'STEM'],
    'electrical_engineer': ['Electrical Engineer', '注册电气工程师', 'STEM'],
    'metrology_engineer': ['Metrology Engineer', '注册计量师', 'STEM'],
    'high_school_mathematics': ['High School Mathematics', '高中数学', 'STEM'],
    'high_school_physics': ['High School Physics', '高中物理', 'STEM'],
    'high_school_chemistry': ['High School Chemistry', '高中化学', 'STEM'],
    'high_school_biology': ['High School Biology', '高中生物', 'STEM'],
    'middle_school_mathematics': ['Middle School Mathematics', '初中数学', 'STEM'],
    'middle_school_biology': ['Middle School Biology', '初中生物', 'STEM'],
    'middle_school_physics': ['Middle School Physics', '初中物理', 'STEM'],
    'middle_school_chemistry': ['Middle School Chemistry', '初中化学', 'STEM'],
    'veterinary_medicine': ['Veterinary Medicine', '兽医学', 'STEM'],
    'college_economics': ['College Economics', '大学经济学', 'Social Science'],
    'business_administration': ['Business Administration', '工商管理', 'Social Science'],
    'marxism': ['Marxism', '马克思主义基本原理', 'Social Science'],
    'mao_zedong_thought': ['Mao Zedong Thought', '毛泽东思想和中国特色社会主义理论体系概论', 'Social Science'],
    'education_science': ['Education Science', '教育学', 'Social Science'],
    'teacher_qualification': ['Teacher Qualification', '教师资格', 'Social Science'],
    'high_school_politics': ['High School Politics', '高中政治', 'Social Science'],
    'high_school_geography': ['High School Geography', '高中地理', 'Social Science'],
    'middle_school_politics': ['Middle School Politics', '初中政治', 'Social Science'],
    'middle_school_geography': ['Middle School Geography', '初中地理', 'Social Science'],
    'modern_chinese_history': ['Modern Chinese History', '近代史纲要', 'Humanities'],
    'ideological_and_moral_cultivation': ['Ideological and Moral Cultivation', '思想道德修养与法律基础', 'Humanities'],
    'logic': ['Logic', '逻辑学', 'Humanities'],
    'law': ['Law', '法学', 'Humanities'],
    'chinese_language_and_literature': ['Chinese Language and Literature', '中国语言文学', 'Humanities'],
    'art_studies': ['Art Studies', '艺术学', 'Humanities'],
    'professional_tour_guide': ['Professional Tour Guide', '导游资格', 'Humanities'],
    'legal_professional': ['Legal Professional', '法律职业资格', 'Humanities'],
    'high_school_chinese': ['High School Chinese', '高中语文', 'Humanities'],
    'high_school_history': ['High School History', '高中历史', 'Humanities'],
    'middle_school_history': ['Middle School History', '初中历史', 'Humanities'],
    'civil_servant': ['Civil Servant', '公务员', 'Other'],
    'sports_science': ['Sports Science', '体育学', 'Other'],
    'plant_protection': ['Plant Protection', '植物保护', 'Other'],
    'basic_medicine': ['Basic Medicine', '基础医学', 'Other'],
    'clinical_medicine': ['Clinical Medicine', '临床医学', 'Other'],
    'urban_and_rural_planner': ['Urban and Rural Planner', '注册城乡规划师', 'Other'],
    'accountant': ['Accountant', '注册会计师', 'Other'],
    'fire_engineer': ['Fire Engineer', '注册消防工程师', 'Other'],
    'environmental_impact_assessment_engineer': ['Environmental Impact Assessment Engineer', '环境影响评价工程师', 'Other'],
    'tax_accountant': ['Tax Accountant', '税务师', 'Other'],
    'physician': ['Physician', '医师资格', 'Other'],
}
_ceval_sets = list(_ceval_subject_mapping.keys())

_cmmlu_subject_mapping = {
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
# cmmlu_sets = list(cmmlu_subject_mapping.keys())
_cmmlu_sets = [
    'agronomy',
    'anatomy',
    'ancient_chinese',
    'arts',
    'astronomy',
    'business_ethics',
    'chinese_civil_service_exam',
    'chinese_driving_rule',
    'chinese_food_culture',
    'chinese_foreign_policy',
    'chinese_history',
    'chinese_literature',
    'chinese_teacher_qualification',
    'clinical_knowledge',
    'college_actuarial_science',
    'college_education',
    'college_engineering_hydrology',
    'college_law',
    'college_mathematics',
    'college_medical_statistics',
    'college_medicine',
    'computer_science',
    'computer_security',
    'conceptual_physics',
    'construction_project_management',
    'economics',
    'education',
    'electrical_engineering',
    'elementary_chinese',
    'elementary_commonsense',
    'elementary_information_and_technology',
    'elementary_mathematics',
    'ethnology',
    'food_science',
    'genetics',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_geography',
    'high_school_mathematics',
    'high_school_physics',
    'high_school_politics',
    'human_sexuality',
    'international_law',
    'legal_and_moral_basis',
    'logical',
    'machine_learning',
    'management',
    'marketing',
    'marxist_theory',
    'modern_chinese',
    'nutrition',
    'philosophy',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_study',
    'sociology',
    'sports_science',
    'traditional_chinese_medicine',
    'virology',
    'world_history',
    'world_religions'
]

_mmlu_sets = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]


math500_sets = ['math_prm800k_500']

dataset_path_list = [
    ['humaneval', 'opencompass/humaneval', []],
    ['sanitized_mbpp', 'opencompass/sanitized_mbpp', []],
    ['gsm8k', 'opencompass/gsm8k', []],
    ['math500', 'opencompass/math', []],
    ['bbh', 'opencompass/bbh', bbh_sets],
    ['ceval', 'opencompass/ceval-exam', _ceval_sets],
    ['cmmlu', 'opencompass/cmmlu', _cmmlu_sets],
    ['mmlu', 'opencompass/mmlu', _mmlu_sets],
]


benbench_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='reference'
)

benbench_eval_cfg = dict(
    evaluator=dict(
        type=BenBenchEvaluator
    )
)

benbench_datasets = []

for n_gram in [
    # 5,
    10,
    # 15,
]:
    benbench_infer_cfg = dict(
        prompt_template=dict(type=PromptTemplate, template='{prompt}'),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=n_gram)
    )

    for dataset_abbr, dataset_path, sets_list in dataset_path_list:
        # print(dataset_abbr, dataset_path, sets_list)
        if not sets_list:
            benbench_datasets.append(
                dict(
                    abbr=f'{dataset_abbr}-test-origin-{n_gram}gram',
                    type=BenBenchDataset,
                    num_gram=n_gram,
                    # path=os.path.join(os.environ.get(
                    #     'COMPASS_DATA_CACHE', './'), dataset_path),
                    path=dataset_path,
                    # tokenizer_path=model_path, # todo, define tokenizer path
                    dataset_kwargs=dict(
                        dataset=dataset_abbr,
                        # name=dataset_abbr
                    ),
                    reader_cfg=benbench_reader_cfg,
                    infer_cfg=benbench_infer_cfg,
                    eval_cfg=benbench_eval_cfg
                )
            )
        else:
            for set_name in sets_list:
                # print(set_name)
                benbench_datasets.append(
                    dict(
                        abbr=f'{dataset_abbr}-test-{set_name}-origin-{n_gram}gram',
                        type=BenBenchDataset,
                        num_gram=n_gram,
                        # path=os.path.join(os.environ.get(
                        #     'COMPASS_DATA_CACHE', './'), dataset_path),
                        path=dataset_path,
                        # tokenizer_path=model_path, # todo, define tokenizer path
                        dataset_kwargs=dict(
                            dataset=dataset_abbr,
                            name=set_name
                        ),
                        reader_cfg=benbench_reader_cfg,
                        infer_cfg=benbench_infer_cfg,
                        eval_cfg=benbench_eval_cfg
                    )
                )
