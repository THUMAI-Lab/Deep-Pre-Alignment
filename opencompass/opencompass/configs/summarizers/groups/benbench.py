benbench_summary_groups = []

bbh_datasets = [
    'boolean_expressions',
    'causal_judgement',
    'date_understanding',
    'disambiguation_qa',
    'dyck_languages',
    'formal_fallacies',
    'geometric_shapes',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'logical_deduction_three_objects',
    'movie_recommendation',
    'multistep_arithmetic_two',
    'navigate',
    'object_counting',
    'penguins_in_a_table',
    'reasoning_about_colored_objects',
    'ruin_names',
    'salient_translation_error_detection',
    'snarks',
    'sports_understanding',
    'temporal_sequences',
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects',
    'web_of_lies',
    'word_sorting',
]

_mmlu_weights = {'college_biology': 144,'college_chemistry': 100,'college_computer_science': 100,'college_mathematics': 100,'college_physics': 102,'electrical_engineering': 145,'astronomy': 152,'anatomy': 135,'abstract_algebra': 100,'machine_learning': 112,'clinical_knowledge': 265,'global_facts': 100,'management': 103,'nutrition': 306,'marketing': 234,'professional_accounting': 282,'high_school_geography': 198,'international_law': 121,'moral_scenarios': 895,'computer_security': 100,'high_school_microeconomics': 238,'professional_law': 1534,'medical_genetics': 100,'professional_psychology': 612,'jurisprudence': 108,'world_religions': 171,'philosophy': 311,'virology': 166,'high_school_chemistry': 203,'public_relations': 110,'high_school_macroeconomics': 390,'human_sexuality': 131,'elementary_mathematics': 378,'high_school_physics': 151,'high_school_computer_science': 100,'high_school_european_history': 165,'business_ethics': 100,'moral_disputes': 346,'high_school_statistics': 216,'miscellaneous': 783,'formal_logic': 126,'high_school_government_and_politics': 193,'prehistory': 324,'security_studies': 245,'high_school_biology': 310,'logical_fallacies': 163,'high_school_world_history': 237,'professional_medicine': 272,'high_school_mathematics': 270,'college_medicine': 173,'high_school_us_history': 204,'sociology': 201,'econometrics': 114,'high_school_psychology': 545,'human_aging': 223,'us_foreign_policy': 100,'conceptual_physics': 235}

ceval_subject_mapping = {
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
ceval_sets = list(ceval_subject_mapping.keys())

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
    # 'journalism': '新闻学',
    # 'jurisprudence': '法理学',
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
_cmmlu_sets = list(cmmlu_subject_mapping.keys())

_bbh_ngram_weights = {
    'bbh-test-temporal_sequences-origin': 1250,
    'bbh-test-disambiguation_qa-origin': 1250,
    'bbh-test-date_understanding-origin': 1250,
    'bbh-test-tracking_shuffled_objects_three_objects-origin': 1250,
    'bbh-test-penguins_in_a_table-origin': 730,
    'bbh-test-geometric_shapes-origin': 1250,
    'bbh-test-snarks-origin': 890,
    'bbh-test-ruin_names-origin': 1250,
    'bbh-test-tracking_shuffled_objects_seven_objects-origin': 1250,
    'bbh-test-tracking_shuffled_objects_five_objects-origin': 1250,
    'bbh-test-logical_deduction_three_objects-origin': 1250,
    'bbh-test-hyperbaton-origin': 1250,
    'bbh-test-logical_deduction_five_objects-origin': 1250,
    'bbh-test-logical_deduction_seven_objects-origin': 1250,
    'bbh-test-movie_recommendation-origin': 1250,
    'bbh-test-salient_translation_error_detection-origin': 1250,
    'bbh-test-reasoning_about_colored_objects-origin': 1250,
    'bbh-test-multistep_arithmetic_two-origin': 1250,
    'bbh-test-navigate-origin': 1250,
    'bbh-test-dyck_languages-origin': 1250,
    'bbh-test-word_sorting-origin': 1250,
    'bbh-test-sports_understanding-origin': 1250,
    'bbh-test-boolean_expressions-origin': 1250,
    'bbh-test-object_counting-origin': 1250,
    'bbh-test-formal_fallacies-origin': 1250,
    'bbh-test-causal_judgement-origin': 935,
    'bbh-test-web_of_lies-origin': 1250,
}

_ceval_ngram_weights = {
    'ceval-test-computer_network-origin': 95,
    'ceval-test-operating_system-origin': 95,
    'ceval-test-computer_architecture-origin': 105,
    'ceval-test-college_programming-origin': 185,
    'ceval-test-college_physics-origin': 95,
    'ceval-test-college_chemistry-origin': 120,
    'ceval-test-advanced_mathematics-origin': 95,
    'ceval-test-probability_and_statistics-origin': 90,
    'ceval-test-discrete_mathematics-origin': 80,
    'ceval-test-electrical_engineer-origin': 185,
    'ceval-test-metrology_engineer-origin': 120,
    'ceval-test-high_school_mathematics-origin': 90,
    'ceval-test-high_school_physics-origin': 95,
    'ceval-test-high_school_chemistry-origin': 95,
    'ceval-test-high_school_biology-origin': 95,
    'ceval-test-middle_school_mathematics-origin': 95,
    'ceval-test-middle_school_biology-origin': 105,
    'ceval-test-middle_school_physics-origin': 95,
    'ceval-test-middle_school_chemistry-origin': 100,
    'ceval-test-veterinary_medicine-origin': 115,
    'ceval-test-college_economics-origin': 275,
    'ceval-test-business_administration-origin': 165,
    'ceval-test-marxism-origin': 95,
    'ceval-test-mao_zedong_thought-origin': 120,
    'ceval-test-education_science-origin': 145,
    'ceval-test-teacher_qualification-origin': 220,
    'ceval-test-high_school_politics-origin': 95,
    'ceval-test-high_school_geography-origin': 95,
    'ceval-test-middle_school_politics-origin': 105,
    'ceval-test-middle_school_geography-origin': 60,
    'ceval-test-modern_chinese_history-origin': 115,
    'ceval-test-ideological_and_moral_cultivation-origin': 95,
    'ceval-test-logic-origin': 110,
    'ceval-test-law-origin': 120,
    'ceval-test-chinese_language_and_literature-origin': 115,
    'ceval-test-art_studies-origin': 165,
    'ceval-test-professional_tour_guide-origin': 145,
    'ceval-test-legal_professional-origin': 115,
    'ceval-test-high_school_chinese-origin': 95,
    'ceval-test-high_school_history-origin': 100,
    'ceval-test-middle_school_history-origin': 110,
    'ceval-test-civil_servant-origin': 235,
    'ceval-test-sports_science-origin': 95,
    'ceval-test-plant_protection-origin': 110,
    'ceval-test-basic_medicine-origin': 95,
    'ceval-test-clinical_medicine-origin': 110,
    'ceval-test-urban_and_rural_planner-origin': 230,
    'ceval-test-accountant-origin': 245,
    'ceval-test-fire_engineer-origin': 155,
    'ceval-test-environmental_impact_assessment_engineer-origin': 155,
    'ceval-test-tax_accountant-origin': 245,
    'ceval-test-physician-origin': 245
}

_cmmlu_ngram_weights = {
    'cmmlu-test-agronomy-origin': 845,
    'cmmlu-test-anatomy-origin': 740,
    'cmmlu-test-ancient_chinese-origin': 820,
    'cmmlu-test-arts-origin': 800,
    'cmmlu-test-astronomy-origin': 825,
    'cmmlu-test-business_ethics-origin': 1045,
    'cmmlu-test-chinese_civil_service_exam-origin': 800,
    'cmmlu-test-chinese_driving_rule-origin': 655,
    'cmmlu-test-chinese_food_culture-origin': 680,
    'cmmlu-test-chinese_foreign_policy-origin': 535,
    'cmmlu-test-chinese_history-origin': 1615,
    'cmmlu-test-chinese_literature-origin': 1020,
    'cmmlu-test-chinese_teacher_qualification-origin': 895,
    'cmmlu-test-clinical_knowledge-origin': 1185,
    'cmmlu-test-college_actuarial_science-origin': 530,
    'cmmlu-test-college_education-origin': 535,
    'cmmlu-test-college_engineering_hydrology-origin': 530,
    'cmmlu-test-college_law-origin': 540,
    'cmmlu-test-college_mathematics-origin': 525,
    'cmmlu-test-college_medical_statistics-origin': 530,
    'cmmlu-test-college_medicine-origin': 1365,
    'cmmlu-test-computer_science-origin': 1020,
    'cmmlu-test-computer_security-origin': 855,
    'cmmlu-test-conceptual_physics-origin': 735,
    'cmmlu-test-construction_project_management-origin': 695,
    'cmmlu-test-economics-origin': 795,
    'cmmlu-test-education-origin': 815,
    'cmmlu-test-electrical_engineering-origin': 860,
    'cmmlu-test-elementary_chinese-origin': 1260,
    'cmmlu-test-elementary_commonsense-origin': 990,
    'cmmlu-test-elementary_information_and_technology-origin': 1190,
    'cmmlu-test-elementary_mathematics-origin': 1150,
    'cmmlu-test-ethnology-origin': 675,
    'cmmlu-test-food_science-origin': 715,
    'cmmlu-test-genetics-origin': 880,
    'cmmlu-test-global_facts-origin': 745,
    'cmmlu-test-high_school_biology-origin': 845,
    'cmmlu-test-high_school_chemistry-origin': 660,
    'cmmlu-test-high_school_geography-origin': 590,
    'cmmlu-test-high_school_mathematics-origin': 820,
    'cmmlu-test-high_school_physics-origin': 550,
    'cmmlu-test-high_school_politics-origin': 715,
    'cmmlu-test-human_sexuality-origin': 630,
    'cmmlu-test-international_law-origin': 925,
    'cmmlu-test-legal_and_moral_basis-origin': 1070,
    'cmmlu-test-logical-origin': 615,
    'cmmlu-test-machine_learning-origin': 610,
    'cmmlu-test-management-origin': 1050,
    'cmmlu-test-marketing-origin': 900,
    'cmmlu-test-marxist_theory-origin': 945,
    'cmmlu-test-modern_chinese-origin': 580,
    'cmmlu-test-nutrition-origin': 725,
    'cmmlu-test-philosophy-origin': 525,
    'cmmlu-test-professional_accounting-origin': 875,
    'cmmlu-test-professional_law-origin': 1055,
    'cmmlu-test-professional_medicine-origin': 1880,
    'cmmlu-test-professional_psychology-origin': 1160,
    'cmmlu-test-public_relations-origin': 870,
    'cmmlu-test-security_study-origin': 675,
    'cmmlu-test-sociology-origin': 1130,
    'cmmlu-test-sports_science-origin': 825,
    'cmmlu-test-traditional_chinese_medicine-origin': 925,
    'cmmlu-test-virology-origin': 845,
    'cmmlu-test-world_history-origin': 805,
    'cmmlu-test-world_religions-origin': 800,
}

_mmlu_ngram_weights = {
    'mmlu-test-college_biology-origin': 720,
    'mmlu-test-college_chemistry-origin': 500,
    'mmlu-test-college_computer_science-origin': 500,
    'mmlu-test-college_mathematics-origin': 500,
    'mmlu-test-college_physics-origin': 510,
    'mmlu-test-electrical_engineering-origin': 725,
    'mmlu-test-astronomy-origin': 760,
    'mmlu-test-anatomy-origin': 675,
    'mmlu-test-abstract_algebra-origin': 500,
    'mmlu-test-machine_learning-origin': 560,
    'mmlu-test-clinical_knowledge-origin': 1325,
    'mmlu-test-global_facts-origin': 500,
    'mmlu-test-management-origin': 515,
    'mmlu-test-nutrition-origin': 1530,
    'mmlu-test-marketing-origin': 1170,
    'mmlu-test-professional_accounting-origin': 1410,
    'mmlu-test-high_school_geography-origin': 990,
    'mmlu-test-international_law-origin': 605,
    'mmlu-test-moral_scenarios-origin': 4475,
    'mmlu-test-computer_security-origin': 500,
    'mmlu-test-high_school_microeconomics-origin': 1190,
    'mmlu-test-professional_law-origin': 7670,
    'mmlu-test-medical_genetics-origin': 500,
    'mmlu-test-professional_psychology-origin': 3060,
    'mmlu-test-jurisprudence-origin': 540,
    'mmlu-test-world_religions-origin': 855,
    'mmlu-test-philosophy-origin': 1555,
    'mmlu-test-virology-origin': 830,
    'mmlu-test-high_school_chemistry-origin': 1015,
    'mmlu-test-public_relations-origin': 550,
    'mmlu-test-high_school_macroeconomics-origin': 1950,
    'mmlu-test-human_sexuality-origin': 655,
    'mmlu-test-elementary_mathematics-origin': 1890,
    'mmlu-test-high_school_physics-origin': 755,
    'mmlu-test-high_school_computer_science-origin': 500,
    'mmlu-test-high_school_european_history-origin': 825,
    'mmlu-test-business_ethics-origin': 500,
    'mmlu-test-moral_disputes-origin': 1730,
    'mmlu-test-high_school_statistics-origin': 1080,
    'mmlu-test-miscellaneous-origin': 3915,
    'mmlu-test-formal_logic-origin': 630,
    'mmlu-test-high_school_government_and_politics-origin': 965,
    'mmlu-test-prehistory-origin': 1620,
    'mmlu-test-security_studies-origin': 1225,
    'mmlu-test-high_school_biology-origin': 1550,
    'mmlu-test-logical_fallacies-origin': 815,
    'mmlu-test-high_school_world_history-origin': 1185,
    'mmlu-test-professional_medicine-origin': 1360,
    'mmlu-test-high_school_mathematics-origin': 1350,
    'mmlu-test-college_medicine-origin': 865,
    'mmlu-test-high_school_us_history-origin': 1020,
    'mmlu-test-sociology-origin': 1005,
    'mmlu-test-econometrics-origin': 570,
    'mmlu-test-high_school_psychology-origin': 2725,
    'mmlu-test-human_aging-origin': 1115,
    'mmlu-test-us_foreign_policy-origin': 500,
    'mmlu-test-conceptual_physics-origin': 1175,
}

for n_gram in [5, 10, 15, 18, 20]:
    # GSM8K_origin test - origin train 差异
    benbench_summary_groups.append({
        'name': f'GSM8K_origin-test-train-{n_gram}gram-Δ', 
        'subsets': [f'GSM8K-origin-test-{n_gram}gram', f'GSM8K-origin-train-{n_gram}gram'], 
        'substraction': True
        })
    
    # GSM8K_rewritten 平均值
    benbench_summary_groups.append(
        {'name': f'GSM8K_rewritten-train-{n_gram}gram', 'subsets': [f'GSM8K_rewritten-train-1-{n_gram}gram', f'GSM8K_rewritten-train-2-{n_gram}gram', f'GSM8K_rewritten-train-3-{n_gram}gram']})
    
    benbench_summary_groups.append(
        {'name': f'GSM8K_rewritten-test-{n_gram}gram', 'subsets': [f'GSM8K_rewritten-test-1-{n_gram}gram', f'GSM8K_rewritten-test-2-{n_gram}gram', f'GSM8K_rewritten-test-3-{n_gram}gram']})
    
    # GSM8K_rewritten 标准差
    benbench_summary_groups.append(
        {'name': f'GSM8K_rewritten-train-{n_gram}gram-std', 'subsets': [f'GSM8K_rewritten-train-1-{n_gram}gram', f'GSM8K_rewritten-train-2-{n_gram}gram', f'GSM8K_rewritten-train-3-{n_gram}gram'], 'std': True})

    benbench_summary_groups.append(
        {'name': f'GSM8K_rewritten-test-{n_gram}gram-std', 'subsets': [f'GSM8K_rewritten-test-1-{n_gram}gram', f'GSM8K_rewritten-test-2-{n_gram}gram', f'GSM8K_rewritten-test-3-{n_gram}gram'], 'std': True})
    
    # GSM8K_rewritten - origin 差异
    benbench_summary_groups.append({
        'name': f'GSM8K_rewritten-origin-train-{n_gram}gram-Δ', 
        'subsets': [f'GSM8K_rewritten-train-{n_gram}gram', f'GSM8K-origin-train-{n_gram}gram'], 
        'substraction': True
        })
    
    benbench_summary_groups.append({
        'name': f'GSM8K_rewritten-origin-test-{n_gram}gram-Δ', 
        'subsets': [f'GSM8K_rewritten-test-{n_gram}gram', f'GSM8K-origin-test-{n_gram}gram'], 
        'substraction': True
        })
    
    # MATH origin test - origin train 差异
    benbench_summary_groups.append({
        'name': f'MATH_origin-test-train-{n_gram}gram-Δ', 
        'subsets': [f'MATH-origin-test-{n_gram}gram', f'MATH-origin-train-{n_gram}gram'], 
        'substraction': True
        })
    
    # MATH500 origin test - MATH500 origin test 差异
    benbench_summary_groups.append({
        'name': f'MATH_500-test-MATH-origin-test-{n_gram}gram-Δ', 
        'subsets': [f'MATH-origin-prm800k-500-test-{n_gram}gram', f'MATH-origin-test-{n_gram}gram'], 
        'substraction': True
        })
    
    #  MATH_rewritten 平均值
    benbench_summary_groups.append(
        {'name': f'MATH_rewritten-train-{n_gram}gram', 'subsets': [f'MATH_rewritten-train-1-{n_gram}gram', f'MATH_rewritten-train-2-{n_gram}gram', f'MATH_rewritten-train-3-{n_gram}gram']})
    
    benbench_summary_groups.append(
        {'name': f'MATH_rewritten-test-{n_gram}gram', 'subsets': [f'MATH_rewritten-test-1-{n_gram}gram', f'MATH_rewritten-test-2-{n_gram}gram', f'MATH_rewritten-test-3-{n_gram}gram']})
    
    # MATH_rewritten 标准差
    benbench_summary_groups.append(
        {'name': f'MATH_rewritten-train-{n_gram}gram-std', 'subsets': [f'MATH_rewritten-train-1-{n_gram}gram', f'MATH_rewritten-train-2-{n_gram}gram', f'MATH_rewritten-train-3-{n_gram}gram'], 'std': True})
    
    benbench_summary_groups.append(
        {'name': f'MATH_rewritten-test-{n_gram}gram-std', 'subsets': [f'MATH_rewritten-test-1-{n_gram}gram', f'MATH_rewritten-test-2-{n_gram}gram', f'MATH_rewritten-test-3-{n_gram}gram'], 'std': True})
    
    # MATH_rewritten - origin 差异
    benbench_summary_groups.append({
        'name': f'MATH_rewritten-origin-train-{n_gram}gram-Δ', 
        'subsets': [f'MATH_rewritten-train-{n_gram}gram', f'MATH-origin-train-{n_gram}gram'], 
        'substraction': True
        })
    
    benbench_summary_groups.append({
        'name': f'MATH_rewritten-origin-test-{n_gram}gram-Δ', 
        'subsets': [f'MATH_rewritten-test-{n_gram}gram', f'MATH-origin-test-{n_gram}gram'], 
        'substraction': True
        })
    
    # MCQ 标准差
    benbench_summary_groups.append(
        {'name': f'MMLU-origin-{n_gram}gram-std', 'subsets': [f'MMLU-origin-dev-{n_gram}gram', f'MMLU-origin-val-{n_gram}gram', f'MMLU-origin-test-{n_gram}gram'], 'std': True})
    
    benbench_summary_groups.append(
        {'name': f'CEVAL-origin-{n_gram}gram-std', 'subsets': [f'CEVAL-origin-dev-{n_gram}gram', f'CEVAL-origin-val-{n_gram}gram', f'CEVAL-origin-test-{n_gram}gram'], 'std': True})
    
    benbench_summary_groups.append(
        {'name': f'BBH-origin-{n_gram}gram', 'subsets': [f'BBH-{set_name}-origin-{n_gram}gram' for set_name in bbh_datasets]})
    
    # core datasets
    benbench_summary_groups.append(
        {'name': f'bbh-test-origin-{n_gram}gram', 'subsets': [f'bbh-test-{set_name}-origin-{n_gram}gram' for set_name in bbh_datasets]})
    
    benbench_summary_groups.append(
        {'name': f'bbh-test-origin-subsets-weighted-{n_gram}gram', 'subsets': [f'bbh-test-{set_name}-origin-{n_gram}gram' for set_name in bbh_datasets], 'weights': {f'{subset_name}-{n_gram}gram': _bbh_ngram_weights[subset_name] for subset_name in _bbh_ngram_weights.keys()}})
    
    benbench_summary_groups.append(
        {'name': f'ceval-test-origin-{n_gram}gram', 'subsets': [f'ceval-test-{set_name}-origin-{n_gram}gram' for set_name in ceval_sets]})
    
    benbench_summary_groups.append(
        {'name': f'ceval-test-origin-subsets-weighted-{n_gram}gram', 'subsets': [f'ceval-test-{set_name}-origin-{n_gram}gram' for set_name in ceval_sets], 'weights': {f'{subset_name}-{n_gram}gram': _ceval_ngram_weights[subset_name] for subset_name in _ceval_ngram_weights.keys()}})
    
    benbench_summary_groups.append(
        {'name': f'cmmlu-test-origin-{n_gram}gram', 'subsets': [f'cmmlu-test-{set_name}-origin-{n_gram}gram' for set_name in _cmmlu_sets]})
    
    benbench_summary_groups.append(
        {'name': f'cmmlu-test-origin-subsets-weighted-{n_gram}gram', 'subsets': [f'cmmlu-test-{set_name}-origin-{n_gram}gram' for set_name in _cmmlu_sets], 'weights': {f'{subset_name}-{n_gram}gram': _cmmlu_ngram_weights[subset_name] for subset_name in _cmmlu_ngram_weights.keys()}})

    benbench_summary_groups.append(
        {'name': f'mmlu-test-origin-{n_gram}gram', 'subsets': [f'mmlu-test-{set_name}-origin-{n_gram}gram' for set_name in _mmlu_weights.keys()]})
    
    benbench_summary_groups.append(
        {'name': f'mmlu-test-origin-weighted-{n_gram}gram', 'subsets': [f'mmlu-test-{set_name}-origin-{n_gram}gram' for set_name in _mmlu_weights.keys()], 'weights': {f'mmlu-test-{set_name}-origin-{n_gram}gram': _mmlu_weights[set_name] for set_name in _mmlu_weights.keys()}})
    
    benbench_summary_groups.append(
        {'name': f'mmlu-test-origin-subsets-weighted-{n_gram}gram', 'subsets': [f'mmlu-test-{set_name}-origin-{n_gram}gram' for set_name in _mmlu_weights.keys()], 'weights': {f'{subset_name}-{n_gram}gram': _mmlu_ngram_weights[subset_name] for subset_name in _mmlu_ngram_weights.keys()}})
    
    
    
    # MATH-500 没有rewritten, 待官方更新
    # benbench_summary_groups.append(
    #     {'name': f'MATH_rewritten-prm800k-500-test-{n_gram}gram-std', 'subsets': [f'MATH_rewritten-prm800k-500-test-1-{n_gram}gram', f'MATH_rewritten-prm800k-500-test-2-{n_gram}gram', f'MATH_rewritten-prm800k-500-test-3-{n_gram}gram'], 'std': True})
    
    # benbench_summary_groups.append(
    #     {'name': f'MATH_rewritten-prm800k-500-test-{n_gram}gram', 'subsets': [f'MATH_rewritten-prm800k-500-test-1-{n_gram}gram', f'MATH_rewritten-prm800k-500-test-2-{n_gram}gram', f'MATH_rewritten-prm800k-500-test-3-{n_gram}gram']})
    
    # benbench_summary_groups.append(
        # {'name': f'MATH_rewritten-prm800k-500-train-{n_gram}gram-std', 'subsets': [f'MATH_rewritten-prm800k-500-train-1-{n_gram}gram', f'MATH_rewritten-prm800k-500-train-2-{n_gram}gram', f'MATH_rewritten-prm800k-500-train-3-{n_gram}gram'], 'std': True})