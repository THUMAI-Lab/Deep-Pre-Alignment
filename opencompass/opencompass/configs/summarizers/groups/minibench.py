ceval_minibench_summary_groups = []

_ceval_stem = ['computer_network', 'operating_system', 'college_programming', 'college_physics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'high_school_mathematics', 'high_school_physics', 'middle_school_biology', 'veterinary_medicine']
_ceval_stem = ['ceval_minibench-' + s for s in _ceval_stem]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-stem', 'subsets': _ceval_stem})

_ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'education_science', 'teacher_qualification', 'high_school_politics', 'middle_school_politics', 'middle_school_geography']
_ceval_social_science = ['ceval_minibench-' + s for s in _ceval_social_science]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-social-science', 'subsets': _ceval_social_science})

_ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
_ceval_humanities = ['ceval_minibench-' + s for s in _ceval_humanities]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-humanities', 'subsets': _ceval_humanities})

_ceval_other = ['civil_servant', 'plant_protection', 'basic_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
_ceval_other = ['ceval_minibench-' + s for s in _ceval_other]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-other', 'subsets': _ceval_other})

_ceval_hard = ['discrete_mathematics', 'probability_and_statistics', 'college_physics', 'high_school_mathematics', 'high_school_physics']
_ceval_hard = ['ceval_minibench-' + s for s in _ceval_hard]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-hard', 'subsets': _ceval_hard})

_ceval_all = _ceval_stem + _ceval_social_science + _ceval_humanities + _ceval_other
ceval_minibench_summary_groups.append({'name': 'ceval', 'subsets': _ceval_all})

_ceval_stem = ['computer_network', 'operating_system', 'college_programming', 'college_physics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'high_school_mathematics', 'high_school_physics', 'middle_school_biology', 'veterinary_medicine']
_ceval_stem = ['ceval_minibench-test-' + s for s in _ceval_stem]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-test-stem', 'subsets': _ceval_stem})

_ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'education_science', 'teacher_qualification', 'high_school_politics', 'middle_school_politics', 'middle_school_geography']
_ceval_social_science = ['ceval_minibench-test-' + s for s in _ceval_social_science]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-test-social-science', 'subsets': _ceval_social_science})

_ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
_ceval_humanities = ['ceval_minibench-test-' + s for s in _ceval_humanities]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-test-humanities', 'subsets': _ceval_humanities})

_ceval_other = ['civil_servant', 'plant_protection', 'basic_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
_ceval_other = ['ceval_minibench-test-' + s for s in _ceval_other]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-test-other', 'subsets': _ceval_other})

_ceval_hard = ['discrete_mathematics', 'probability_and_statistics', 'college_physics', 'high_school_mathematics', 'high_school_physics']
_ceval_hard = ['ceval_minibench-test-' + s for s in _ceval_hard]
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-test-hard', 'subsets': _ceval_hard})

_ceval_all = _ceval_stem + _ceval_social_science + _ceval_humanities + _ceval_other
ceval_minibench_summary_groups.append({'name': 'ceval_minibench-test', 'subsets': _ceval_all})

mmlu_minibench_summary_groups = []

_mmlu_minibench_humanities = ['high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions']
_mmlu_minibench_humanities = ['lukaemon_mmlu_minibench_' + s for s in _mmlu_minibench_humanities]
mmlu_minibench_summary_groups.append({'name': 'mmlu_minibench-humanities', 'subsets': _mmlu_minibench_humanities})

_mmlu_minibench_stem = ['anatomy', 'astronomy', 'college_physics', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics']
_mmlu_minibench_stem = ['lukaemon_mmlu_minibench_' + s for s in _mmlu_minibench_stem]
mmlu_minibench_summary_groups.append({'name': 'mmlu_minibench-stem', 'subsets': _mmlu_minibench_stem})

_mmlu_minibench_social_science = ['econometrics', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'professional_psychology', 'sociology', 'us_foreign_policy']
_mmlu_minibench_social_science = ['lukaemon_mmlu_minibench_' + s for s in _mmlu_minibench_social_science]
mmlu_minibench_summary_groups.append({'name': 'mmlu_minibench-social-science', 'subsets': _mmlu_minibench_social_science})

_mmlu_minibench_other = ['business_ethics', 'clinical_knowledge', 'college_medicine', 'human_aging', 'marketing', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine', 'virology']
_mmlu_minibench_other = ['lukaemon_mmlu_minibench_' + s for s in _mmlu_minibench_other]
mmlu_minibench_summary_groups.append({'name': 'mmlu_minibench-other', 'subsets': _mmlu_minibench_other})

_mmlu_minibench_all = _mmlu_minibench_humanities + _mmlu_minibench_stem + _mmlu_minibench_social_science + _mmlu_minibench_other
mmlu_minibench_summary_groups.append({'name': 'mmlu_minibench', 'subsets': _mmlu_minibench_all})


subcategories = {
    'agronomy': ['other'],
    'anatomy': ['biology'],
    'ancient_chinese': ['linguistics','china specific'],
    'arts': ['arts'],
    'astronomy': ['physics'],
    'business_ethics': ['business'],
    'chinese_civil_service_exam': ['politics','china specific'],
    'chinese_driving_rule': ['other','china specific'],
    'chinese_food_culture': ['culture','china specific'],
    'chinese_foreign_policy': ['politics','china specific'],
    'chinese_history':['history','china specific'],
    # 'chinese_literature': ['literature','china specific'],
    # 'chinese_teacher_qualification': ['education','china specific'],
    # 'college_actuarial_science':['math'],
    'college_education':['education'],
    'college_engineering_hydrology': ['engineering'],
    'college_law': ['law'],
    'college_mathematics': ['math'],
    # 'college_medical_statistics':['statistics'],
    'clinical_knowledge': ['other'],
    'college_medicine': ['other'],
    'computer_science': ['computer science'],
    'computer_security': ['other'],
    'conceptual_physics': ['physics'],
    # 'construction_project_management': ['other','china specific'],
    'economics': ['economics'],
    'education': ['education'],
    'elementary_chinese':['linguistics','china specific'],
    'elementary_commonsense':['other','china specific'],
    # 'elementary_information_and_technology': ['other'],
    'electrical_engineering': ['engineering'],
    'elementary_mathematics': ['math'],
    'ethnology': ['culture','china specific'],
    'food_science': ['other'],
    'genetics': ['biology'],
    'global_facts': ['global'],
    'high_school_biology': ['biology'],
    'high_school_chemistry': ['chemistry'],
    'high_school_geography': ['geography'],
    'high_school_mathematics': ['math'],
    # 'high_school_physics': ['physics'],
    'high_school_politics': ['politics','china specific'],
    # 'human_sexuality': ['other'],
    'international_law': ['law'],
    # 'journalism': ['sociology'],
    'jurisprudence': ['law'],
    'legal_and_moral_basis': ['other'],
    # 'logical': ['philosophy'],
    'machine_learning': ['computer science'],
    'management': ['business'],
    'marketing': ['business'],
    'marxist_theory': ['philosophy'],
    'modern_chinese': ['linguistics','china specific'],
    'nutrition': ['other'],
    # 'philosophy': ['philosophy'],
    'professional_accounting': ['business'],
    'professional_law': ['law'],
    'professional_medicine': ['other'],
    'professional_psychology': ['psychology'],
    # 'public_relations': ['politics'],
    'security_study': ['politics'],
    'sociology': ['culture'],
    'sports_science': ['other'],
    'traditional_chinese_medicine': ['other','china specific'],
    'virology': ['biology'],
    'world_history':['history'],
    'world_religions': ['global'],
}

categories = {
    'STEM': ['physics', 'chemistry', 'biology', 'computer science', 'math', 'engineering', 'statistics'],
    'Humanities': ['history', 'philosophy', 'law', 'arts', 'literature', 'global'],
    'Social Science': ['linguistics','business', 'politics', 'culture', 'economics', 'geography', 'psychology', 'education', 'sociology'],
    'Other':['other'],
    'China specific': ['china specific'],
}

category2subject = {}
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                category2subject.setdefault(k, []).append(subject)

cmmlu_minibench_summary_groups = []

_cmmlu_minibench_humanities = ['cmmlu_minibench-' + s for s in category2subject['Humanities']]
cmmlu_minibench_summary_groups.append({'name': 'cmmlu_minibench-humanities', 'subsets': _cmmlu_minibench_humanities})

_cmmlu_minibench_stem = ['cmmlu_minibench-' + s for s in category2subject['STEM']]
cmmlu_minibench_summary_groups.append({'name': 'cmmlu_minibench-stem', 'subsets': _cmmlu_minibench_stem})

_cmmlu_minibench_social_science = ['cmmlu_minibench-' + s for s in category2subject['Social Science']]
cmmlu_minibench_summary_groups.append({'name': 'cmmlu_minibench-social-science', 'subsets': _cmmlu_minibench_social_science})

_cmmlu_minibench_other = ['cmmlu_minibench-' + s for s in category2subject['Other']]
cmmlu_minibench_summary_groups.append({'name': 'cmmlu_minibench-other', 'subsets': _cmmlu_minibench_other})

_cmmlu_minibench_china_specific = ['cmmlu_minibench-' + s for s in category2subject['China specific']]
cmmlu_minibench_summary_groups.append({'name': 'cmmlu_minibench-china-specific', 'subsets': _cmmlu_minibench_china_specific})

_cmmlu_minibench_all = ['cmmlu_minibench-' + s for s in subcategories.keys()]
cmmlu_minibench_summary_groups.append({'name': 'cmmlu_minibench', 'subsets': _cmmlu_minibench_all})
