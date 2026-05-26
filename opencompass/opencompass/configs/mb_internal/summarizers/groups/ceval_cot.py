ceval_cot_summary_groups = []

_ceval_stem = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine']
_ceval_stem = ['ceval_cot-' + s for s in _ceval_stem]
ceval_cot_summary_groups.append({'name': 'ceval_cot-stem', 'subsets': _ceval_stem})

_ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography']
_ceval_social_science = ['ceval_cot-' + s for s in _ceval_social_science]
ceval_cot_summary_groups.append({'name': 'ceval_cot-social-science', 'subsets': _ceval_social_science})

_ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
_ceval_humanities = ['ceval_cot-' + s for s in _ceval_humanities]
ceval_cot_summary_groups.append({'name': 'ceval_cot-humanities', 'subsets': _ceval_humanities})

_ceval_other = ['civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
_ceval_other = ['ceval_cot-' + s for s in _ceval_other]
ceval_cot_summary_groups.append({'name': 'ceval_cot-other', 'subsets': _ceval_other})

_ceval_hard = ['advanced_mathematics', 'discrete_mathematics', 'probability_and_statistics', 'college_chemistry', 'college_physics', 'high_school_mathematics', 'high_school_chemistry', 'high_school_physics']
_ceval_hard = ['ceval_cot-' + s for s in _ceval_hard]
ceval_cot_summary_groups.append({'name': 'ceval_cot-hard', 'subsets': _ceval_hard})

_ceval_all = _ceval_stem + _ceval_social_science + _ceval_humanities + _ceval_other
ceval_cot_summary_groups.append({'name': 'ceval_cot', 'subsets': _ceval_all})

_ceval_stem = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine']
_ceval_stem = ['ceval_cot-test-' + s for s in _ceval_stem]
ceval_cot_summary_groups.append({'name': 'ceval_cot-test-stem', 'subsets': _ceval_stem})

_ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography']
_ceval_social_science = ['ceval_cot-test-' + s for s in _ceval_social_science]
ceval_cot_summary_groups.append({'name': 'ceval_cot-test-social-science', 'subsets': _ceval_social_science})

_ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
_ceval_humanities = ['ceval_cot-test-' + s for s in _ceval_humanities]
ceval_cot_summary_groups.append({'name': 'ceval_cot-test-humanities', 'subsets': _ceval_humanities})

_ceval_other = ['civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
_ceval_other = ['ceval_cot-test-' + s for s in _ceval_other]
ceval_cot_summary_groups.append({'name': 'ceval_cot-test-other', 'subsets': _ceval_other})

_ceval_hard = ['advanced_mathematics', 'discrete_mathematics', 'probability_and_statistics', 'college_chemistry', 'college_physics', 'high_school_mathematics', 'high_school_chemistry', 'high_school_physics']
_ceval_hard = ['ceval_cot-test-' + s for s in _ceval_hard]
ceval_cot_summary_groups.append({'name': 'ceval_cot-test-hard', 'subsets': _ceval_hard})

_ceval_all = _ceval_stem + _ceval_social_science + _ceval_humanities + _ceval_other
ceval_cot_summary_groups.append({'name': 'ceval_cot-test', 'subsets': _ceval_all})
