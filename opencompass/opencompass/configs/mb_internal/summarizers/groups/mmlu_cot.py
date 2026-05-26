mmlu_cot_summary_groups = []

_mmlu_cot_humanities = ['formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions']
_mmlu_cot_humanities = ['lukaemon_mmlu_cot_' + s for s in _mmlu_cot_humanities]
mmlu_cot_summary_groups.append({'name': 'mmlu_cot-humanities', 'subsets': _mmlu_cot_humanities})

_mmlu_cot_stem = ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning']
_mmlu_cot_stem = ['lukaemon_mmlu_cot_' + s for s in _mmlu_cot_stem]
mmlu_cot_summary_groups.append({'name': 'mmlu_cot-stem', 'subsets': _mmlu_cot_stem})

_mmlu_cot_social_science = ['econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy']
_mmlu_cot_social_science = ['lukaemon_mmlu_cot_' + s for s in _mmlu_cot_social_science]
mmlu_cot_summary_groups.append({'name': 'mmlu_cot-social-science', 'subsets': _mmlu_cot_social_science})

_mmlu_cot_other = ['business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts', 'human_aging', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine', 'virology']
_mmlu_cot_other = ['lukaemon_mmlu_cot_' + s for s in _mmlu_cot_other]
mmlu_cot_summary_groups.append({'name': 'mmlu_cot-other', 'subsets': _mmlu_cot_other})

_mmlu_cot_all = _mmlu_cot_humanities + _mmlu_cot_stem + _mmlu_cot_social_science + _mmlu_cot_other
_mmlu_cot_weights = {'college_biology': 144,'college_chemistry': 100,'college_computer_science': 100,'college_mathematics': 100,'college_physics': 102,'electrical_engineering': 145,'astronomy': 152,'anatomy': 135,'abstract_algebra': 100,'machine_learning': 112,'clinical_knowledge': 265,'global_facts': 100,'management': 103,'nutrition': 306,'marketing': 234,'professional_accounting': 282,'high_school_geography': 198,'international_law': 121,'moral_scenarios': 895,'computer_security': 100,'high_school_microeconomics': 238,'professional_law': 1534,'medical_genetics': 100,'professional_psychology': 612,'jurisprudence': 108,'world_religions': 171,'philosophy': 311,'virology': 166,'high_school_chemistry': 203,'public_relations': 110,'high_school_macroeconomics': 390,'human_sexuality': 131,'elementary_mathematics': 378,'high_school_physics': 151,'high_school_computer_science': 100,'high_school_european_history': 165,'business_ethics': 100,'moral_disputes': 346,'high_school_statistics': 216,'miscellaneous': 783,'formal_logic': 126,'high_school_government_and_politics': 193,'prehistory': 324,'security_studies': 245,'high_school_biology': 310,'logical_fallacies': 163,'high_school_world_history': 237,'professional_medicine': 272,'high_school_mathematics': 270,'college_medicine': 173,'high_school_us_history': 204,'sociology': 201,'econometrics': 114,'high_school_psychology': 545,'human_aging': 223,'us_foreign_policy': 100,'conceptual_physics': 235}
_mmlu_cot_weights = {'lukaemon_mmlu_cot_' + k : v for k,v in _mmlu_cot_weights.items()}
mmlu_cot_summary_groups.append({'name': 'mmlu_cot', 'subsets': _mmlu_cot_all})
mmlu_cot_summary_groups.append({'name': 'mmlu_cot-weighted', 'subsets': _mmlu_cot_all, 'weights': _mmlu_cot_weights})
