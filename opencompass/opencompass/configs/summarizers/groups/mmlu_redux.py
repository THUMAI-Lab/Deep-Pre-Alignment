_mmlu_redux_categories = ['anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'econometrics', 'electrical_engineering', 'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'human_aging', 'logical_fallacies', 'machine_learning', 'miscellaneous', 'philosophy', 'professional_accounting', 'public_relations', 'virology', 'conceptual_physics', 'high_school_us_history', 'astronomy', 'high_school_geography', 'high_school_macroeconomics', 'professional_law']

mmlu_redux_summary_groups = [
    {'name': 'mmlu_redux', 'subsets': ['mmlu_redux_' + c.replace(' ', '_') for c in _mmlu_redux_categories]},
]

mmlu_redux_summary_groups.append({'name': 'mmlu_redux', 'subsets': _mmlu_redux_categories})