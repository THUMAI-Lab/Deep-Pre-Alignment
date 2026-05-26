gsm8k_contamination_summary_groups = []

# gsm8k-contamination
gsm8k_contamination_summary_groups.append({'name': 'gsm8k-train_test_substraction', 'subsets': ['gsm8k-test-ppl', 'gsm8k-train-ppl'], 'substraction': True})
gsm8k_contamination_summary_groups.append({'name': 'gsm8k-ref_test_substraction', 'subsets': ['gsm8k-ref-ppl', 'gsm8k-test-ppl'], 'substraction': True})
gsm8k_contamination_summary_groups.append({'name': 'gsm8k-ref_with_question_test_substraction', 'subsets': ['gsm8k-ref-with-question-ppl', 'gsm8k-test-ppl'], 'substraction': True})
