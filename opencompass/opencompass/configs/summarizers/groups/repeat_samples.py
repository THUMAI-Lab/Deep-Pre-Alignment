repeat_samples_summary_groups = []

for num_repeat in [4, 8, 16, 32, 64]:
    for dataset in ['aime2024', 'aime2025', 'math_prm800k_500', 'gsm8k', 'GPQA_diamond', 'mbpp', 'openai_humaneval', 'GPQA_diamond', 'lcb_code_generation_split_v4', 'lcb_code_generation_split_v5', 'lcb_code_generation_split_v6']:
        repeat_samples_summary_groups.append({
            'name': f'{dataset}_repeat{num_repeat}',
            'subsets': [f'{dataset}_{i}' for i in range(num_repeat)],
        })
