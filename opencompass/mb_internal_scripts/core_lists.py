six_hours = [
    'humaneval_openai_sample_evals_gen_dcae0e',
    'sanitized_mbpp_mdblock_0shot_nomax_gen_2e2332',
    'livecodebench_v6_code_gen_o1_postprocess_gen_b5b6c5',
    
    'ceval_0shot_cot_fulldetail_nomax_gen_ac5ebc',
    'mmlu_redux_0shot_simple_evals_gen_fe2877',
    'aime2025_repeat16_gen_1b3f77',
    # 'aime2024_repeat16_mathverify_gen_1b3f77',
    'math_prm800k_500_0shot_cot_mathverify_gen_11c4b5',
    'IFEval_gen_353ae7',
    
    'bbh_0shot_nomaxtoken_gen_b5bdf1',  # 0shot
    'gpqa_openai_simple_evals_gen_5aeece',
    'ruler_32k_with_chat_template_gen_e8a78d',
]

twelve_hours = [
    'aime2024_repeat16_gen_1b3f77',
    # 'aime2024_repeat16_mathverify_gen_1b3f77',
    'gsm8k_0shot_v2_gen_17d799',
    
    'alignbench_v1_1_llmjudge_qwen2_5_72b_gen_5ddc62',
    'bbeh_nomax_gen_86c3a0',
    
    'livecodebench_v5_code_gen_o1_postprocess_gen_b5b6c5',
    'multiple_top_ten_internal_gen_f44aaf',
    
    'cmmlu_0shot_fulldetail_gen_78148a',
    'mmlu_openai_simple_evals_fulldetail_nomax_gen_b618ea',
    'simpleqa_llmjudge_qwen2_5_72b_gen_64c17d',
]

twenty_four_hours = [
    'zebra_logic_gen_083ddf',
    'CNMO2024_greedy_gen_353ae7',
    
    'rbench_gen_544610',
    'drop_openai_simple_evals_gen_3857b0',
    'lawbench_one_shot_gen_002588',
    'flores_gen_2697d7',
    
    'bigcodebench_hard_v1_2_internal_gen_7c886e',
    
    'mb_safety_0shot_llmjudge_qwen2_5_72b_gen_ec22ab',
    'supergpqa_gen_12b8bc',
    'mmlu_pro_0shot_cot_fulldetail_gen_08c1de',
    'ruler_32k_gen_e8a78d',
    'arena_hard_v2_qwen2_5_72b_gen_b6417a',
    'ruler_64k_gen_e8a78d',
    
    # 'aime2025_repeat16_gen_1b3f77',
    # 'aime2024_repeat16_gen_1b3f77',
    # 'lawbench_zero_shot_gen_002588',
    # 'bigcodebench_hard_v1_4_internal_gen_c3d5ad',
    # 'ruler_64k_with_chat_template_gen_e8a78d',
]

p0s = six_hours + twelve_hours + twenty_four_hours

p1s = [
    'gsmhard_0shot_nomax_gen_17d799',
]

fulls = p0s + p1s

# 预设 base 评测集合
base_datasets = [
    'internal_humaneval_gen_d2537e',
    'sanitized_mbpp_gen_742f0c',
    'ceval_ppl_578f8d',
    'cmmlu_ppl_041cbf',
    'mmlu_ppl_ac766d',
    'math_500_4shot_base_gen_db136b',
    'gsm8k_gen_17d0dc',
    'bbh_gen_98fba6',
    'ARC_c_ppl_a450bd',
    'ARC_e_ppl_a450bd',
    'gpqa_ppl_6bf57a',

    # 如果使用0shot的情况，可使用以下几个config
    # 'ceval_internal_0shot_ppl_1d28a4',
    # 'cmmlu_0shot_ppl_835fb0',
    # 'mmlu_0shot_ppl_7be85f',

    # 部分数据集的完整版但不推荐
    # 'math_4shot_base_gen_db136b',  # 完整版math
]

# 预设 sft 评测集合
sft_datasets = [
    'humaneval_openai_sample_evals_gen_dcae0e',
    'sanitized_mbpp_mdblock_0shot_nomax_gen_2e2332',
    'IFEval_gen_353ae7',

    # 'math_prm800k_500_0shot_cot_gen_7b6333',  # DeepSeek prompt
    'math_prm800k_500_0shot_cot_v2_gen_11c4b5',
    'gsm8k_0shot_v2_gen_17d799',
    'bbh_0shot_nomaxtoken_gen_b5bdf1',  # 0shot
    'ceval_0shot_cot_fulldetail_nomax_gen_ac5ebc',
    'cmmlu_0shot_fulldetail_gen_78148a',
    'mmlu_redux_0shot_simple_evals_gen_fe2877',
    'mmlu_openai_simple_evals_fulldetail_nomax_gen_b618ea',

    # 如果要使用few-shot，可使用以下几个config
    # 'ceval_fulldetail_gen_5f30c7',
    # 'cmmlu_gen_c13365',
    # 'mmlu_gen_4d595a',
    # 'bbh_gen_ee62e9',  # 3shot 目前不少模型评bbh还是需要3shot
    # 'sanitized_mbpp_mdblock_gen_a447ff',
    # 'gsm8k_4shot_nomax_gen_4fb824',

    # 部分数据集的完整版但不推荐
    'math_0shot_gen_11c4b5',
    # 'mbpp_gen_830460',
    # 'mbpp_full_0shot_mdblock_nomaxtoken_gen_22778f'

    # 已对齐，但是较少测试
    # 'hellaswag_10shot_gen_e42710',
]

deep_think = [
    # 重复16次版本
    'aime2024_repeat16_gen_1b3f77',
    'aime2025_repeat16_gen_1b3f77',
    'math_prm800k_500_0shot_cot_gen_7b6333',
    'livecodebench_v5_code_gen_o1_postprocess_gen_b5b6c5',
    'livecodebench_v6_code_gen_o1_postprocess_gen_b5b6c5',
    'gpqa_openai_simple_evals_gen_5aeece',
    'CNMO2024_greedy_gen_353ae7',

    # 单次测试
    'aime2024_gen_f505ab',
    'aime2025_gen_f505ab',
]

lcbs = [
    # 'livecodebench_v3_code_gen_o1_postprocess_gen_b5b6c5',
    # 'livecodebench_v4_code_gen_o1_postprocess_gen_b5b6c5',
    'livecodebench_v5_code_gen_o1_postprocess_gen_b5b6c5',
    'livecodebench_v6_code_gen_o1_postprocess_gen_b5b6c5',
]

maths = [
    'aime2025_repeat16_gen_1b3f77',
    'aime2024_repeat16_gen_1b3f77',
    'math_prm800k_500_0shot_cot_mathverify_gen_11c4b5',
    'CNMO2024_greedy_gen_353ae7',
]

codes = [
    'livecodebench_v6_code_gen_o1_postprocess_gen_b5b6c5',
    'livecodebench_v5_code_gen_o1_postprocess_gen_b5b6c5',
    'humaneval_openai_sample_evals_gen_dcae0e',
    'sanitized_mbpp_mdblock_0shot_nomax_gen_2e2332',
    'multiple_top_ten_internal_gen_f44aaf',
    'bigcodebench_hard_v1_2_internal_gen_7c886e',
]

subjectives = [
    'alignbench_v1_1_llmjudge_gen_5ddc62',
]

contaminations = [
    'benbench_coredataset_5gram_gen_1429a7',
    'benbench_coredataset_10gram_gen_e3d3cf',
    'benbench_gsm8k_test_5gram_gen_9a7d6c',

    # 'benbench_math_5gram_gen_24f487',
    # 'benbench_math_10gram_gen_a785f0',
    # 'benbench_mcq_5gram_gen_05cd8d',
    # 'benbench_mcq_10gram_gen_2cc931',
]

datasets_map = {
    'base': base_datasets,
    'sft': sft_datasets,
    'deep': deep_think,
    'lcb': lcbs,
    'contamination': contaminations,
    '6h': six_hours,
    '12h': twelve_hours,
    '24h': twenty_four_hours,
    'full': fulls,
    'p0': p0s,
    'p1': p1s,
    'subjectives': subjectives,
    'contaminations': contaminations,
    'math': maths,
    'code': codes,
}


qwen3s = [
    ['vllm_qwen3_nothink_sft', 'Qwen/Qwen3-4B-Instruct-2507'],
    ['vllm_qwen3_think_sft', 'Qwen/Qwen3-4B-Thinking-2507'],

    ['vllm_qwen3_nothink_sft', 'Qwen/Qwen3-4B'],
    ['vllm_qwen3_think_sft', 'Qwen/Qwen3-4B'],

    ['vllm_qwen3_nothink_sft', 'Qwen/Qwen3-8B'],
    ['vllm_qwen3_think_sft', 'Qwen/Qwen3-8B'],

    ['vllm_qwen3_nothink_sft', 'Qwen/Qwen3-14B'],
    ['vllm_qwen3_think_sft', 'Qwen/Qwen3-14B'],

    ['vllm_qwen3_nothink_sft', 'Qwen/Qwen3-0.6B'],
    ['vllm_qwen3_think_sft', 'Qwen/Qwen3-0.6B'],

    ['vllm_qwen3_nothink_sft', 'Qwen/Qwen3-1.7B'],
    ['vllm_qwen3_think_sft', 'Qwen/Qwen3-1.7B'],

    ['vllm_general_sft', 'Qwen/Qwen2.5-7B-Instruct'],

    ['vllm_qwen3_nothink_sft_2gpu',
        'Qwen/Qwen3-30B-A3B-Instruct-2507'],
    ['vllm_qwen3_think_sft_2gpu', 'Qwen/Qwen3-30B-A3B-Thinking-2507'],
    
    ['vllm_qwen3_nothink_sft_2gpu',
        'Qwen/Qwen3-30B-A3B'],
    ['vllm_qwen3_think_sft_2gpu', 'Qwen/Qwen3-30B-A3B'],
]

models_map = {
    'qwen3': qwen3s
}
