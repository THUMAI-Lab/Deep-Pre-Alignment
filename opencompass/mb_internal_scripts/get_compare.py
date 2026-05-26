import glob
import json
import os

import pandas as pd
from tqdm import tqdm


def read_jsonl(jsonl_path):
    df = pd.read_json(jsonl_path, lines=True)
    data_list = df.to_dict(orient='records')
    return data_list


def save_json(js_content, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        # for line in js_content:
        f.write(json.dumps(js_content, ensure_ascii=False, indent=4))
    return


folder_a_list = [
    'outputs/api_ark_deepseek_r1/20250206_202912-api_ark_deepseek_r1-aime2024_gen_6e39a4',
    'outputs/api_ark_deepseek_r1/20250223_112715-api_ark_deepseek_r1-aime2025_gen_6e39a4',
    'outputs/api_ark_deepseek_r1/20250225_172303-api_ark_deepseek_r1-gsm8k_0shot_v2_minibench_gen_6e39a4',
    'outputs/api_ark_deepseek_r1/20250209_125608-api_ark_deepseek_r1-math_0shot_gen_393424',
]

folder_b_list = [
    'outputs/llamacpp_deepseek_8080/20250220_105335-deepseek_llamacpp-aime2024_gen_6e39a4',
    'outputs/llamacpp_deepseek_8081/20250222_135353-deepseek_llamacpp-aime2025_gen_6e39a4',
    'outputs/llamacpp_deepseek_8080/20250223_135751-deepseek_llamacpp-gsm8k_0shot_v2_minibench_gen_6e39a4',
    'outputs/llamacpp_deepseek_8081/20250221_174824-deepseek_llamacpp-math_prm800k_500_0shot_cot_gen'
]

output_root = 'outputs/compare/'


def read_json_file(file_path):
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


for a, b in zip(folder_a_list, folder_b_list):
    # outputs/llamacpp_deepseek_8081/20250221_174824-deepseek_llamacpp-math_prm800k_500_0shot_cot_gen/results/llamacpp_DeepSeek-R1-UD-IQ1_S/math_prm800k_500.json
    result_folder_a = os.path.join(a, 'results')
    result_folder_b = os.path.join(b, 'results')
    a_result_path = glob.glob(os.path.join(result_folder_a, '**',
                                           '*.json'))[-1]
    b_result_path = glob.glob(os.path.join(result_folder_b, '**',
                                           '*.json'))[-1]
    print(a_result_path)
    print(b_result_path)

    a_results = read_json_file(a_result_path)
    b_results = read_json_file(b_result_path)

    a_reason_path = os.path.join(a, 'reasoning', 'reasoning.jsonl')
    b_reason_path = os.path.join(b, 'reasoning', 'reasoning.jsonl')

    a_reason = []
    b_reason = []

    if os.path.exists(a_reason_path) and os.path.exists(b_reason_path):
        a_reason = read_jsonl(a_reason_path)
        b_reason = read_jsonl(b_reason_path)

    output_folder = os.path.join(
        output_root, f'{os.path.basename(a)}_with_{os.path.basename(b)}')
    # print(os.path.dirname(a))
    # print(os.path.dirname(b))
    os.makedirs(output_folder, exist_ok=True)

    compare_res = {
        'ALL_RIGHT': {},
        'ALL_WRONG': {},
        'A_RIGHT_B_WRONG': {},
        'A_WRONG_B_RIGHT': {},
    }

    # for res, value in {
    #     "ALL_RIGHT": [True, True],
    #     "ALL_WRONG": [False, False],
    #     "A_RIGHT_B_WRONG": [True, False],
    #     "A_RIGHT_B_WRONG": [False, True],
    # }.items():

    for idx_a, (a_, a_info) in enumerate(tqdm(a_results['details'].items())):
        for idx_b, (b_, b_info) in enumerate(b_results['details'].items()):
            if a_info['prompt'] == b_info['prompt']:

                reasoning_a = ''
                reasoning_b = ''

                compare_info = {
                    'name_a': os.path.basename(a),
                    'name_b': os.path.basename(b),
                    'prompt': a_info['prompt'],
                    'reasoning_a': '',
                    'reasoning_b': '',
                    'origin_prediction_a': a_info['origin_prediction'],
                    'origin_prediction_b': b_info['origin_prediction'],
                    'predictions_a': a_info['predictions'],
                    'predictions_b': b_info['predictions'],
                    'references': a_info['references'],
                    'correct_a': a_info['correct'],
                    'correct_b': b_info['correct'],
                }

                if a_reason and b_reason:
                    for r_a in a_reason:
                        # print(r_a)
                        if isinstance(r_a['prompt'], list):
                            r_a_prompt = r_a['prompt'][0]['content']
                        else:
                            r_a_prompt = r_a['prompt'].replace(
                                '<｜User｜>', '').replace('<｜Assistant｜>', '')
                        if a_info['prompt'] == r_a_prompt:
                            reasoning_a = r_a['reasoning']
                            # print("yesa")
                            break

                    for r_b in b_reason:
                        if isinstance(r_b['prompt'], list):
                            r_b_prompt = r_b['prompt'][0]['content']
                        else:
                            r_b_prompt = r_b['prompt'].replace(
                                '<｜User｜>', '').replace('<｜Assistant｜>', '')
                        # print("-" * 100)
                        # print(b_info["prompt"])
                        # print("-" * 100)
                        # print(r_b_prompt)
                        if b_info['prompt'] == r_b_prompt:
                            reasoning_b = r_b['reasoning']
                            # print("yesb")
                            break

                if reasoning_a and reasoning_b:
                    compare_info['reasoning_a'] = reasoning_a
                    compare_info['reasoning_b'] = reasoning_b
                else:
                    compare_info.pop('reasoning_a')
                    compare_info.pop('reasoning_b')

                if a_info['correct'] == True and b_info['correct'] == True:
                    compare_res['ALL_RIGHT'][a_] = compare_info
                elif a_info['correct'] == False and b_info['correct'] == False:
                    compare_res['ALL_WRONG'][a_] = compare_info
                elif a_info['correct'] == True and b_info['correct'] == False:
                    compare_res['A_RIGHT_B_WRONG'][a_] = compare_info
                elif a_info['correct'] == False and b_info['correct'] == True:
                    compare_res['A_WRONG_B_RIGHT'][a_] = compare_info
                break

    for compare_type in compare_res:
        save_path = os.path.join(output_folder, f'{compare_type}.json')
        print(save_path)
        save_json(compare_res[compare_type], save_path)
