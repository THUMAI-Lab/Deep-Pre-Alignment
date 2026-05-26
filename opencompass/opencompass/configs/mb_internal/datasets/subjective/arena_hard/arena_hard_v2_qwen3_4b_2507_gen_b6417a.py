from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ArenaHardV2Dataset, ArenaHardV2Evaluator, arenahard_v2_postprocess
from opencompass.models import OpenAI, TurboMindModelwithChatTemplate
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
import os
import torch

api_meta_template = dict(round=[
    dict(role='SYSTEM', api_role='SYSTEM'),
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# 修正输入字段为 'prompt'
subjective_reader_cfg = dict(
    input_columns=['prompt', 'baseline'],
    output_column='reference',
)

# 定义子数据集
subjective_all_sets = [
    # 'hard_prompt',
    'coding',
    'math',
    'creative_writing',
]

dataset_root = os.path.join(os.environ.get(
    'COMPASS_DATA_CACHE', './'), 'data/mb_internal/arena-hard-data/data/arena-hard-v2.0')

# 原始Arena Hard prompt
OG_ARENA_HARD_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."

# 创意写作专用prompt (去掉了"生成自己答案"的要求)
CREATIVE_WRITING_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nWhen evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."

JUDGE_SETTINGS = {
    'hard_prompt': {
        'baseline': 'o3-mini-2025-01-31',
        'system_prompt': OG_ARENA_HARD_PROMPT,
    },
    'coding': {
        'baseline': 'o3-mini-2025-01-31',
        'system_prompt': OG_ARENA_HARD_PROMPT,
    },
    'math': {
        'baseline': 'o3-mini-2025-01-31',
        'system_prompt': OG_ARENA_HARD_PROMPT,
    },
    'creative_writing': {
        'baseline': 'gemini-2.0-flash-001',
        'system_prompt': CREATIVE_WRITING_PROMPT,
    },
    'arena-hard-v0.1': {
        'baseline': 'gpt-4-0314',
        'system_prompt': OG_ARENA_HARD_PROMPT,
    },
}
# 评判prompt模板
judge_prompt = "<|User Prompt|>\n{QUESTION}\n\n<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n<|The End of Assistant B's Answer|>"

# 主数据集列表
arenahard_v2_datasets = []

# 为每个子数据集创建配置
for _name in subjective_all_sets:
    # 基准模型配置
    baseline = JUDGE_SETTINGS[_name]['baseline']

    # 推理配置
    subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}'
                ),
            ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    # 评估配置
    subjective_eval_cfg = dict(
        evaluator=dict(
            type=ArenaHardV2Evaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(
                            role='SYSTEM',
                            prompt=JUDGE_SETTINGS[_name]['system_prompt']
                        ),
                        dict(
                            role='HUMAN',
                            prompt=judge_prompt
                        )
                    ]),
            ),
            dataset_cfg=dict(
                abbr=f'{_name}-llmjudge',
                type=ArenaHardV2Dataset,
                path=dataset_root,
                baseline=baseline,
                name=_name,  # 使用子数据集名称作为name参数
                reader_cfg=subjective_reader_cfg,
            ),
            # judge_cfg=dict(
            #     abbr='gpt-4.1',
            #     type=OpenAI,
            #     path='gpt-4.1',
            #     key='ENV',  # 从环境变量获取API key
            #     meta_template=api_meta_template,
            #     query_per_second=16,
            #     max_out_len=16000,
            #     max_seq_len=65536,
            #     batch_size=1,
            #     temperature=0.0,
            #     verbose=True
            # ),
            judge_cfg=dict(
                type=TurboMindModelwithChatTemplate,
                abbr=f'qwen3_4b_2507-turbomind',
                path='Qwen/Qwen3-4B-Thinking-2507',
                engine_config=dict(session_len=131072,
                                   max_batch_size=32, tp=1),
                gen_config=dict(top_k=1, temperature=0.6,
                                top_p=0.95, max_new_tokens=16384),
                meta_template=api_meta_template,
                max_seq_len=131072,
                max_out_len=16384,
                batch_size=32,
                run_cfg=dict(num_gpus=1),
            ),
            dict_postprocessor=dict(type='arenahard_v2_postprocess'),
        ),
        pred_role='BOT',
    )

    # 添加到数据集列表
    arenahard_v2_datasets.append(
        dict(
            abbr=f'arena_hard_v2-{_name}',
            type=ArenaHardV2Dataset,
            path=dataset_root,
            name=_name,  # 对应数据集category
            baseline=baseline,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            # mode='m2n',
            # infer_order='double',  # 双向推理避免位置偏见
            # given_pred=[{
            #     'abbr': baseline,
            #     'path': os.path.join(dataset_root, 'model_answer', f'{baseline}.jsonl')
            # }]
        ))

# ## ------------- Evaluation Configuration
# eval = dict(
#     partitioner=dict(
#         type=SubjectiveNaivePartitioner,
#         # models=models,
#         judge_models=dict(
#                 type=TurboMindModelwithChatTemplate,
#                 abbr=f'qwen2.5_72b-turbomind',
#                 path='Qwen/Qwen2.5-72B-Instruct',
#                 engine_config=dict(session_len=131072, max_batch_size=1, tp=4),
#                 gen_config=dict(top_k=1, temperature=1e-6,
#                                 top_p=0.9, max_new_tokens=4096),
#                 max_seq_len=131072,
#                 max_out_len=4096,
#                 batch_size=1,
#                 run_cfg=dict(num_gpus=4),
#             ),
#     ),
#     runner=dict(type=LocalRunner,
#                 max_num_workers=16,
#                 task=dict(type=SubjectiveEvalTask)),
# )
