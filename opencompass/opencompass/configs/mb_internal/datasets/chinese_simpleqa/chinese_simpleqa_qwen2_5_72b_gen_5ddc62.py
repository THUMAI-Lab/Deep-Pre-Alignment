from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator.generic_llm_evaluator import GenericLLMEvaluator
from opencompass.datasets import CsimpleqaDataset, csimpleqa_postprocess
from opencompass.models import OpenAI, TurboMindModelwithChatTemplate
import os

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

subjective_reader_cfg = dict(input_columns=['primary_category', 'question', 'gold_ans',
                             'messages', 'system_prompt', 'prompt_template'], output_column='judge')

subjective_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{question}'
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

subjective_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='{system_prompt}')
                ],
                round=[
                    dict(
                        role='HUMAN',
                        prompt='{prompt_template}'
                    ),
                ]
            ),
        ),
        dataset_cfg=dict(
            abbr='chinese_simpleqa-llmjudge',
            type=CsimpleqaDataset,
            path='./data/mb_internal/chinese_simpleqa',
            local_mode=True,
            name='chinese_simpleqa',
            reader_cfg=subjective_reader_cfg,
        ),
        judge_cfg=dict(
            type=TurboMindModelwithChatTemplate,
            abbr=f'qwen2.5_72b-turbomind',
            path='Qwen/Qwen2.5-72B-Instruct',
            engine_config=dict(session_len=131072, max_batch_size=1, tp=4),
            gen_config=dict(top_k=1, temperature=1e-6,
                            top_p=0.9, max_new_tokens=4096),
            max_seq_len=131072,
            max_out_len=4096,
            batch_size=1,
            run_cfg=dict(num_gpus=4),
        ),
        dict_postprocessor=dict(type=csimpleqa_postprocess),
    ),
    pred_role='BOT',
)

csimpleqa_datasets = [
    dict(
        abbr='chinese_simpleqa',
        type=CsimpleqaDataset,
        name='chinese_simpleqa',
        path='./data/mb_internal/chinese_simpleqa',
        local_mode=True,
        reader_cfg=subjective_reader_cfg,
        infer_cfg=subjective_infer_cfg,
        eval_cfg=subjective_eval_cfg,
        mode='singlescore',
    )
]
