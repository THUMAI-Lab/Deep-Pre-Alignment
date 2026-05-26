from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator.generic_llm_evaluator import GenericLLMEvaluator
from opencompass.datasets import CsimpleqaDataset, csimpleqa_postprocess
from opencompass.models import OpenAI
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
            path='opencompass/chinese_simpleqa',
            name='chinese_simpleqa',
            reader_cfg=subjective_reader_cfg,
        ),
        judge_cfg=dict(
            type=OpenAI,
            # path=os.environ.get('JUDGE_MODEL', 'MUST DEFINE_JUDGE_MODEL'),
            path='gpt-4o',
            key='ENV',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
            meta_template=api_meta_template,
            query_per_second=1,
            max_out_len=16384,
            max_seq_len=131072,
            batch_size=1,
            verbose=True
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
        path='opencompass/chinese_simpleqa',
        reader_cfg=subjective_reader_cfg,
        infer_cfg=subjective_infer_cfg,
        eval_cfg=subjective_eval_cfg,
        mode='singlescore',
    )
]
