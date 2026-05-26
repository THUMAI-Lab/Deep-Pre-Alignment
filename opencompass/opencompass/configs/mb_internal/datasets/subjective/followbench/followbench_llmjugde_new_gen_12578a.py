from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.models import OpenAI
from opencompass.datasets import FollowBenchDataset, followbench_postprocess
import os

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )


subjective_reader_cfg = dict(
    input_columns=['instruction', 'judge_prompt',],
    output_column='judge',
)

subjective_all_sets = [
    'followbench_llmeval_cn', 'followbench_llmeval_en',
]
data_path = 'data/subjective/followbench/converted_data'

followbench_llmeval_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt='{instruction}'
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
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{judge_prompt}'
                    ),
                ]),
            ),
            dataset_cfg=dict(
                abbr=f'{_name}-llmjudge',
                type=FollowBenchDataset,
                path=data_path,
                name=_name,
                mode='singlescore',
                cate='llm',
                reader_cfg=subjective_reader_cfg,
            ),
            judge_cfg=dict(
                abbr=os.environ.get('JUDGE_MODEL', 'MUST DEFINE_JUDGE_MODEL'),
                type=OpenAI,
                path=os.environ.get('JUDGE_MODEL', 'MUST DEFINE_JUDGE_MODEL'),
                key='ENV',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
                meta_template=api_meta_template,
                query_per_second=1,
                max_out_len=32768,
                max_seq_len=131072,
                batch_size=1,
                verbose=True
            ),
            dict_postprocessor=dict(type=followbench_postprocess),
        ),
        pred_role='BOT',
    )

    followbench_llmeval_datasets.append(
        dict(
            abbr=f'{_name}',
            type=FollowBenchDataset,
            path=data_path,
            name=_name,
            mode='singlescore',
            cate='llm',
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
        ))
