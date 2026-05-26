from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.datasets import Aime2025Dataset


aime2025_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)


aime2025_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}'.strip() + '\n\nPut your final answer within a \\boxed{}.'),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

aime2025_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator)
)

aime2025_datasets = []
repeat_num = 16

for i in range(repeat_num):
    aime2025_datasets.append(
        dict(
            abbr=f'aime2025_{i}',
            type=Aime2025Dataset,
            path='opencompass/aime2025',
            reader_cfg=aime2025_reader_cfg,
            infer_cfg=aime2025_infer_cfg,
            eval_cfg=aime2025_eval_cfg,
        )
    )
