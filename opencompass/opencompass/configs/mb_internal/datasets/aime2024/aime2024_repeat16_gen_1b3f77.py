from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import Aime2024Dataset, MATHEvaluator, math_postprocess_v2


aime2024_reader_cfg = dict(
    input_columns=['question'], 
    output_column='answer'
)


aime2024_infer_cfg = dict(
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

aime2024_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

aime2024_datasets = []
repeat_num = 16

for i in range(repeat_num):
    aime2024_datasets.append(
        dict(
            abbr=f'aime2024_{i}',
            type=Aime2024Dataset,
            path='opencompass/aime2024',
            reader_cfg=aime2024_reader_cfg,
            infer_cfg=aime2024_infer_cfg,
            eval_cfg=aime2024_eval_cfg
        )
    )
