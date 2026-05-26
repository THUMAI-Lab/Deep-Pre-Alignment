import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import GSMHardDataset, math_postprocess_v2, gsm8k_dataset_postprocess, MATHEvaluator

gsmhard_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsmhard_infer_cfg = dict(
        prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

gsmhard_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
)

gsmhard_datasets = [
    dict(
        abbr='gsm-hard',
        type=GSMHardDataset,
        path='./data/gsm-hard/gsmhardv2.jsonl',
        reader_cfg=gsmhard_reader_cfg,
        infer_cfg=gsmhard_infer_cfg,
        eval_cfg=gsmhard_eval_cfg)
]
