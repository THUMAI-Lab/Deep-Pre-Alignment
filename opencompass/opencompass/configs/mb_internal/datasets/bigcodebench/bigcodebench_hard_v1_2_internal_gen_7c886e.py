from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (BigCodeBenchDataset, BigCodeBenchEvaluator)

bigcodebench_hard_v1_2_datasets = []

for _subset in ['complete', 'instruct']:

    bigcodebench_hard_v1_2_reader_cfg = dict(
        input_columns=[f'{_subset}_prompt'],
        output_column='test',
    )

    bigcodebench_hard_v1_2_infer_cfg = dict(prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[dict(role='system', fallback_role='HUMAN', prompt='')],
            round=[
                dict(role='HUMAN', prompt='{complete_prompt}' if _subset ==
                     'complete' else '{instruct_prompt}'),
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer)
    )

    bigcodebench_hard_v1_2_eval_cfg = dict(
        evaluator=dict(
            type=BigCodeBenchEvaluator,
            release_version='v0.1.2',
            eval_type=_subset,
            remote_execute_api='https://bigcode-bigcodebench-evaluator.hf.space/',
            dataset_version='hard',
        ),
        pred_role='BOT',
    )

    bigcodebench_hard_v1_2_datasets.append(
        dict(
            abbr=f'bigcodebench_hard_{_subset}_v0.1.2',
            type=BigCodeBenchDataset,
            path='opencompass/bigcodebench',
            reader_cfg=bigcodebench_hard_v1_2_reader_cfg,
            infer_cfg=bigcodebench_hard_v1_2_infer_cfg,
            eval_cfg=bigcodebench_hard_v1_2_eval_cfg,
            release_version='v0.1.2',
            dataset_version='hard',
        )
    )
