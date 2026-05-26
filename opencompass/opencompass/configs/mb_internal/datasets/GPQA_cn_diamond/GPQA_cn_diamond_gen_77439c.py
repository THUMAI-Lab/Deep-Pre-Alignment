from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithFullDetailsEvaluator
from opencompass.datasets import GPQACnDiamondDataset
import os

QUERY_TEMPLATE = '''    
你回答的最后一行**必须**是以下格式 '答案：$选项' (不带引号), 其中选项是ABCD之一。请在回答之前一步步思考。

{prompt}
'''.strip()

GPQA_cn_diamond_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='gold'
)

GPQA_cn_diamond_sets = ['GPQA_cn_diamond']
GPQA_cn_diamond_datasets = []

for name in GPQA_cn_diamond_sets:
    GPQA_cn_diamond_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    GPQA_cn_diamond_eval_cfg = dict(
        evaluator=dict(type=AccwithFullDetailsEvaluator),
    )

    GPQA_cn_diamond_datasets.append(
        dict(
            abbr=f'GPQA_cn_diamond',
            type=GPQACnDiamondDataset,
            # path='opencompass/GPQA_cn_diamond',
            path=os.path.join(os.environ.get('COMPASS_DATA_CACHE', './'), 'data/mb_internal/GPQA_cn_diamond'),
            name=name,
            reader_cfg=GPQA_cn_diamond_reader_cfg,
            infer_cfg=GPQA_cn_diamond_infer_cfg,
            eval_cfg=GPQA_cn_diamond_eval_cfg,
        ))
