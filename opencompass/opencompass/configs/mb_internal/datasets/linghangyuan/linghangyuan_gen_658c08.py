from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithFullDetailsEvaluator
from opencompass.datasets import LinghangyuanDataset
import os

QUERY_TEMPLATE_zh = '''    
你回答的最后一行**必须**是以下格式 '答案：$选项' (不带引号), 其中选项是ABCD之一。请在回答之前一步步思考。

{question}
'''.strip()

QUERY_TEMPLATE_en = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{input}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


linghangyuan_sets = ['500-singlechoice']
linghangyuan_datasets = []

linghangyuan_eval_cfg = dict(
    evaluator=dict(type=AccwithFullDetailsEvaluator),
)

for name in linghangyuan_sets:
    linghangyuan_zh_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=QUERY_TEMPLATE_zh),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    linghangyuan_zh_reader_cfg = dict(
        input_columns=['question'],
        output_column='answer',
        train_split='zh',
        test_split='zh'
    )

    linghangyuan_en_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=QUERY_TEMPLATE_en),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    linghangyuan_en_reader_cfg = dict(
        input_columns=['question'],
        output_column='answer',
        train_split='en',
        test_split='en'
    )

    linghangyuan_datasets.append(
        dict(
            abbr=f'linghangyuan_zh',
            type=LinghangyuanDataset,
            # path='opencompass/linghangyuan',
            path=os.path.join(
                os.environ.get('COMPASS_DATA_CACHE', './'), 'data/mb_internal/linghangyuan'),
            name=name,
            reader_cfg=linghangyuan_zh_reader_cfg,
            infer_cfg=linghangyuan_zh_infer_cfg,
            eval_cfg=linghangyuan_eval_cfg,
        ))

    linghangyuan_datasets.append(
        dict(
            abbr=f'linghangyuan_en',
            type=LinghangyuanDataset,
            # path='opencompass/linghangyuan',
            path=os.path.join(
                os.environ.get('COMPASS_DATA_CACHE', './'), 'data/mb_internal/linghangyuan'),
            name=name,
            reader_cfg=linghangyuan_en_reader_cfg,
            infer_cfg=linghangyuan_en_infer_cfg,
            eval_cfg=linghangyuan_eval_cfg,
        ))
