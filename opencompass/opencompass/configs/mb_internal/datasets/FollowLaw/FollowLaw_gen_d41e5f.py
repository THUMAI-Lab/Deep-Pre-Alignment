from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import FollowLawDataset, FollowLawEvaluator

import os
import os.path as osp

followlaw_reader_cfg = dict(
    input_columns=['prompt'], output_column='reference')

followlaw_path = osp.join(os.environ.get(
    'COMPASS_DATA_CACHE', './'), 'data/mb_internal/FollowLaw')
# followlaw_path = "data/FollowLaw2"

# followlaw_all_sets = []
followlaw_all_sets = [os.path.basename(f) for f in os.listdir(
    followlaw_path) if f.endswith('.jsonl')]

# followlaw_all_sets = [
#     "1合并上诉状-1.jsonl",
# "2一审民事审理要点-1.jsonl",
# "3争议焦点（带开庭笔录）-1.jsonl",
# "5核心法条适用要点-1.jsonl",
# "6执行驳回裁定说理-1.jsonl,"
# ]

followlaw_datasets = []

for _name in followlaw_all_sets:
    followlaw_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}'),
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=1025))

    followlaw_eval_cfg = dict(
        evaluator=dict(type=FollowLawEvaluator),
        pred_role='BOT',
    )

    followlaw_datasets.append(
        dict(
            abbr=f'followlaw_{os.path.splitext(_name)[0]}',
            type=FollowLawDataset,
            path=followlaw_path,
            name=_name,
            reader_cfg=followlaw_reader_cfg,
            infer_cfg=followlaw_infer_cfg,
            eval_cfg=followlaw_eval_cfg)
    )

del _name
