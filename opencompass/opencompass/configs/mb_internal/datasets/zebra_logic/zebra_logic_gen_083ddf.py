from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ZebraLogicDataset
from opencompass.datasets import ZebraLogicEvaluator

zebra_logic_reader_cfg = dict(
    input_columns=['puzzle'],
    output_column='solution',
    train_split='test',
    test_split='test'
)

zebra_logic_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{puzzle}'
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

zebra_logic_eval_cfg = dict(
    evaluator=dict(type=ZebraLogicEvaluator),
    pred_role='BOT'
)

zebra_logic_datasets = [
    dict(
        abbr='zebra_logic',
        type=ZebraLogicDataset,
        # path='allenai/ZebraLogicBench',  # No path needed for HuggingFace dataset
        # No path needed for HuggingFace dataset
        path='allenai/ZebraLogicBench-private',
        reader_cfg=zebra_logic_reader_cfg,
        infer_cfg=zebra_logic_infer_cfg,
        eval_cfg=zebra_logic_eval_cfg,
    )
]
