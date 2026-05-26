from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SanitizedMBPPDataset, MBPPEvaluator

sanitized_mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='test_list_2')

sanitized_mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task:\nWrite a function to find the similar elements from the given two tuple lists.\nYour code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n',),
                dict(role='BOT', prompt='```python\ndef similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)```',),
                
                dict(role='HUMAN', prompt='You are an expert Python programmer, and here is your task:\n{text}\nYour code should pass these tests:\n\n{test_list}\n',),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

sanitized_mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator), pred_role='BOT')

sanitized_mbpp_datasets = [
    dict(
        type=SanitizedMBPPDataset,
        abbr='sanitized_mbpp',
        path='opencompass/sanitized_mbpp',
        reader_cfg=sanitized_mbpp_reader_cfg,
        infer_cfg=sanitized_mbpp_infer_cfg,
        eval_cfg=sanitized_mbpp_eval_cfg,
    )
]
