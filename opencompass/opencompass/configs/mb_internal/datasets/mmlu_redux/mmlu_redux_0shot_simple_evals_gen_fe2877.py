from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithFullDetailsEvaluator
from opencompass.datasets import MMLUReduxDataset


mmlu_redux_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',)

mmlu_redux_all_sets = [
    'anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'econometrics', 'electrical_engineering', 'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_mathematics','high_school_physics', 'high_school_statistics', 'human_aging', 'logical_fallacies', 'machine_learning', 'miscellaneous', 'philosophy', 'professional_accounting', 'public_relations', 'virology', 'conceptual_physics', 'high_school_us_history', 'astronomy', 'high_school_geography', 'high_school_macroeconomics', 'professional_law'
]

_hint = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{input}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


mmlu_redux_datasets = []
for _name in mmlu_redux_all_sets:
    mmlu_redux_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt=_hint,
                    ),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_redux_eval_cfg = dict(
        evaluator=dict(type=AccwithFullDetailsEvaluator),
    )

    mmlu_redux_datasets.append(
        dict(
            abbr=f'mmlu_redux_{_name}',
            type=MMLUReduxDataset,
            path='edinburgh-dawg/mmlu-redux',
            name=_name,
            reader_cfg=mmlu_redux_reader_cfg,
            infer_cfg=mmlu_redux_infer_cfg,
            eval_cfg=mmlu_redux_eval_cfg,
        ))

del _name, _hint
