from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator, AccEvaluator
from opencompass.datasets import MMLUReduxDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


mmlu_redux_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',)

mmlu_redux_all_sets = ['anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'econometrics', 'electrical_engineering', 'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'human_aging', 'logical_fallacies', 'machine_learning', 'miscellaneous', 'philosophy', 'professional_accounting', 'public_relations', 'virology', 'conceptual_physics', 'high_school_us_history', 'astronomy', 'high_school_geography', 'high_school_macroeconomics', 'professional_law']

mmlu_redux_datasets = []
for _name in mmlu_redux_all_sets:
    _hint = f'There is a single choice question about {_name.replace("_", " ")}. Answer the question by replying A, B, C or D.'
    mmlu_redux_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                ),
                dict(role='BOT', prompt='{target}\n')
            ]),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f"{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nLet's think step by step. A:"
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=1024),
    )

    mmlu_redux_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

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
