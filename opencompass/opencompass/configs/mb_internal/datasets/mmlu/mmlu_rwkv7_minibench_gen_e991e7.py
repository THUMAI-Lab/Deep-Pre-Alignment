from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# with read_base():
#     from .mmlu_all_sets import mmlu_all_sets
mmlu_all_sets = ['anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_medicine', 'college_physics', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'jurisprudence', 'logical_fallacies', 'marketing', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

# QUERY_TEMPLATE = f"""{_hint} You are a very talented expert. Answer this question:
# {input}
# A. {A}
# B. {B}
# C. {C}
# D. {D}
# """.strip()

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')


mmlu_datasets = []
for name in mmlu_all_sets:
    _hint = f'You are a very talented expert in {name}. Answer this question by replying A, B, C or D:'
    mmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN', 
                        prompt=f"""{_hint}\n
{{input}}
A. {{A}}
B. {{B}}
C. {{C}}
D. {{D}}
""".strip()),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        # pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'(?i)ANSWER\s*:\s*([A-D])'))
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))
        # pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'boxed{([A-D])}'))

    mmlu_datasets.append(
        dict(
            abbr=f'lukaemon_mmlu_minibench_{name}',
            type=MMLUDataset,
            path='minibench/mmlu',
            name=name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        ))
