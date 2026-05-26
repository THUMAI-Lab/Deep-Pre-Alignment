from mmengine.config import read_base
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import OlympiadBenchDataset, OlympiadBenchEvaluator, olympiadbench_postprocess_v2
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.models import OpenAISDK
from opencompass.configs.summarizers.OlympiadBench import summarizer
import os
from opencompass.datasets import (
    MATHDataset,
    MATHEvaluator,
    math_postprocess_v2,
    normalize_final_answer,
)
# from opencompass.openicl.icl_evaluator import MATHEvaluator


# with read_base():
#     from .OlympiadBench_categories import categories

categories = [
    'OE_TO_maths_en_COMP',  # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_COMP',  # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_CEE',  # OpenEnded - TextOnly - maths - CEE
    'OE_TO_physics_en_COMP',  # OpenEnded - TextOnly - physics - COMP
    'OE_TO_physics_zh_CEE'  # OpenEnded - TextOnly - physics - CEE
]

math_categories = [
    'OE_TO_maths_en_COMP',  # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_COMP',  # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_CEE',  # OpenEnded - TextOnly - maths - CEE
]

physics_categories = [
    'OE_TO_physics_en_COMP',  # OpenEnded - TextOnly - physics - COMP
    'OE_TO_physics_zh_CEE'  # OpenEnded - TextOnly - physics - CEE
]


# Create prompter instance for problems
olympiadbench_prompter_cfg = dict(
    type='OlympiadBenchPrompter'
)

olympiadbench_reader_cfg = dict(
    input_columns=[
        'problem', 'language', 'subject', 'question_type',
        'answer_type', 'is_multiple_answer', 'unit', 'questions'
    ],
    output_column='solution'
)

GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
    
    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer is consistent with the standard answer.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT 
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>: \n{problem}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{solution}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n
    
    Judging the correctness of candidates' answers:
""".strip()

olympiadbench_datasets = []
for _name in categories:
    olympiadbench_infer_cfg = dict(
        prompt_template=dict(
            type='OlympiadBenchTemplate'
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    # Evaluation configuration
    olympiadbench_eval_cfg = dict(
        evaluator=dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.")
                    ],
                    round=[
                        dict(
                            role='HUMAN',
                            prompt=GRADER_TEMPLATE
                        ),
                    ]),
            ),
            dataset_cfg=dict(
                type=OlympiadBenchDataset,
                path='opencompass/OlympiadBench',
                name=_name,
                reader_cfg=olympiadbench_reader_cfg,
            ),
            judge_cfg=dict(
                type=OpenAISDK,
                path=None,
                key=os.environ.get('OC_JUDGE_API_KEY'),
                openai_api_base=[
                    os.environ.get('OC_JUDGE_API_BASE',
                                   'https://one-api.modelbest.co/v1/')
                ],
                meta_template=dict(round=[
                    dict(role='HUMAN', api_role='HUMAN'),
                    dict(role='BOT', api_role='BOT', generate=True),
                ], ),
                query_per_second=8,
                batch_size=1024,
                temperature=0.001,
                tokenizer_path='gpt-4o-mini',
                verbose=True,
                max_out_len=16384,
                max_seq_len=49152,
            ),
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )
    olympiadbench_eval_cfg = dict(
        evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
    )

    olympiadbench_datasets.append(
        dict(
            type=OlympiadBenchDataset,
            abbr=f'OlympiadBench_{_name}',
            path='opencompass/OlympiadBench',
            name=_name,
            reader_cfg=olympiadbench_reader_cfg,
            infer_cfg=olympiadbench_infer_cfg,
            eval_cfg=olympiadbench_eval_cfg,
        )
    )

del _name
