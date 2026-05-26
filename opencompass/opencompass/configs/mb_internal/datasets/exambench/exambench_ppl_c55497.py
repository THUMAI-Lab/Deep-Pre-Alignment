from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MyDataset


exambench_subject_mapping = {
    'highschool-biology-multiple-choice': ['highschool-biology', '高中生物'],
    'highschool-chemistry-multiple-choice': ['highschool-chemistry', '高中化学'],
    # 'highschool-chinese-multiple-choice': ['highschool-chinese', '高中语文'],
    # 'highschool-english-5-out-of-7': ['highschool-english', '高中英文7选5'],
    # 'highschool-english-cloze-the-blank': ['highschool-english', '高中英语完型'],
    # 'highschool-english-reading-comprehension': ['highschool-english', '高中英语阅读'],
    # 'highschool-geography-multiple-choice': ['highschool-geography', '高中地理'],
    # 'highschool-history-multiple-choice': ['highschool-history', '高中历史'],
    'highschool-math-multiple-choice': ['highschool-math', '高中数学'],
    'highschool-physics-multiple-choice': ['highschool-physics', '高中物理'],
    'highschool-politics-multiple-choice': ['highschool-politics', '高中政治'],
    # 'middleschool-biology-multiple-choice': ['middleschool-biology', '初中生物'],
    # 'middleschool-chemistry-multiple-choice': ['middleschool-chemistry', '初中化学'],
    # 'middleschool-chinese-multiple-choice': ['middleschool-chinese', '初中语文'],
    # 'middleschool-english-cloze-the-blank': ['middleschool-english', '初中英语完型'],
    # 'middleschool-english-multiple-choice': ['middleschool-english', '初中英语选择'],
    # 'middleschool-english-reading-comprehension': ['middleschool-english', '初中英语阅读'],
    # 'middleschool-geography-multiple-choice': ['middleschool-geography', '初中地理'],
    'middleschool-history-multiple-choice': ['middleschool-history', '初中历史'],
    'middleschool-math-multiple-choice': ['middleschool-math', '初中数学'],
    'middleschool-physics-multiple-choice': ['middleschool-physics', '初中物理'],
    # 'middleschool-politics-multiple-choice': ['middleschool-politics', '初中政治'],
    # 'primary-math-multiple-choice': ['primary-math', '小学数学'],

   
}
exambench_all_sets = list(exambench_subject_mapping.keys())

mydata_datasets = []


for _name in exambench_all_sets:
    mydata_bench_reader_cfg = dict(
        input_columns=['question','A','B','C','D','answer'],
        output_column='answer_key',
        train_split='dev',
        test_split='dev')
    
    _ch_name = exambench_subject_mapping[_name][1]

    hint = f'以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。'
    question_and_options = '{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'

    mydata_bench_infer_cfg = dict(
        # prompt_template=dict(
        #     type=PromptTemplate,
        #     template={
        #         "input": "{input}",
        #         "answer": "{answer}"
        #     }),
        ice_template=dict(
                type=PromptTemplate,
                template={answer: f'{question_and_options}\n答案: {answer}\n' for answer in ['A', 'B', 'C', 'D']},
            ),
    
        prompt_template=dict(
                type=PromptTemplate,
                template={answer: f'{hint}\n</E>{question_and_options}\n答案: {answer}' for answer in ['A', 'B', 'C', 'D']},
                ice_token='</E>',
            ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, ]),
        inferencer=dict(type=PPLInferencer)
    )



    mydata_bench_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator))


    mydata_datasets.append(
            dict(
                type=MyDataset,
                abbr=f'mydata-{_name}',
                path='/data/jjm/opencompass/data/custom_data',
                file_name=f'{_name}.jsonl',
                reader_cfg=mydata_bench_reader_cfg,
                infer_cfg=mydata_bench_infer_cfg,
                eval_cfg=mydata_bench_eval_cfg)
            )