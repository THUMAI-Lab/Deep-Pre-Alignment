from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ExamBenchDataset, ExamBenchEvaluator

exambench_subject_mapping = {
    'highschool_biology_multiple_choice': ['multiple_choice', '以下是中国关于高中生物考试的单项选择题，请选出其中的正确答案。'],
    'highschool_chemistry_multiple_choice': ['multiple_choice', '以下是中国关于高中化学考试的单项选择题，请选出其中的正确答案。'],
    'highschool_chinese_multiple_choice': ['multiple_choice', '以下是中国关于高中语文考试的单项选择题，请选出其中的正确答案。'],
    'highschool_geography_multiple_choice': ['multiple_choice', '以下是中国关于高中地理考试的单项选择题，请选出其中的正确答案。'],
    'highschool_history_multiple_choice': ['multiple_choice', '以下是中国关于高中历史考试的单项选择题，请选出其中的正确答案。'],
    'highschool_math_multiple_choice': ['multiple_choice', '以下是中国关于高中数学考试的单项选择题，请选出其中的正确答案。'],
    'highschool_physics_multiple_choice': ['multiple_choice', '以下是中国关于高中物理考试的单项选择题，请选出其中的正确答案。'],
    'highschool_politics_multiple_choice': ['multiple_choice', '以下是中国关于高中政治考试的单项选择题，请选出其中的正确答案。'],
    'middleschool_biology_multiple_choice': ['multiple_choice', '以下是中国关于初中生物考试的单项选择题，请选出其中的正确答案。'],
    'middleschool_chemistry_multiple_choice': ['multiple_choice', '以下是中国关于初中化学考试的单项选择题，请选出其中的正确答案。'],
    'middleschool_chinese_multiple_choice': ['multiple_choice', '以下是中国关于初中语文考试的单项选择题，请选出其中的正确答案。'],
    'middleschool_english_multiple_choice': ['multiple_choice', '以下是中国关于初中英语考试的单项选择题，请选出其中的正确答案。'],
    'middleschool_geography_multiple_choice': ['multiple_choice', '以下是中国关于初中地理考试的单项选择题，请选出其中的正确答案'],
    'middleschool_history_multiple_choice': ['multiple_choice', '以下是中国关于初中历史考试的单项选择题，请选出其中的正确答案。'],
    'middleschool_math_multiple_choice': ['multiple_choice', '以下是中国关于初中数学考试的单项选择题，请选出其中的正确答案。'],
    'middleschool_physics_multiple_choice': ['multiple_choice', '以下是中国关于初中物理考试的单项选择题，请选出其中的正确答案。'],
    'middleschool_politics_multiple_choice': ['multiple_choice', '以下是中国关于初中政治考试的单项选择题，请选出其中的正确答案。'],
    'primary_math_multiple_choice': ['multiple_choice', '以下是中国关于小学数学考试的单项选择题，请选出其中的正确答案。'],
}

exambench_all_sets = list(exambench_subject_mapping.keys())

exambench_datasets = []

for _split in ['eval']:
    for _name in exambench_all_sets:
        exambench_reader_cfg = dict(
            input_columns = ['question'],
            output_column = 'ground_truth',
            train_split = 'shot',
            test_split = _split
        )

        _ch_prompt = exambench_subject_mapping[_name][1]
        
        exambench_infer_cfg = dict(
            ice_template = dict(
                type = PromptTemplate,
                template={
                    answer: dict(
                        begin='</E>',
                        round=[
                            dict(
                                role='HUMAN',
                                prompt=f'{_ch_prompt}\n{{question}}\n答案： '
                            ),
                            dict(role='BOT', prompt=answer),
                        ])
                    for answer in ['A', 'B', 'C', 'D']
                },
                ice_token='</E>',
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(type=PPLInferencer)
        )

        exambench_eval_cfg = dict(
            evaluator = dict(type=AccEvaluator)
        )

        exambench_datasets.append(
            dict(
                type = ExamBenchDataset,
                abbr = f'exambench-{_name}',
                path = '/data/jjm/opencompass/data/exambench_v20240911',
                name=_name,
                reader_cfg = exambench_reader_cfg,
                infer_cfg = exambench_infer_cfg,
                eval_cfg = exambench_eval_cfg
            )
        )

del _name, _ch_prompt
