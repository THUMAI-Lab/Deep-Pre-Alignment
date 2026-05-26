from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ExamBenchDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


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

exambench_datasets = []
for _split in ['eval']:
    for _name in exambench_all_sets:
        _ch_name = exambench_subject_mapping[_name][1]
        exambench_infer_cfg = dict(
            ice_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        begin='</E>',
                        round=[
                            dict(
                                role='HUMAN',
                                prompt=
                                f'以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: '
                            ),
                            dict(role='BOT', prompt='{answer}'),
                        ]),
                    ice_token='</E>',

                ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(type=GenInferencer, max_out_len=20),
        )

        exambench_eval_cfg = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=first_capital_postprocess))

        exambench_datasets.append(
                dict(
                    type=ExamBenchDataset,
                    path='./data/exambench',
                    abbr=f'exambench-{_name}',
                    # file_name=f'{_name}.jsonl',
                    name=_name,
                    reader_cfg=dict(
                        # input_columns=['question','A','B','C','D','answer'],
                        input_columns=['question','A','B','C','D'],
                        output_column='answer',
                        train_split='shot',
                        test_split=_split),
                    infer_cfg=exambench_infer_cfg,
                    eval_cfg=exambench_eval_cfg)
                )
del _split, _name, _ch_name