from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ExamBenchDataset, ExamBenchEvaluator

exambench_subject_mapping = {
    # 'highschool_chemistry_fill_in_the_blank': ['fill_in_the_blank', '以下是中国关于高中化学考试的填空题，请你参考示例完成以下填空题，把空缺的内容写在“答案：”后，如果题目中有多处空缺，应该用“；”隔开答案。'],
    # 'highschool_chinese_fill_in_the_blank': ['fill_in_the_blank', '以下是中国关于高中语文考试的填空题，请你参考示例完成以下填空题，把空缺的内容写在“答案：”后，如果题目中有多处空缺，应该用“；”隔开答案。'],
    # 'highschool_english_5_out_of_7': ['5_out_of_7', '阅读下面短文，从短文后的选项中选出可以填入空白处的最佳选项。选项中有两项为多余选项。请参考示例在“答案：”后依次写出题号和相应的答案。'],
    'highschool_english_cloze_the_blank': ['cloze_the_blank', '阅读下面短文，从每题所给的A、B、C、D四个选项中选出最佳选项。请参考示例在“答案：”后依次写出题号和相应的答案。'],
    'highschool_english_reading_comprehension': ['reading_comprehension', '阅读下列短文，从每题所给的A、B、C、D四个选项中选出最佳选项。请参考示例在“答案：”后依次写出题号和相应的答案。'],
    # 'highschool_math_fill_in_the_blank': ['fill_in_the_blank', '以下是中国关于高中数学考试的填空题，请你参考示例完成以下填空题，把空缺的内容写在“答案：”后，如果题目中有多处空缺，应该用“；”隔开答案。'],
    # 'highschool_math_word_problem': ['word_problem', '以下是中国关于高中数学考试的应用题，请按照给定的示例回答问题，通过列式计算之后，把最终答案写在“答案：”后。'],
    # 'highschool_physics_calculation': ['calculation', '以下是中国关于高中物理考试的计算题，请按照给定的示例回答问题，通过列式计算之后，把最终答案写在“答案：”后。'],
    # 'middleschool_chemistry_fill_in_the_blank': ['fill_in_the_blank', '以下是中国关于初中化学考试的填空题，请你参考示例完成以下填空题，把空缺的内容写在“答案：”后，如果题目中有多处空缺，应该用“；”隔开答案。'],
    # 'middleschool_chinese_fill_in_the_blank': ['fill_in_the_blank', '以下是中国关于初中语文考试的填空题，请你参考示例完成以下填空题，把空缺的内容写在“答案：”后，如果题目中有多处空缺，应该用“；”隔开答案。'],
    'middleschool_english_cloze_the_blank': ['cloze_the_blank', '阅读下面短文，从每题所给的A、B、C、D四个选项中选出最佳选项。请参考示例在“答案：”后依次写出题号和相应的答案。'],
    'middleschool_english_reading_comprehension': ['reading_comprehension', '阅读下列短文，从每题所给的A、B、C、D四个选项中选出最佳选项。请参考示例在“答案：”后依次写出题号和相应的答案。'],
    # 'middleschool_math_fill_in_the_blank': ['fill_in_the_blank', '以下是中国关于初中数学考试的填空题，请你参考示例完成以下填空题，把空缺的内容写在“答案：”后，如果题目中有多处空缺，应该用“；”隔开答案。'],
    # 'middleschool_math_word_problem': ['word_problem', '以下是中国关于初中数学考试的应用题，请按照给定的示例回答问题，通过列式计算之后，把最终答案写在“答案：”后。'],
    # 'middleschool_physics_calculation': ['calculation', '以下是中国关于初中物理考试的计算题，请按照给定的示例回答问题，通过列式计算之后，把最终答案写在“答案：”后。'],
    # 'middleschool_physics_fill_in_the_blank': ['fill_in_the_blank', '以下是中国关于初中物理考试的填空题，请你参考示例完成以下填空题，把空缺的内容写在“答案：”后，如果题目中有多处空缺，应该用“；”隔开答案。'],
    # 'primary_math_calculation': ['calculation', '以下是中国关于小学数学考试的计算题，请按照给定的示例回答问题，通过列式计算之后，把最终答案写在“答案：”后：'],
    # 'primary_math_fill_in_the_blank': ['fill_in_the_blank', '以下是中国关于小学数学考试的填空题，请你参考示例完成以下填空题，把空缺的内容写在“答案：”后，如果题目中有多处空缺，应该用“；”隔开答案。'],
    # 'primary_math_true_false_question': ['true_false_question', '以下是中国关于小学数学考试的判断题，请直接回答正确或者错误。'],
    # 'primary_math_word_problem': ['word_problem', '以下是中国关于小学数学考试的应用题，请按照给定的示例回答问题，通过列式计算之后，把最终答案写在“答案：”后。']
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
                template = dict(
                    begin = '</E>',
                    round = [
                        dict(
                            role = 'HUMAN',
                            prompt = f'{_ch_prompt}\n{{question}}\n答案：'
                        ),
                        dict(
                            role='BOT', 
                            prompt='{ground_truth}'
                        )
                    ]
                ),
                ice_token = '</E>'
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            # inferencer=dict(type=GenInferencer, max_out_len=128, stopping_criteria=['以下是中国', '请你参考示例', '**解释', '以下是解释', '**解析', '\n\n\n', '阅读下面短文', '阅读下列短文'])
            inferencer=dict(type=GenInferencer)
        )

        exambench_eval_cfg = dict(
            evaluator = dict(type = 'ExamBenchEvaluator'+ '_' + exambench_subject_mapping[_name][0])
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
