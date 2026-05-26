from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import BenBenchDataset, BenBenchEvaluator
import os

dataset_settings = [
    ['boolean_expressions', 'bbh/boolean_expressions.json'],
    ['causal_judgement', 'bbh/causal_judgement.json'],
    ['date_understanding', 'bbh/date_understanding.json'],
    ['disambiguation_qa', 'bbh/disambiguation_qa.json'],
    ['dyck_languages', 'bbh/dyck_languages.json'],
    ['formal_fallacies', 'bbh/formal_fallacies.json'],
    ['geometric_shapes', 'bbh/geometric_shapes.json'],
    ['hyperbaton', 'bbh/hyperbaton.json'],
    ['logical_deduction_five_objects', 'bbh/logical_deduction_five_objects.json'],
    ['logical_deduction_seven_objects', 'bbh/logical_deduction_seven_objects.json'],
    ['logical_deduction_three_objects', 'bbh/logical_deduction_three_objects.json'],
    ['movie_recommendation', 'bbh/movie_recommendation.json'],
    ['multistep_arithmetic_two', 'bbh/multistep_arithmetic_two.json'],
    ['navigate', 'bbh/navigate.json'],
    ['object_counting', 'bbh/object_counting.json'],
    ['penguins_in_a_table', 'bbh/penguins_in_a_table.json'],
    ['reasoning_about_colored_objects', 'bbh/reasoning_about_colored_objects.json'],
    ['ruin_names', 'bbh/ruin_names.json'],
    ['salient_translation_error_detection', 'bbh/salient_translation_error_detection.json'],
    ['snarks', 'bbh/snarks.json'],
    ['sports_understanding', 'bbh/sports_understanding.json'],
    ['temporal_sequences', 'bbh/temporal_sequences.json'],
    ['tracking_shuffled_objects_five_objects', 'bbh/tracking_shuffled_objects_five_objects.json'],
    ['tracking_shuffled_objects_seven_objects', 'bbh/tracking_shuffled_objects_seven_objects.json'],
    ['tracking_shuffled_objects_three_objects', 'bbh/tracking_shuffled_objects_three_objects.json'],
    ['web_of_lies', 'bbh/web_of_lies.json'],
    ['word_sorting', 'bbh/word_sorting.json'],
]

n_gram = 5  #
# n_gram = 10  #

reader_cfg = dict(
    input_columns=['prompt'],
    output_column='reference'
)

infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=n_gram)
)

eval_cfg = dict(
    evaluator=dict(
        type=BenBenchEvaluator
    )
)

BenBench_datasets = []

for dataset_abbr, dataset_path in dataset_settings:
    BenBench_datasets.append(
        dict(
            abbr=f'BBH-{dataset_abbr}-origin-{n_gram}gram',
            type=BenBenchDataset,
            num_gram=n_gram,
            path=os.path.join(os.environ.get('COMPASS_DATA_CACHE', './'), 'data/BBH/data'),
            # tokenizer_path=model_path, # todo, define tokenizer path
            dataset_kwargs=dict(
                dataset='bbh',
                name=dataset_abbr
            ),
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        )
    )
