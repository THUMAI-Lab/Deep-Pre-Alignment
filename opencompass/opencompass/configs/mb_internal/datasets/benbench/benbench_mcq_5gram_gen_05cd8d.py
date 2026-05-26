from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import BenBenchDataset, BenBenchEvaluator
import os

dataset_settings = [
    ['MMLU-origin-dev', 'mmlu/dev.jsonl'],
    ['MMLU-origin-val', 'mmlu/val.jsonl'],
    ['MMLU-origin-test', 'mmlu/test.jsonl'],
    
    ['CEVAL-origin-dev', 'ceval/dev.jsonl'],
    ['CEVAL-origin-val', 'ceval/val.jsonl'],
    ['CEVAL-origin-test', 'ceval/test.jsonl'],
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
            abbr=dataset_abbr + f'-{n_gram}gram',
            type=BenBenchDataset,
            num_gram=n_gram,
            path=os.path.join(os.environ.get('COMPASS_DATA_CACHE', './'), 'data/mb_internal/benbench', dataset_path),
            # tokenizer_path=model_path, # todo, define tokenizer path
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg
        )
    )
