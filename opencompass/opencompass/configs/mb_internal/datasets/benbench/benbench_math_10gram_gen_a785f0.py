from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import BenBenchDataset, BenBenchEvaluator
import os


dataset_settings = [
    ['GSM8K-origin-train', '../../gsm8k/train.jsonl'],
    ['GSM8K-origin-test', '../../gsm8k/test.jsonl'],
    
    ['GSM8K_rewritten-train-1', 'rewritten/GSM8K_rewritten-train-1.jsonl'],
    ['GSM8K_rewritten-train-2', 'rewritten/GSM8K_rewritten-train-2.jsonl'],
    ['GSM8K_rewritten-train-3', 'rewritten/GSM8K_rewritten-train-3.jsonl'],
    ['GSM8K_rewritten-test-1', 'rewritten/GSM8K_rewritten-test-1.jsonl'],
    ['GSM8K_rewritten-test-2', 'rewritten/GSM8K_rewritten-test-2.jsonl'],
    ['GSM8K_rewritten-test-3', 'rewritten/GSM8K_rewritten-test-3.jsonl'],
    
    ['MATH-origin-train', '../../math/train.jsonl'],
    ['MATH-origin-test', '../../math/test.jsonl'],
    
    ['MATH-origin-prm800k-500-test', '../../math/test_prm800k_500.jsonl'],
    
    ['MATH_rewritten-train-1', 'rewritten/MATH_rewritten-train-1.jsonl'],
    ['MATH_rewritten-train-2', 'rewritten/MATH_rewritten-train-2.jsonl'],
    ['MATH_rewritten-train-3', 'rewritten/MATH_rewritten-train-3.jsonl'],
    ['MATH_rewritten-test-1', 'rewritten/MATH_rewritten-test-1.jsonl'],
    ['MATH_rewritten-test-2', 'rewritten/MATH_rewritten-test-2.jsonl'],
    ['MATH_rewritten-test-3', 'rewritten/MATH_rewritten-test-3.jsonl'],
    
    
    # ['GSM8K-origin-test', 'original/GSM8K-origin-test.jsonl'],
    # ['GSM8K-origin-train', 'original/GSM8K-origin-train.jsonl'],
    # ['MATH-origin-test', 'original/MATH-origin-test.jsonl'],
    # ['MATH-origin-train', 'original/MATH-origin-train.jsonl'],
    # ['MATH-origin-prm800k-500-test', 'original/MATH-origin-prm800k-500-test.jsonl'],
    # ['MMLU-origin-dev', 'original/MMLU-origin-dev.jsonl'],
    # ['MMLU-origin-val', 'original/MMLU-origin-val.jsonl'],
    # ['MMLU-origin-test', 'original/MMLU-origin-test.jsonl'],
    # ['GPQA_DIAMOND-origin-test', 'original/GPQA_DIAMOND-origin-test.jsonl'],
    # ['HUMANEVAL-origin-test', 'original/HUMANEVAL-origin-test.jsonl'],
    # ['IFEVAL-origin-test', 'original/IFEVAL-origin-test.jsonl'],
    # ['TRIVIAQA-origin-test', 'original/TRIVIAQA-origin-test.jsonl'],
    # ['TRIVIAQA-origin-train', 'original/TRIVIAQA-origin-train.jsonl'],
    # ['MATH_rewritten-prm800k-500-test-1',
    #     'rewritten/MATH_rewritten-prm800k-500-test-1.jsonl'],
    # ['MATH_rewritten-prm800k-500-test-2',
    #     'rewritten/MATH_rewritten-prm800k-500-test-2.jsonl'],
    # ['MATH_rewritten-prm800k-500-test-3',
        # 'rewritten/MATH_rewritten-prm800k-500-test-3.jsonl'],
]

# n_gram = 5  #
n_gram = 10  #

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
