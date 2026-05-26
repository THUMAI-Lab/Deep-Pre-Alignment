import os
from opencompass.openicl.internal.icl_prompt_template import PromptTemplateInternal
from opencompass.openicl.internal import ZeroRetrieverForPPLOnly, PPLOnlyInferencerScalingBench
from opencompass.openicl.icl_evaluator import AveragePPLEvaluator
from opencompass.datasets import ScalingBenchDataset


scaling_bench_all_sets = [
    'mmlu',
    'cmmlu',
    'ceval',
    'bbh',
    'math',
    'mbpp',
    'human_eval',
    'gaokao2024',
    'livecodebench'
]


scaling_bench_reader_cfg = dict(
    input_columns=['input', 'answer'],
    output_column='answer_key',
    train_split='dev',
    test_split='dev')

scaling_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplateInternal,
        template={
            'input': '{input}',
            'answer': '{answer}'
        }),
    retriever=dict(type=ZeroRetrieverForPPLOnly),
    inferencer=dict(type=PPLOnlyInferencerScalingBench, save_every=100))


scaling_bench_eval_cfg = dict(
    evaluator=dict(type=AveragePPLEvaluator))

scaling_bench_datasets = []
for _name in scaling_bench_all_sets:
    scaling_bench_datasets.append(
        dict(
            type=ScalingBenchDataset,
            abbr=f'scaling_bench_{_name}',
            path=os.path.join(os.getenv('COMPASS_DATA_CACHE',
                              './'), f'data/mb_internal/scaling_bench/{_name}'),
            file_name='data.jsonl',
            reader_cfg=scaling_bench_reader_cfg,
            infer_cfg=scaling_bench_infer_cfg,
            eval_cfg=scaling_bench_eval_cfg)
    )
