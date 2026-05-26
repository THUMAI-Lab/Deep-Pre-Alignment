from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import AlignmentBenchDataset, alignbench_postprocess
from opencompass.models import OpenAI
from opencompass.evaluator import GenericLLMEvaluator
import os

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

subjective_reader_cfg = dict(
    input_columns=['question', 'capability', 'critiquellm_prefix'],
    output_column='judge',
    )

subjective_all_sets = [
    'alignment_bench_v1_1', # Changed to Alignbench_v1_1 since 06/15/2024, refer to https://github.com/THUDM/AlignBench
]
data_path ='data/subjective/alignment_bench'

alignment_bench_config_path = 'data/subjective/alignment_bench/config'
alignment_bench_config_name = 'multi-dimension'

alignbench_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{question}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = '{critiquellm_prefix}[助手的答案开始]\n{prediction}\n[助手的答案结束]\n'
                    ),
                ]),
            ),
            dataset_cfg=dict(
                abbr=f'{_name}-llmjudge',
                type=AlignmentBenchDataset,
                path=data_path,
                name=_name,
                alignment_bench_config_path=alignment_bench_config_path,
                alignment_bench_config_name=alignment_bench_config_name,
                reader_cfg=subjective_reader_cfg,
            ),
            judge_cfg=dict(
                abbr=os.environ.get('JUDGE_MODEL', 'MUST DEFINE_JUDGE_MODEL'),
                type=OpenAI,
                path=os.environ.get('JUDGE_MODEL', 'MUST DEFINE_JUDGE_MODEL'),
                key='ENV',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
                meta_template=api_meta_template,
                query_per_second=1,
                max_out_len=16384,
                max_seq_len=131072,
                batch_size=1,
                verbose=True
            ),
            dict_postprocessor=dict(type=alignbench_postprocess, judge_type='general'),
        ),
        pred_role='BOT',
    )

    alignbench_datasets.append(
        dict(
            abbr=f'{_name}',
            type=AlignmentBenchDataset,
            path=data_path,
            name=_name,
            alignment_bench_config_path=alignment_bench_config_path,
            alignment_bench_config_name=alignment_bench_config_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg,
            mode='singlescore',
        ))
