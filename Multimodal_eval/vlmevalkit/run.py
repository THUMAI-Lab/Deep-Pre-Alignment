import torch
import torch.distributed as dist

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer

import pandas as pd
import shutil
import os
import ast

def process_and_save_excel(file_path):
    """
    备份并处理指定的Excel文件。
    它会提取'ori_prediction'和'prediction'列中被 "['...']" 包裹的字符串。

    Args:
        file_path (str): 要处理的Excel文件路径。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。")
        return

    # 1. 备份原始文件
    backup_file = file_path + '.bak'
    try:
        shutil.copy(file_path, backup_file)
        print(f"文件已成功备份到: {backup_file}")
    except Exception as e:
        print(f"备份文件时出错: {e}")
        return

    # 2. 读取 Excel 文件
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return
        
    # 定义要处理的列
    columns_to_process = ["ori_prediction", "prediction"]

    # 检查列是否存在于DataFrame中
    for col in columns_to_process:
        if col not in df.columns:
            print(f"警告：列 '{col}' 在文件中不存在，将跳过处理该列。")
            columns_to_process.remove(col)

    # 定义提取函数
    def extract_string_from_list_format(value):
        """
        如果值是 "['...']" 格式的字符串, 则提取其中的内容。
        否则, 返回原始值。
        """
        # 仅当值是字符串且符合特定格式时处理
        if isinstance(value, str) and value.startswith("['") and value.endswith("']"):
            try:
                # 使用 ast.literal_eval 安全地将字符串'['...']'转换为真实的列表
                list_val = ast.literal_eval(value)
                # 检查转换结果是否为只有一个元素的列表
                if isinstance(list_val, list) and len(list_val) == 1:
                    # 返回列表中的第一个元素
                    return list_val[0]
            except (ValueError, SyntaxError):
                # 如果字符串不是有效的Python字面量（例如格式损坏），则返回原始值
                return value
        if isinstance(value, str) and value.startswith('["') and value.endswith('"]'):
            try:
                list_val = ast.literal_eval(value)
                if isinstance(list_val, list) and len(list_val) == 1:
                    return list_val[0]
            except (ValueError, SyntaxError):
                return value
        # 如果不符合格式，直接返回原始值
        return value

    # 3. 对指定的列应用提取函数
    print(f"开始处理列: {', '.join(columns_to_process)}...")
    for col in columns_to_process:
        df[col] = df[col].apply(extract_string_from_list_format)
    
    print("处理完成。")

    # 4. 将处理后的数据保存回原始文件
    try:
        # index=False 表示不将DataFrame的行索引写入Excel文件
        df.to_excel(file_path, index=False)
        print(f"已将更新后的数据保存回: {file_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    # Args that only apply to Video Dataset
    parser.add_argument('--nframe', type=int, default=8)
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--use-subtitle', action='store_true')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Rerun: will remove all evaluation temp files
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference. ')
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')

    args = parse_args()
    assert len(args.data), '--data should be a list of data files'

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if args.retry is not None:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))

    for _, model_name in enumerate(args.model):
        model = None

        pred_root = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root, exist_ok=True)

        for _, dataset_name in enumerate(args.data):
            dataset_kwargs = {}
            if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                dataset_kwargs['model'] = model_name
            if dataset_name == 'MMBench-Video':
                dataset_kwargs['pack'] = args.pack
            if dataset_name == 'Video-MME':
                dataset_kwargs['use_subtitle'] = args.use_subtitle

            # If distributed, first build the dataset on the main process for doing preparation works
            if world_size > 1:
                dataset = build_dataset(dataset_name, **dataset_kwargs) if rank == 0 else None
                dist.barrier()
                dataset_list = [dataset]
                dist.broadcast_object_list(dataset_list, src=0)
                dataset = dataset_list[0]
            else:
                dataset = build_dataset(dataset_name, **dataset_kwargs)
            if dataset is None:
                logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                continue

            result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'
            if dataset_name in ['MMBench-Video']:
                packstr = 'pack' if args.pack else 'nopack'
                result_file = f'{pred_root}/{model_name}_MMBench-Video_{args.nframe}frame_{packstr}.xlsx'
            elif dataset.MODALITY == 'VIDEO':
                if args.pack:
                    logger.info(f'{dataset_name} not support Pack Mode, directly change to unpack')
                    args.pack = False
                packstr = 'pack' if args.pack else 'nopack'
                result_file = f'{pred_root}/{model_name}_{dataset_name}_{args.nframe}frame_{packstr}.xlsx'
                if dataset_name in ['Video-MME']:
                    subtitlestr = 'subs' if args.use_subtitle else 'nosubs'
                    result_file = result_file.replace('.xlsx', f'_{subtitlestr}.xlsx')

            if dataset.TYPE == 'MT':
                result_file = result_file.replace('.xlsx', '.tsv')

            if osp.exists(result_file) and args.rerun:
                for keyword in ['openai', 'gpt', 'auxmatch']:
                    os.system(f'rm {pred_root}/{model_name}_{dataset_name}_{keyword}*')

            if model is None:
                model = model_name  # which is only a name

            print(f'Running {model_name} on {dataset_name} dataset, result file: {result_file}, modality: {dataset.MODALITY}, type: {dataset.TYPE}', flush=True)

            # Perform the Inference
            if dataset.MODALITY == 'VIDEO':
                model = infer_data_job_video(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    nframe=args.nframe,
                    pack=args.pack,
                    verbose=args.verbose,
                    subtitle=args.use_subtitle,
                    api_nproc=args.nproc,
                    batch_size=args.batch_size,)
            elif dataset.TYPE == 'MT':
                model = infer_data_job_mt(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=args.nproc,
                    ignore_failed=args.ignore,
                    batch_size=args.batch_size)
            else:
                model = infer_data_job(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=args.nproc,
                    ignore_failed=args.ignore,
                    batch_size=args.batch_size)

            # Set the judge kwargs first before evaluation or dumping
            judge_kwargs = {
                'nproc': args.nproc,
                'verbose': args.verbose,
            }
            if args.retry is not None:
                judge_kwargs['retry'] = args.retry
            if args.judge is not None:
                judge_kwargs['model'] = args.judge
            else:
                if dataset.TYPE in ['MCQ', 'Y/N']:
                    judge_kwargs['model'] = 'chatgpt-0125'
                elif listinstr(['MMVet', 'LLaVABench', 'MMBench-Video'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'
                elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT'], dataset_name):  # noqa: E501
                    # print(f"using gpt-4o-mini to evaluate {dataset_name}")
                    judge_kwargs['model'] = 'gpt-4o-mini'
                elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'
            if 'OPENAI_API_KEY_JUDGE' in os.environ and len(os.environ['OPENAI_API_KEY_JUDGE']):
                judge_kwargs['key'] = os.environ['OPENAI_API_KEY_JUDGE']
            if 'OPENAI_API_BASE_JUDGE' in os.environ and len(os.environ['OPENAI_API_BASE_JUDGE']):
                judge_kwargs['api_base'] = os.environ['OPENAI_API_BASE_JUDGE']

            if rank == 0:
                if dataset_name in ['MMMU_TEST']:
                    result_json = MMMU_result_transfer(result_file)
                    logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                f'json file saved in {result_json}')  # noqa: E501
                    continue
                elif 'MMT-Bench_ALL' in dataset_name:
                    submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                    logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                f'submission file saved in {submission_file}')  # noqa: E501
                    continue
                elif 'MLLMGuard_DS' in dataset_name:
                    logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')  # noqa: E501
                    continue
                elif 'AesBench_TEST' == dataset_name:
                    logger.info(f'The results are saved in {result_file}. '
                                f'Please send it to the AesBench Team via huangyipo@hotmail.com.')  # noqa: E501
                    continue

            if dataset_name in [
                'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
            ]:
                if not MMBenchOfficialServer(dataset_name):
                    logger.error(
                        f'Can not evaluate {dataset_name} on non-official servers, '
                        'will skip the evaluation. '
                    )
                    continue

            process_and_save_excel(result_file)

            eval_proxy = os.environ.get('EVAL_PROXY', None)
            old_proxy = os.environ.get('HTTP_PROXY', '')

            if rank == 0 and args.mode == 'all':
                if eval_proxy is not None:
                    proxy_set(eval_proxy)

                eval_results = dataset.evaluate(result_file, **judge_kwargs)
                if eval_results is not None:
                    assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                    logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                    logger.info('Evaluation Results:')
                if isinstance(eval_results, dict):
                    logger.info('\n' + json.dumps(eval_results, indent=4))
                elif isinstance(eval_results, pd.DataFrame):
                    if len(eval_results) < len(eval_results.columns):
                        eval_results = eval_results.T
                    logger.info('\n' + tabulate(eval_results))

                if eval_proxy is not None:
                    proxy_set(old_proxy)


if __name__ == '__main__':
    load_env()
    main()
