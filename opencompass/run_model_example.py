import argparse
import os
import subprocess
import torch
import time
try:
    from mb_internal_scripts.core_lists import *
except:
    print('未找到mb_internal_scripts.custom_dataset_configs')

cur_path = os.path.abspath(os.path.dirname(__file__))

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='运行模型评测')
parser.add_argument('-r', '--reuse',  
                    nargs='?',
                    type=str,
                    const='latest', help='是否续用已推理结果')
parser.add_argument('-l', '--task_list', type=str, default='1', help='评测任务编号')
parser.add_argument('-w', '--save-dir-name', type=str, default='', help='保存文件路径')
parser.add_argument('-d', '--debug', action='store_true', help='开启debug模式')
parser.add_argument('-if', '--infer_type', type=str, default='vllm_general_sft', help='推理类型')
parser.add_argument('-data', '--dataset', type=str, default='', help='数据集')
parser.add_argument('-mp', '--model_path', type=str, default='', help='模型路径')
parser.add_argument('-jm', '--judge_model', type=str, default='', help='judge模型')
parser.add_argument('-key', '--judge_key', type=str, default='', help='judge模型api_key')

args, args_opencompass = parser.parse_known_args()


DEVICES = []

# 可以外部使用 CUDA 定义GPU，如 CUDA=0,1 python run_model.py
CUDA = os.environ.get('CUDA')
if CUDA:
    DEVICES = CUDA.split(',')
else:
    if not DEVICES:
        DEVICES = [str(i) for i in range(torch.cuda.device_count())]

print('OPENCOMPASS_RUN_DIR', os.environ['OPENCOMPASS_RUN_DIR'])

ROOT_CACHE_PREFIX = os.environ.get('ROOT_CACHE_PREFIX', './')
OUTPUT_DIR = os.path.join(os.environ.get('OPENCOMPASS_RUN_DIR', './'), 'outputs')
os.environ['COMPASS_DATA_CACHE'] = os.path.join(ROOT_CACHE_PREFIX, 'opencompass/')
os.environ['COMPASS_INTERNAL_CONFIG_DIR'] = f'{cur_path}/opencompass/configs/mb_internal'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

os.environ['GPU_TYPE'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'  # 记录本次评测GPU型号
os.environ['GPU_NUM'] = str(len(DEVICES))  # 记录本次评测GPU总数量
os.environ['LOCAL_RANK'] = '0'

os.environ['JUDGE_MODEL'] = '' # 设置judge模型
os.environ['NLTK_DATA'] = os.path.join(ROOT_CACHE_PREFIX, 'nltk_data')  # 设置NLTK数据路径，常用于IFEval评测
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(ROOT_CACHE_PREFIX, 'tiktoken_cache')  # tiktoken缓存路径

# 使用命令行参数控制任务编号, 默认为1
print('TASK_LISTS', args.task_list)
print('MAX_OUT_LEN',int(os.environ.get('MAX_OUT_LEN', 65536)))
print('BATCH_SIZE',int(os.environ.get('BATCH_SIZE', 32)))
print('LOCAL_PATH',os.environ.get('LOCAL_PATH', ''))
print('LOCAL_RANK',os.environ.get('LOCAL_RANK', ''))
os.system('pip uninstall hf_xet -y')


# 通过task_num可以区分不同的任务列表，方便进行并行测试的场景
if args.dataset and args.model_path:
    from mb_internal_scripts.task_lists import task_list_parser
    task_list = []
    for model_path in args.model_path.split(','):
        raw_task_list = [[args.infer_type, args.dataset, model_path]]
        task_list.extend(task_list_parser(raw_task_list))
elif args.task_list == '1':
    from mb_internal_scripts.task_lists import task_list1 as task_list
elif args.task_list == '2':
    from mb_internal_scripts.task_lists import task_list2 as task_list
elif args.task_list == '3':
    from mb_internal_scripts.task_lists import task_list3 as task_list

def run_eval_command(com):
    """执行命令，实时显示输出并提取工作目录"""
    import sys
    work_dir = None
    try:
        # 使用Popen实现实时输出
        process = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # 实时读取并显示输出，同时寻找工作目录标识符
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())  # 实时打印每一行
            sys.stdout.flush()    # 确保立即显示
            
            # 检查是否包含工作目录信息
            if 'OPENCOMPASS_WORK_DIR' in line and ':' in line:
                work_dir = line.split(':', 1)[1].strip()
        
        process.wait()  # 等待进程结束
        
        # 检查返回码，如果非零则表示执行失败
        if process.returncode != 0:
            error_msg = f"评测命令执行失败，退出码: {process.returncode}"
            if work_dir:
                error_msg += f"\n详细日志请查看工作目录: {work_dir}"
            else:
                error_msg += f"\n执行的命令: {com}"
            print(f"错误: {error_msg}")
            raise RuntimeError(error_msg)
        
        return work_dir
        
    except subprocess.SubprocessError as e:
        error_msg = f"子进程执行出错: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

for task in task_list:
    print(f"task: {task}")
    infer_type, data_type, model_path = task[0], task[1], task[2]
    model_path = os.path.normpath(model_path)
    if len(task) > 3:
        re_run_config = task[3]
    else:
        re_run_config = None
    datasets = datasets_map.get(data_type, [data_type])
    max_num_workers = 16

    for dataset_idx, dataset in enumerate(datasets):
        is_first_dataset = (dataset_idx == 0)
        is_last_dataset = (dataset_idx == len(datasets) - 1)
        # 优化max_num_workers设置
        if 'livecodebench' in dataset:
            max_num_workers = 8
        if 'multiple' in dataset:
            max_num_workers = 8
        if '8gpu' in infer_type:
            max_num_workers = 1
        elif '4gpu' in infer_type:
            max_num_workers = 2
        elif '2gpu' in infer_type:
            max_num_workers = 8
        com = []
        for os_environ_key, os_environ_value in os.environ.items():
            if os_environ_key in {'OPENCOMPASS_RUN_DIR', 'COMPASS_DATA_CACHE', 'HF_ENDPOINT', 'HTTP_PROXY', 'HTTPS_PROXY', 'OPENAI_BASE_URL', 'OPENAI_API_KEY', 'JUDGE_MODEL', 'BATCH_SIZE', 'VLLM_USE_V1', 'TIKTOKEN_CACHE_DIR'}:
                com.append(f'{os_environ_key}={os_environ_value}')

        com.extend([
            f'CUDA_VISIBLE_DEVICES={",".join(DEVICES)}',
            f'TOKENIZER_MODEL={model_path}',
            f'python3 {cur_path}/run.py --models {infer_type} ',
            f'--datasets {dataset}  --max-num-workers {max_num_workers} --dump-eval-details',
            *args_opencompass
        ])
        com = ' '.join(com)

        # 添加 local_path 本地模型路径
        if model_path and ('minicpm3' in com or 'ensemble' in com):
            save_path = os.path.basename(model_path)
            com += f' --local_path {model_path}'
            if args.save_dir_name:
                com += f' -w {OUTPUT_DIR}/{args.save_dir_name}/{save_path}'
            else:
                com += f' -w {OUTPUT_DIR}/minicpm3/{save_path}'
        elif model_path:
            save_path = os.path.basename(model_path)
            com += f' --local_path {model_path}'
            if args.save_dir_name:
                if args.save_dir_name.startswith('/'):
                    com += f' -w {args.save_dir_name}/{save_path}'
                else:
                    com += f' -w {OUTPUT_DIR}/{args.save_dir_name}/{save_path}'
            else:
                com += f' -w {OUTPUT_DIR}/{save_path}'
        else:
            com += f' -w {OUTPUT_DIR}/{infer_type}'
        
        if re_run_config:
            com += f' -r {re_run_config}'
        # 是否续用已推理结果
        elif args.reuse:
            if args.reuse == 'latest':
                com += ' -r latest'
            else:
                com += f' -r {args.reuse}'
        
        if args.debug:
            com += ' --debug'
            
        # 是否使用内部评测配置
        if os.environ.get('COMPASS_INTERNAL_CONFIG_DIR'):
            com += f' --config-dir {os.environ.get("COMPASS_INTERNAL_CONFIG_DIR")}'
        # 打印当前评测命令
        print('=' * 100)
        print('Now Running:')
        print(com)
        os.environ['EVAL_TASK_COMMAND'] = com
        print('=' * 100)
        
        try:        
            # os.system(com)
            # work_dir = None
            work_dir = run_eval_command(com)
            print('=' * 100)
            if work_dir:
                print('如需对本次评测进行重跑/续跑，可使用如下命令：')
                # 获取时间戳目录名用于重跑
                work_dir_basename = os.path.basename(work_dir.rstrip('/'))
                rerun_com = com + f' -r {work_dir_basename}'
                # os.system(rerun_com)
                print(rerun_com)
            else:
                print(com)
            print('=' * 100)
            os.system(f'python {cur_path}/mb_internal_scripts/filter_recent_results.py')
            # os.system('python mb_internal_scripts/filter_recent_results.py 4 && python auto_send_feishu_table.py ')

        except RuntimeError as e:
            error_context = f"评测失败 - 模型: {os.path.basename(model_path)}, 数据集: {dataset}"
            print('=' * 100)
            print(f"❌ 错误: {error_context}")
            print(f"❌ 详细信息: {str(e)}")
            print('=' * 100)
            
            # 如果是第一个或最后一个数据集，或者只有一个数据集，抛出异常终止程序
            # if is_first_dataset or is_last_dataset:
            #     print(f"❌ 由于数据集{dataset}失败，终止后续评测任务")
            #     raise RuntimeError(f"{error_context}\n{str(e)}")
            # else:
            #     print(f"⚠️  跳过当前数据集，继续执行后续数据集")
                # continue
        # 判断work_dir目录下是否存在'results'文件夹，以此判断是否正常完成评测
        results_dir = os.path.join(work_dir, 'results')
        if os.path.isdir(results_dir):
            print(f"✅ 评测已完成，结果文件夹存在: {results_dir}")
        else:
            print(f"⚠️ 警告：未检测到结果文件夹 {results_dir}，评测未正常完成！")
            if is_last_dataset:
                raise RuntimeError(f"评测未正常完成！")
                
        time.sleep(1)
