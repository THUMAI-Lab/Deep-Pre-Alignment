import copy
import glob
import os
import os.path as osp
import subprocess
import uuid
import time
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from typing import Any, Dict, List, Tuple

import mmengine
import numpy as np
from tqdm import tqdm

from opencompass.registry import RUNNERS
from opencompass.utils import get_logger, model_abbr_from_cfg
from mmengine.device import is_npu_available
# 避免循环导入，直接在这里定义函数

from .base import BaseRunner
from ..tasks import TASKS


def generate_short_log_filename(task_idx, node_idx, gpu_idx=None, suffix="out"):
    """生成简短的日志文件名格式: taskX_nodeY_gpuZ.out"""
    if gpu_idx is not None:
        return f"task{task_idx}_node{node_idx}_gpu{gpu_idx}.{suffix}"
    else:
        return f"task{task_idx}_node{node_idx}.{suffix}"


def write_log_header(log_file, task_name, model_info, dataset_info, node_idx, gpu_idx=None):
    """在日志文件开头写入任务详细信息"""
    header_lines = [
        "=" * 80,
        "OpenCompass Task Execution Log", 
        "=" * 80,
        f"Task Name: {task_name}",
        f"Node Index: {node_idx}",
    ]
    
    if gpu_idx is not None:
        header_lines.append(f"GPU Index: {gpu_idx}")
    
    header_lines.extend([
        f"Model Info: {model_info}",
        f"Dataset Info: {dataset_info}",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        ""
    ])
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(header_lines) + '\n')
        f.flush()


def get_multinode_command_template(gpu_ids: List[int], 
                                 master_addr: str = None, 
                                 master_port: str = None,
                                 world_size: int = None,
                                 rank: int = None,
                                 nproc_per_node: int = None) -> str:
    """Format command template for multi-node distributed training."""
    import sys
    
    if is_npu_available():
        device_env = 'ASCEND_RT_VISIBLE_DEVICES=' + ','.join(str(i) for i in gpu_ids)
    elif sys.platform == 'win32':
        device_env = 'set CUDA_VISIBLE_DEVICES=' + ','.join(str(i) for i in gpu_ids) + ' &'
    else:
        device_env = 'CUDA_VISIBLE_DEVICES=' + ','.join(str(i) for i in gpu_ids)
    
    # 构建多节点环境变量
    env_vars = [device_env]
    if master_addr:
        env_vars.append(f'MASTER_ADDR={master_addr}')
    if master_port:
        env_vars.append(f'MASTER_PORT={master_port}')
    if world_size is not None:
        env_vars.append(f'WORLD_SIZE={world_size}')
    if rank is not None:
        env_vars.append(f'RANK={rank}')
    if nproc_per_node is not None:
        env_vars.append(f'NPROC_PER_NODE={nproc_per_node}')
    
    tmpl = ' '.join(env_vars) + ' {task_cmd}'
    return tmpl


@RUNNERS.register_module()
class PytorchJobRunner(BaseRunner):
    """PyTorch Job Runner，支持正确的多节点推理和评估流程.
    
    专为 PyTorchJob 环境设计，支持：
    1. 推理阶段：各节点处理数据片段，Master 节点合并预测结果
    2. 评估阶段：只在 Master 节点基于合并后的完整预测结果进行评估
    """

    def __init__(self,
                 task,
                 debug=False,
                 lark_bot_url=None,
                 max_num_workers=32,
                 max_workers_per_gpu=4,
                 keep_tmp_file=False,
                 gpus_per_node=8,
                 **kwargs):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.max_num_workers = max_num_workers
        self.max_workers_per_gpu = max_workers_per_gpu
        self.keep_tmp_file = keep_tmp_file
        self.gpus_per_node = gpus_per_node
        
        # 从环境变量获取多节点信息
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.rank = int(os.getenv('RANK', '0'))
        self.master_addr = os.getenv('MASTER_ADDR', 'localhost')
        self.master_port = os.getenv('MASTER_PORT', '23456')
        self.nproc_per_node = int(os.getenv('NPROC_PER_NODE', str(self.gpus_per_node)))
        
        # 确定当前节点是否为 master
        self.is_master = (self.rank == 0)
        
        logger = get_logger()
        logger.info(f'PytorchJobRunner 初始化: '
                   f'world_size={self.world_size}, '
                   f'rank={self.rank}, '
                   f'master_addr={self.master_addr}, '
                   f'master_port={self.master_port}, '
                   f'is_master={self.is_master}')
        
        for k, v in kwargs.items():
            logger.warning(f'忽略参数 {self.__module__}: {k}={v}')

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """启动多个任务."""
        import signal
        import torch
        
        print(f"🔍 [TRACE] PytorchJobRunner.launch() 开始, 任务数: {len(tasks)}")
        
        # 在主线程中设置全局中断标志和信号处理器
        self.interrupted = [False]
        
        def signal_handler(signum, frame):
            logger = get_logger()
            logger.info('🛑 收到Ctrl+C中断信号，正在停止所有任务...')
            print('🛑 收到Ctrl+C中断信号，正在停止所有任务...')
            self.interrupted[0] = True
        
        # 在主线程中注册信号处理器
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            status = []

            # 获取可用设备信息
            if is_npu_available():
                visible_devices = 'ASCEND_RT_VISIBLE_DEVICES'
                device_nums = torch.npu.device_count()
            else:
                visible_devices = 'CUDA_VISIBLE_DEVICES'
                device_nums = torch.cuda.device_count()
            
            if visible_devices in os.environ:
                all_gpu_ids = [
                    int(i) for i in re.findall(r'(?<!-)\d+', os.environ[visible_devices])
                ]
            else:
                all_gpu_ids = list(range(device_nums))

            logger = get_logger()
            logger.info(f'节点 {self.rank} 可用设备: {all_gpu_ids}')
            
            # 创建GPU资源管理器（类似LocalRunner）
            if len(all_gpu_ids) > 0:
                gpus = np.zeros(max(all_gpu_ids) + 1, dtype=np.uint)
                gpus[all_gpu_ids] = self.max_workers_per_gpu
                logger.info(f'节点 {self.rank} GPU资源初始化: {dict(zip(all_gpu_ids, gpus[all_gpu_ids]))}')
            else:
                gpus = np.array([], dtype=np.uint)

            # 判断任务类型
            task_type = self.task_cfg['type']
            print(f"🔍 [TRACE] 任务类型: {task_type}, 是否Master: {self.is_master}")
            logger.info(f'🔍 Debug: 任务类型={task_type}, 是否Master={self.is_master}')
            
            if 'OpenICLEvalTask' in task_type:
                print(f"🔍 [TRACE] 进入评估任务分支")
                # 🔧 简化：只使用原始OpenICLEvalTask，在多节点环境中只有Master节点运行评估
                if self.world_size > 1:
                    # 多节点环境：只有Master节点执行评估
                    if self.is_master:
                        logger.info('Master 节点执行评估任务')
                        status = self._run_eval_tasks_on_master(tasks, task_type)
                    else:
                        logger.info(f'Worker 节点 {self.rank} 跳过评估任务，等待Master节点完成')
                        status = []  # Worker节点返回空状态，表示跳过评估
                else:
                    # 单节点环境：直接执行评估
                    logger.info('单节点环境，执行评估任务')
                    status = self._run_eval_tasks_on_master(tasks, task_type)
            else:
                print(f"🔍 [TRACE] 进入推理任务分支")
                # 推理任务：检查是否为多节点场景（数据集名包含_node或有task_idx）
                is_multinode = any('_node' in str(task.get('datasets', [])) or 'task_idx' in task for task in tasks)
                
                import time
                print(f"🔍 [TRACE] is_multinode检查: {is_multinode}, 任务数: {len(tasks)}")
                logger.info(f'🔍 [DEBUG] 推理任务路由检查: is_multinode={is_multinode}, 任务数={len(tasks)}, 时间={time.strftime("%H:%M:%S")}')
                
                if is_multinode:
                    print(f"🔍 [TRACE] 走多节点路径: _run_multinode_tasks_with_merge")
                    # 多节点场景：每个节点处理多个任务，需要节点内合并
                    logger.info(f'多节点场景：节点{self.rank}处理{len(tasks)}个任务，需要节点内合并')
                    status = self._run_multinode_tasks_with_merge(tasks, task_type, all_gpu_ids, gpus)
                else:
                    print(f"🔍 [TRACE] 走常规路径: _run_tasks_with_gpu_management")
                    # 常规场景：使用GPU资源管理进行并发执行
                    if self.debug:
                        status = self._run_tasks_debug_mode(tasks, task_type, all_gpu_ids)
                    else:
                        status = self._run_tasks_with_gpu_management(tasks, task_type, all_gpu_ids, gpus)
            
            # 如果是推理任务且是主节点，等待所有节点完成后合并预测结果
            if self.is_master and 'OpenICLInferTask' in task_type and tasks:
                logger.info('🚀 Master 节点开始执行预测合并流程...')
                logger.info('Master 节点等待所有节点完成推理...')
                self._wait_for_all_nodes_completion(tasks)
                logger.info('Master 节点开始合并多节点预测结果...')
                self._merge_multinode_predictions(tasks)
                logger.info('✅ Master 节点预测合并完成！')
            elif self.is_master and 'OpenICLInferTask' in task_type and not tasks:
                logger.info('🎯 Master 节点没有推理任务需要执行，跳过预测合并流程')
            
        except KeyboardInterrupt:
            logger.info('🛑 程序被Ctrl+C中断')
            print('🛑 程序被Ctrl+C中断')
            return []
        finally:
            # 恢复原始信号处理器
            signal.signal(signal.SIGINT, original_handler)
        
        return status

    def _modify_eval_task_for_merged_predictions(self, task_cfg):
        """修改评估任务配置，使其使用合并后的预测结果."""
        modified_cfg = copy.deepcopy(task_cfg)
        
        # 移除数据集配置中的节点后缀，使用原始数据集名称
        for dataset_list in modified_cfg['datasets']:
            for dataset_cfg in dataset_list:
                if '_node' in dataset_cfg['abbr']:
                    # 恢复原始数据集名称
                    dataset_cfg['abbr'] = dataset_cfg['abbr'].rsplit('_node', 1)[0]
                # 移除 test_range 限制，使用完整数据集进行评估
                if 'reader_cfg' in dataset_cfg and 'test_range' in dataset_cfg['reader_cfg']:
                    del dataset_cfg['reader_cfg']['test_range']
        
        return modified_cfg

    def _wait_for_all_nodes_completion(self, tasks):
        """等待所有节点完成推理任务."""
        import time
        import os.path as osp
        import signal
        
        logger = get_logger()
        logger.info(f'Master 节点等待所有 {self.world_size} 个节点完成推理...')
        
        # 添加信号处理，允许优雅中断
        interrupted = [False]  # 使用list以便在内嵌函数中修改
        
        def signal_handler(signum, frame):
            logger.info('收到中断信号，停止等待节点完成...')
            interrupted[0] = True
        
        # 注册信号处理器
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # 获取预期的预测文件路径
            expected_files = []
            for task in tasks:
                task_obj = TASKS.build(dict(cfg=task, type='OpenICLInferTask'))
                
                # 搜索所有可能的工作目录（跨时间戳目录）
                work_base_dir = task.get('work_dir', './outputs')
                # 查找所有时间戳目录下的predictions目录
                timestamp_dirs = glob.glob(f'{work_base_dir}/*/predictions')
                
                # 记录所有可能的工作目录
                possible_work_dirs = []
                if timestamp_dirs:
                    for pred_dir in timestamp_dirs:
                        possible_work_dirs.append(os.path.dirname(pred_dir))
                else:
                    possible_work_dirs.append(work_base_dir)
                
                # 为每个节点生成预期的预测文件路径
                for rank in range(self.world_size):
                    # 安全地获取模型和数据集信息
                    try:
                        model_abbr = task_obj.model_cfgs[0]['abbr']
                        dataset_cfg = task_obj.dataset_cfgs[0] if isinstance(task_obj.dataset_cfgs[0], dict) else task_obj.dataset_cfgs[0][0]
                        dataset_abbr = dataset_cfg['abbr']
                    except (IndexError, KeyError, TypeError) as e:
                        logger.warning(f'获取任务配置失败: {e}，跳过等待')
                        continue
                    
                    # 移除当前节点后缀，获取原始数据集名称
                    original_dataset_abbr = dataset_abbr.rsplit('_node', 1)[0] if '_node' in dataset_abbr else dataset_abbr
                    
                    # 在所有可能的工作目录中搜索预测文件
                    for work_dir in possible_work_dirs:
                        pred_file = f'{work_dir}/predictions/{model_abbr}/{original_dataset_abbr}_node{rank}.json'
                        if pred_file not in expected_files:  # 避免重复
                            expected_files.append(pred_file)
            
            # 等待所有预测文件生成
            max_wait_time = 3600  # 最大等待10分钟
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time and not self.interrupted[0]:
                # 智能搜索预测文件，支持同时启动的多节点
                current_expected_files = []
                work_base_dir = tasks[0].get('work_dir', './outputs') if tasks else './outputs'
                
                # 多种搜索模式：
                # 1. 当前工作目录
                # 2. 同名的其他时间戳目录 
                # 3. 共享目录模式
                search_patterns = [
                    f'{work_base_dir}/predictions',
                    f'{work_base_dir}/*/predictions', 
                    './outputs/*/predictions'
                ]
            
                timestamp_dirs = []
                for pattern in search_patterns:
                    timestamp_dirs.extend(glob.glob(pattern))
                
                # 去重并排序（优先使用最新的）
                timestamp_dirs = sorted(set(timestamp_dirs), key=lambda x: os.path.getctime(x) if osp.exists(x) else 0, reverse=True)
                
                for task in tasks:
                    task_obj = TASKS.build(dict(cfg=task, type='OpenICLInferTask'))
                    try:
                        model_abbr = task_obj.model_cfgs[0]['abbr']
                        dataset_cfg = task_obj.dataset_cfgs[0] if isinstance(task_obj.dataset_cfgs[0], dict) else task_obj.dataset_cfgs[0][0]
                        dataset_abbr = dataset_cfg['abbr']
                        original_dataset_abbr = dataset_abbr.rsplit('_node', 1)[0] if '_node' in dataset_abbr else dataset_abbr
                        
                        for rank in range(self.world_size):
                            # 在所有可能的目录中搜索
                            for pred_dir in timestamp_dirs:
                                work_dir = os.path.dirname(pred_dir)
                                pred_file = f'{work_dir}/predictions/{model_abbr}/{original_dataset_abbr}_node{rank}.json'
                                if pred_file not in current_expected_files:
                                    current_expected_files.append(pred_file)
                    except Exception:
                        continue
                
                # 检查哪些文件存在
                existing_files = [f for f in current_expected_files if osp.exists(f)]
                
                # 按节点分组检查，确保每个节点都有对应的预测文件
                nodes_completed = set()
                for existing_file in existing_files:
                    # 从文件路径中提取节点编号
                    if '_node' in existing_file:
                        for rank in range(self.world_size):
                            if f'_node{rank}.json' in existing_file:
                                nodes_completed.add(rank)
                                break
                
                if len(nodes_completed) >= self.world_size:
                    logger.info(f'✅ 所有 {self.world_size} 个节点推理完成，预测文件齐全')
                    logger.info(f'找到的预测文件: {[osp.basename(f) for f in existing_files]}')
                    # 保存找到的文件路径供合并使用
                    self._found_prediction_files = existing_files
                    return True
                
                missing_nodes = set(range(self.world_size)) - nodes_completed
                logger.info(f'等待 {len(missing_nodes)} 个节点完成推理: 节点 {list(missing_nodes)}')
                
                # 使用可中断的睡眠机制，每0.5秒检查一次中断信号
                for _ in range(6):  # 总共睡眠3秒（6 * 0.5秒）
                    if self.interrupted[0]:
                        logger.info('🛑 等待被用户中断，停止等待')
                        return False
                    time.sleep(0.5)
            
            # 如果被中断，提前退出
            if self.interrupted[0]:
                logger.info('🛑 等待被用户中断，停止等待')
                return False
            
            # 超时处理
            missing_files = [f for f in expected_files if not osp.exists(f)]
            if missing_files:
                logger.warning(f'等待超时，仍有 {len(missing_files)} 个预测文件未生成')
                return False
            
            return True
            
        except KeyboardInterrupt:
            logger.info('用户按下Ctrl+C，中断等待节点完成')
            return False
        finally:
            # 恢复原始信号处理器
            signal.signal(signal.SIGINT, original_handler)

    def _merge_multinode_predictions(self, tasks):
        """合并多节点的预测结果."""
        from opencompass.utils.multinode_merger import MultiNodePredictionMerger
        
        logger = get_logger()
        logger.info(f'Master 节点开始合并多节点预测结果...')
        
        # 构建模型-数据集组合
        model_dataset_combinations = []
        for task in tasks:
            task_obj = TASKS.build(dict(cfg=task, type=self.task_cfg['type']))
            # 安全地处理数据集配置格式
            datasets_cfg = task_obj.dataset_cfgs
            if isinstance(datasets_cfg, list) and len(datasets_cfg) > 0:
                # 如果是嵌套列表，取第一层
                if isinstance(datasets_cfg[0], list):
                    datasets_cfg = datasets_cfg[0]
            
            comb = {
                'models': task_obj.model_cfgs,
                'datasets': datasets_cfg
            }
            model_dataset_combinations.append(comb)
        
        # 获取工作目录
        work_dir = tasks[0].get('work_dir', './outputs') if tasks else './outputs'
        
        # 创建合并器并执行合并
        merger = MultiNodePredictionMerger(work_dir, self.world_size)
        merged_files = merger.merge_all_predictions(
            model_dataset_combinations, 
            work_dir
        )
        
        if merged_files:
            logger.info(f'成功合并 {len(merged_files)} 个预测文件')
        else:
            logger.warning('没有找到需要合并的预测文件')
        
        return merged_files
    
    def _run_eval_tasks_on_master(self, tasks, task_type):
        """在Master节点执行评估任务（原有逻辑）."""
        logger = get_logger()
        status = []
        
        for task_idx, task_cfg in enumerate(tasks):
            # 修改任务配置，使用合并后的预测结果
            modified_task_cfg = self._modify_eval_task_for_merged_predictions(task_cfg)
            
            task = TASKS.build(dict(cfg=modified_task_cfg, type=task_type))
            task_name = task.name
            
            try:
                # 创建评估日志目录
                log_dir = f'{modified_task_cfg.work_dir}/logs/eval/{model_abbr_from_cfg(task.model_cfgs[0])}'
                mmengine.mkdir_or_exist(log_dir)
                
                # 使用简短的日志文件名
                short_filename = generate_short_log_filename(task_idx, self.rank, suffix="out")
                log_file = f'{log_dir}/{short_filename}'
                
                # 获取模型和数据集信息
                model_info = model_abbr_from_cfg(task.model_cfgs[0])
                dataset_info = ", ".join([d.get('abbr', 'Unknown') if isinstance(d, dict) else str(d) 
                                        for dataset_list in task.dataset_cfgs for d in (dataset_list if isinstance(dataset_list, list) else [dataset_list])])
                
                # 写入日志头信息
                write_log_header(log_file, task_name, model_info, dataset_info, self.rank)
                
                logger.info(f'Master 节点执行评估任务 {task_name}，日志: {log_file}')
                
                # 直接运行评估任务（不需要分布式）
                import sys
                
                # 捕获输出到日志文件
                with open(log_file, 'a') as f:
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = f
                    sys.stderr = f
                    
                    try:
                        task.run()
                        status.append((task_name, 0))
                        logger.info(f'Master 节点评估任务 {task_name} 完成')
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        
            except Exception as e:
                logger.error(f'Master 节点评估任务 {task_name} 失败: {e}')
                status.append((task_name, 1))
                
        return status
    
    def _run_tasks_debug_mode(self, tasks, task_type, all_gpu_ids):
        """在Debug模式下串行执行任务."""
        logger = get_logger()
        status = []
        
        for task_idx, task_cfg in enumerate(tasks):
            task = TASKS.build(dict(cfg=task_cfg, type=task_type))
            task_name = task.name
            num_gpus = task.num_gpus
            
            try:
                # 获取命令并执行
                mmengine.mkdir_or_exist('tmp/')
                uuid_str = str(uuid.uuid4())
                param_file = f'tmp/{uuid_str}_params.py'
                
                task.cfg.dump(param_file)
                
                tmpl = get_multinode_command_template(
                    all_gpu_ids[:num_gpus],
                    self.master_addr,
                    self.master_port,
                    self.world_size,
                    self.rank,
                    self.nproc_per_node
                )
                cmd = task.get_command(cfg_path=param_file, template=tmpl)
                
                # 创建日志目录和文件
                log_dir = f'{task.cfg.work_dir}/logs'
                if task_type.endswith('InferTask'):
                    log_subdir = f'{log_dir}/infer/{model_abbr_from_cfg(task.model_cfgs[0])}'
                else:
                    log_subdir = f'{log_dir}/eval/{model_abbr_from_cfg(task.model_cfgs[0])}'
                
                mmengine.mkdir_or_exist(log_subdir)
                
                # 使用简短的日志文件名
                short_filename = generate_short_log_filename(task_idx, self.rank, suffix="debug.out")
                log_file = f'{log_subdir}/{short_filename}'
                
                # 获取模型和数据集信息
                model_info = model_abbr_from_cfg(task.model_cfgs[0])
                dataset_info = ", ".join([d.get('abbr', 'Unknown') if isinstance(d, dict) else str(d) 
                                        for dataset_list in task.dataset_cfgs for d in (dataset_list if isinstance(dataset_list, list) else [dataset_list])])
                
                # 写入日志头信息
                write_log_header(log_file, task_name, model_info, dataset_info, self.rank)
                
                logger.info(f'[调试模式] 节点 {self.rank} 执行任务 {task_name}，日志: {log_file}')
                
                # 执行命令并输出到日志文件
                # 使用Popen以支持中断
                with open(log_file, 'a') as f:
                    process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True)
                    
                    # 等待进程完成，但定期检查中断信号
                    try:
                        while process.poll() is None:
                            time.sleep(0.1)  # 短暂睡眠，避免CPU占用过高
                        result_code = process.returncode
                    except KeyboardInterrupt:
                        # 收到中断信号，终止子进程
                        logger.info(f'[调试模式] 任务 {task_name} 收到中断信号，正在终止子进程...')
                        process.terminate()
                        try:
                            process.wait(timeout=5)  # 等待5秒让进程优雅退出
                        except subprocess.TimeoutExpired:
                            process.kill()  # 强制终止
                            process.wait()
                        raise
                    
                    result = type('Result', (), {'returncode': result_code})()
                
                if result.returncode == 0:
                    status.append((task_name, 0))
                    logger.info(f'[调试模式] 节点 {self.rank} 任务 {task_name} 完成')
                else:
                    status.append((task_name, 1))
                    logger.error(f'[调试模式] 节点 {self.rank} 任务 {task_name} 失败，退出码: {result.returncode}')
                    
                # 清理临时文件
                if not self.keep_tmp_file:
                    os.remove(param_file)
                    
            except Exception as e:
                logger.error(f'[调试模式] 节点 {self.rank} 任务 {task_name} 异常: {e}')
                status.append((task_name, 1))
                
        return status
    
    def _run_tasks_with_gpu_management(self, tasks, task_type, all_gpu_ids, gpus):
        """使用GPU资源管理进行并发任务执行（类似LocalRunner）."""
        import signal
        logger = get_logger()
        status = []
        
        # 添加全局中断标志供线程池任务使用
        interrupted = [False]
        
        def signal_handler(signum, frame):
            interrupted[0] = True
            print(f'⏰ [SIGNAL] 收到中断信号，任务将停止等待GPU资源')
        
        # 注册信号处理器
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        pbar = tqdm(total=len(tasks), desc=f'节点{self.rank}任务进度')
        lock = Lock()
        
        def submit_task(task_cfg, index):
            """提交单个任务执行（类似LocalRunner.submit）."""
            import time
            print(f'⏰ [SUBMIT] 任务 {index} 开始提交: {time.strftime("%H:%M:%S.%f")[:-3]}')
            
            print(f'⏰ [BUILD] 任务 {index} 开始构建 TASKS.build: {time.strftime("%H:%M:%S.%f")[:-3]}')
            task = TASKS.build(dict(cfg=task_cfg, type=task_type))
            print(f'⏰ [BUILD] 任务 {index} 构建完成: {time.strftime("%H:%M:%S.%f")[:-3]}')
            
            task_name = task.name
            num_gpus = task.num_gpus
            print(f'⏰ [GPU_REQ] 任务 {index} 需要 {num_gpus} 个GPU: {time.strftime("%H:%M:%S.%f")[:-3]}')
            
            # 修复：检查实际可用GPU数量而不是GPU数组长度
            if len(all_gpu_ids) < num_gpus:
                raise RuntimeError(f'节点{self.rank}物理GPU数量{len(all_gpu_ids)}不足，任务需要{num_gpus}个GPU')
            
            # GPU资源等待循环（类似LocalRunner）
            print(f'⏰ [WAIT] 任务 {index} 开始等待GPU分配: {time.strftime("%H:%M:%S.%f")[:-3]}')
            wait_count = 0
            while True and not interrupted[0]:
                # print(f'⏰ [LOCK] 任务 {index} 尝试获取锁 (等待次数: {wait_count}): {time.strftime("%H:%M:%S.%f")[:-3]}')
                lock.acquire()
                # print(f'⏰ [LOCK] 任务 {index} 已获取锁: {time.strftime("%H:%M:%S.%f")[:-3]}')
                
                # 修复：正确的GPU分配逻辑
                try:
                    # 找到有可用slots的GPU列表
                    available_gpu_ids = []
                    for gpu_id in all_gpu_ids:
                        if gpus[gpu_id] > 0:  # 该GPU有可用slots
                            available_gpu_ids.append(gpu_id)
                    
                    # 检查是否有足够的GPU可用
                    if len(available_gpu_ids) >= num_gpus:
                        # 选择前num_gpus个可用GPU
                        selected_gpu_ids = available_gpu_ids[:num_gpus]
                        
                        # 验证分配是否成功（再次检查避免竞争条件）
                        allocation_success = True
                        for gpu_id in selected_gpu_ids:
                            if gpus[gpu_id] <= 0:
                                allocation_success = False
                                break
                        
                        if allocation_success:
                            # 占用GPU资源
                            for gpu_id in selected_gpu_ids:
                                gpus[gpu_id] -= 1
                            gpu_ids = np.array(selected_gpu_ids)
                            print(f'⏰ [ALLOC] 任务 {index} 分配到GPU {list(gpu_ids)}: {time.strftime("%H:%M:%S.%f")[:-3]}')
                            lock.release()
                            break
                    
                    # 如果分配失败，输出详细状态信息
                    gpu_status = {gpu_id: gpus[gpu_id] for gpu_id in all_gpu_ids}
                    # print(f'⏰ [WAIT] 任务 {index} GPU资源不足，当前状态: {gpu_status}, 需要: {num_gpus}GPU, 可用GPU: {available_gpu_ids}')
                    
                except Exception as e:
                    print(f'⏰ [ERROR] 任务 {index} GPU分配异常: {e}')
                
                lock.release()
                wait_count += 1
                
                # 使用可中断的睡眠机制
                for _ in range(10):  # 总共睡眠1秒（10 * 0.1秒）
                    if interrupted[0]:
                        print(f'⏰ [INTERRUPT] 任务 {index} 收到中断信号，停止等待GPU')
                        raise KeyboardInterrupt("GPU等待被用户中断")
                    time.sleep(0.1)
                
                # 防止无限等待，但给足够的时间
                if wait_count > 3600:  # 30分钟超时
                    raise RuntimeError(f'任务 {index} GPU资源等待超时，当前GPU状态: {dict(zip(all_gpu_ids, gpus[all_gpu_ids]))}')
            
            # 输出任务启动信息
            print(f'⏰ [EXEC] 任务 {index} 开始执行: {time.strftime("%H:%M:%S.%f")[:-3]}')
            if num_gpus > 0:
                tqdm.write(f'🚀 节点{self.rank} 在GPU {list(gpu_ids)} 上启动任务 {task_name}')
            else:
                tqdm.write(f'🚀 节点{self.rank} 在CPU上启动任务 {task_name}')
            
            # 执行任务
            res = self._launch_single_task(task, gpu_ids, index)
            print(f'⏰ [DONE] 任务 {index} 执行完成: {time.strftime("%H:%M:%S.%f")[:-3]}')
            pbar.update()
            
            # 释放GPU资源
            with lock:
                gpus[gpu_ids] += 1
                print(f'⏰ [RELEASE] 任务 {index} 释放GPU {list(gpu_ids)}: {time.strftime("%H:%M:%S.%f")[:-3]}')
            
            return res
        
        # 使用线程池并发执行任务
        import time
        try:
            print(f'⏰ [POOL] 启动线程池，max_workers={self.max_num_workers}, 任务数={len(tasks)}: {time.strftime("%H:%M:%S.%f")[:-3]}')
            with ThreadPoolExecutor(max_workers=self.max_num_workers) as executor:
                print(f'⏰ [POOL] 开始并发执行任务: {time.strftime("%H:%M:%S.%f")[:-3]}')
                status = executor.map(submit_task, tasks, range(len(tasks)))
                print(f'⏰ [POOL] executor.map() 调用完成: {time.strftime("%H:%M:%S.%f")[:-3]}')
        except KeyboardInterrupt:
            print(f'⏰ [INTERRUPT] GPU管理任务被用户中断')
            interrupted[0] = True
            status = []
        finally:
            # 恢复原始信号处理器
            signal.signal(signal.SIGINT, original_handler)
        
        pbar.close()
        return status
    
    def _launch_single_task(self, task, gpu_ids, index):
        """执行单个任务（类似LocalRunner._launch）."""
        logger = get_logger() 
        task_name = task.name
        
        try:
            # 创建临时配置文件
            mmengine.mkdir_or_exist('tmp/')
            uuid_str = str(uuid.uuid4())
            param_file = f'tmp/{uuid_str}_params.py'
            
            task.cfg.dump(param_file)
            
            # 构建命令模板
            tmpl = get_multinode_command_template(
                gpu_ids,
                self.master_addr,
                self.master_port,
                self.world_size,
                self.rank,
                self.nproc_per_node
            )
            cmd = task.get_command(cfg_path=param_file, template=tmpl)
            
            # 🔧 新增：从任务名提取GPU索引并设置CUDA_VISIBLE_DEVICES
            gpu_index = 0
            if '_gpu' in task_name:
                try:
                    # 从任务名中提取GPU索引 (例如: ARC-c-test-1_node0_gpu3 -> 3)
                    gpu_part = task_name.split('_gpu')[-1]
                    gpu_str = ''.join(filter(str.isdigit, gpu_part.split('_')[0]))
                    if gpu_str:
                        gpu_index = int(gpu_str)
                        logger.info(f'🔧 任务 {task_name} 分配到GPU {gpu_index}')
                except (ValueError, IndexError):
                    gpu_index = 0
                    logger.warning(f'⚠️ 任务 {task_name} GPU索引提取失败，使用默认GPU 0')
            
            # 设置环境变量
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
            env['PYTHONPATH'] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
            
            # 创建日志文件
            log_dir = f'{task.cfg.work_dir}/logs/infer/{model_abbr_from_cfg(task.model_cfgs[0])}'
            mmengine.mkdir_or_exist(log_dir)
            
            # 使用简短的日志文件名
            short_filename = generate_short_log_filename(index, self.rank, gpu_index, suffix="out")
            log_file = f'{log_dir}/{short_filename}'
            
            # 获取模型和数据集信息
            model_info = model_abbr_from_cfg(task.model_cfgs[0])
            dataset_info = ", ".join([d.get('abbr', 'Unknown') if isinstance(d, dict) else str(d) 
                                    for dataset_list in task.dataset_cfgs for d in (dataset_list if isinstance(dataset_list, list) else [dataset_list])])
            
            # 写入日志头信息
            write_log_header(log_file, task_name, model_info, dataset_info, self.rank, gpu_index)
            
            logger.info(f'🚀 节点{self.rank} 在GPU {gpu_index} 上执行任务 {task_name}')
            logger.debug(f'执行命令: {cmd}')
            
            # 执行命令（现在使用原始Task类 + 环境变量设置）
            # 使用Popen以支持中断
            import signal
            with open(log_file, 'a', encoding='utf-8') as f:
                process = subprocess.Popen(cmd, shell=True, text=True, stdout=f, stderr=subprocess.STDOUT, env=env)
                
                # 等待进程完成，但定期检查中断信号
                try:
                    while process.poll() is None:
                        time.sleep(0.1)  # 短暂睡眠，避免CPU占用过高
                    result_code = process.returncode
                except KeyboardInterrupt:
                    # 收到中断信号，终止子进程
                    logger.info(f'任务 {task_name} 收到中断信号，正在终止子进程...')
                    process.terminate()
                    try:
                        process.wait(timeout=5)  # 等待5秒让进程优雅退出
                    except subprocess.TimeoutExpired:
                        process.kill()  # 强制终止
                        process.wait()
                    raise
                
                result = type('Result', (), {'returncode': result_code})()
            
            if result.returncode != 0:
                logger.error(f'节点{self.rank} 任务 {task_name} 失败，查看日志: {log_file}')
                
            # 清理临时文件
            if not self.keep_tmp_file:
                os.remove(param_file)
                
            return task_name, result.returncode
            
        except Exception as e:
            logger.error(f'节点{self.rank} 任务 {task_name} 异常: {e}')
            return task_name, 1
    
    def _cleanup_old_single_task_method(self):
        # 这个方法已被新的两级合并架构取代
        pass
    
    def _run_multinode_tasks_with_merge(self, tasks, task_type, all_gpu_ids, gpus):
        """多节点场景下运行多个任务，然后进行节点内合并."""
        logger = get_logger()
        import time
        print(f'⏰ [MERGE] _run_multinode_tasks_with_merge 开始: {time.strftime("%H:%M:%S.%f")[:-3]}, 任务数: {len(tasks)}, debug={self.debug}')
        logger.info(f'⏰ [DEBUG] _run_multinode_tasks_with_merge 开始: {time.strftime("%H:%M:%S")}, 任务数: {len(tasks)}, debug={self.debug}')
        
        # 首先运行所有任务
        if self.debug:
            status = self._run_tasks_debug_mode(tasks, task_type, all_gpu_ids)
        else:
            status = self._run_tasks_with_gpu_management(tasks, task_type, all_gpu_ids, gpus)
        
        # 然后进行节点内合并
        try:
            self._merge_node_internal_predictions(tasks)
            logger.info(f'节点 {self.rank} 内部合并完成')
        except Exception as e:
            logger.error(f'节点 {self.rank} 内部合并失败: {e}')
        
        return status
    
    def _merge_node_internal_predictions(self, tasks):
        """合并节点内部的多个 GPU 任务的预测结果."""
        logger = get_logger()
        logger.info(f'开始节点 {self.rank} 内部合并...')
        
        # 按模型+数据集分组任务
        dataset_task_groups = {}
        
        for task in tasks:
            task_obj = TASKS.build(dict(cfg=task, type='OpenICLInferTask'))
            if not task_obj.dataset_cfgs or not task_obj.dataset_cfgs[0] or not task_obj.model_cfgs:
                continue
            
            # 获取模型信息
            model_cfg = task_obj.model_cfgs[0]
            model_name = model_cfg.get('abbr', 'unknown_model')
            
            # 处理一个任务中的多个数据集
            dataset_cfgs_list = task_obj.dataset_cfgs[0] if isinstance(task_obj.dataset_cfgs[0], list) else [task_obj.dataset_cfgs[0]]
            
            for dataset_cfg in dataset_cfgs_list:
                dataset_abbr = dataset_cfg['abbr']
                
                # 提取原始数据集名（去除 _node{rank}_gpu{idx} 后缀）
                if '_node' in dataset_abbr and '_gpu' in dataset_abbr:
                    # 例如：data_node0_gpu1 -> data
                    original_name = dataset_abbr.split('_node')[0]
                    node_rank = self.rank
                    
                    # 包含模型信息的key，确保每个模型的数据集分开合并
                    key = (model_name, original_name, node_rank)
                    if key not in dataset_task_groups:
                        dataset_task_groups[key] = []
                    dataset_task_groups[key].append((task_obj, dataset_cfg))
        
        # 对每个模型的每个数据集进行节点内合并
        for (model_name, original_name, node_rank), task_group in dataset_task_groups.items():
            self._merge_single_dataset_node_predictions(original_name, node_rank, task_group)
    
    def _merge_single_dataset_node_predictions(self, original_name, node_rank, task_group):
        """合并单个数据集在当前节点的所有 GPU 任务结果."""
        logger = get_logger()
        logger.info(f'合并数据集 {original_name} 在节点 {node_rank} 的结果...')
        
        # 收集所有 GPU 任务的预测文件
        gpu_predictions = []
        work_dir = None
        model_abbr = None
        
        for task_obj, dataset_cfg in task_group:
            if work_dir is None:
                work_dir = task_obj.cfg.work_dir
                model_abbr = model_abbr_from_cfg(task_obj.model_cfgs[0])
            
            # 预测文件路径：{work_dir}/predictions/{model_abbr}/{dataset_abbr}.json
            pred_file = f'{work_dir}/predictions/{model_abbr}/{dataset_cfg["abbr"]}.json'
            
            if osp.exists(pred_file):
                try:
                    with open(pred_file, 'r', encoding='utf-8') as f:
                        import json
                        pred_data = json.load(f)
                    
                    # 提取 GPU 编号
                    gpu_idx = 0
                    if '_gpu' in dataset_cfg['abbr']:
                        gpu_part = dataset_cfg['abbr'].split('_gpu')[1]
                        gpu_idx = int(gpu_part) if gpu_part.isdigit() else 0
                    
                    gpu_predictions.append((gpu_idx, pred_data))
                    logger.info(f'加载 GPU {gpu_idx} 预测文件: {pred_file} (数量: {len(pred_data)})')
                    
                except Exception as e:
                    logger.error(f'加载预测文件 {pred_file} 失败: {e}')
            else:
                logger.warning(f'预测文件不存在: {pred_file}')
        
        if not gpu_predictions:
            logger.warning(f'没有找到数据集 {original_name} 在节点 {node_rank} 的任何预测文件')
            return
        
        # 按 GPU 编号排序并合并预测结果
        gpu_predictions.sort(key=lambda x: x[0])
        merged_predictions = {}
        current_idx = 0
        
        for gpu_idx, pred_data in gpu_predictions:
            logger.info(f'合并 GPU {gpu_idx} 的 {len(pred_data)} 条预测')
            for key, value in pred_data.items():
                merged_predictions[str(current_idx)] = value
                current_idx += 1
        
        # 保存合并后的结果为 {original_name}_node{node_rank}.json
        merged_filename = f'{work_dir}/predictions/{model_abbr}/{original_name}_node{node_rank}.json'
        mmengine.mkdir_or_exist(osp.dirname(merged_filename))
        
        try:
            with open(merged_filename, 'w', encoding='utf-8') as f:
                import json
                json.dump(merged_predictions, f, ensure_ascii=False, indent=2)
            
            logger.info(f'成功合并 {len(merged_predictions)} 条预测结果到: {merged_filename}')
            
            # 可选：清理原始 GPU 文件
            # for gpu_idx, _ in gpu_predictions:
            #     gpu_file = f'{work_dir}/predictions/{model_abbr}/{original_name}_node{node_rank}_gpu{gpu_idx}.json'
            #     if osp.exists(gpu_file):
            #         os.remove(gpu_file)
            
        except Exception as e:
            logger.error(f'保存合并结果失败: {e}')
