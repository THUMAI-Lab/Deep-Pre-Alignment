import copy
import math
import os
import os.path as osp
from typing import Dict, List, Optional

import mmengine
from mmengine.config import Config, ConfigDict

from opencompass.registry import PARTITIONERS
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger)

from .base import BasePartitioner


@PARTITIONERS.register_module()
class MultiNodePartitioner(BasePartitioner):
    """多节点分区器，根据节点数量和排名分发数据集片段.
    
    这个分区器会根据 PyTorchJob 环境变量自动检测节点信息，
    并为每个节点分配对应的数据集片段。
    
    Args:
        out_dir (str): 任务输出目录
        min_task_size (int): 最小任务大小，默认为 1
        dataset_size_path (str): 数据集大小缓存文件路径
        keep_keys (list[str]): 需要保留的配置键列表
    """

    def __init__(self,
                 out_dir: str,
                 min_task_size: int = 1,
                 tasks_per_node: int = None,  # 每个节点的任务数量
                 dataset_size_path: str = '.cache/dataset_size.json',
                 keep_keys: Optional[List[str]] = None):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)
        
        # 从环境变量获取多节点配置
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.rank = int(os.getenv('RANK', '0'))
        self.master_addr = os.getenv('MASTER_ADDR', 'localhost')
        self.master_port = os.getenv('MASTER_PORT', '23456')
        
        self.min_task_size = min_task_size
        # 如果未指定每个节点的任务数，默认使用GPU数量
        if tasks_per_node is None:
            import torch
            try:
                self.tasks_per_node = torch.cuda.device_count() or 1
            except:
                self.tasks_per_node = 1
        else:
            self.tasks_per_node = tasks_per_node
            
        self.dataset_size_path = dataset_size_path
        self._dataset_size = {}
        
        logger = get_logger()
        logger.info(f'MultiNodePartitioner 初始化: '
                   f'world_size={self.world_size}, '
                   f'rank={self.rank}, '
                   f'tasks_per_node={self.tasks_per_node}, '
                   f'master_addr={self.master_addr}, '
                   f'master_port={self.master_port}')

        # 加载数据集大小缓存
        if osp.exists(self.dataset_size_path):
            self._dataset_size = mmengine.load(self.dataset_size_path)

    @property
    def dataset_size(self):
        return self._dataset_size

    def partition(self,
                  model_dataset_combinations: List[Dict[str, List]],
                  work_dir: str,
                  out_dir: str,
                  add_cfg: Dict = {}) -> List[ConfigDict]:
        """将模型-数据集组合分区为多节点任务.
        
        每个节点只会处理分配给它的数据集片段。
        """
        logger = get_logger()
        tasks = []
        
        for comb in model_dataset_combinations:
            for model in comb['models']:
                # 检查是否所有数据集任务都已完成
                all_completed = True
                for dataset in comb['datasets']:
                    filename = get_infer_output_path(model, dataset, out_dir)
                    if not osp.exists(filename):
                        all_completed = False
                        break
                
                if all_completed:
                    logger.info(f'节点 {self.rank}: 跳过已完成的模型 {model["abbr"]}')
                    continue
                
                # 为每个模型按GPU维度合并数据集
                # 首先收集所有数据集的分片信息
                all_dataset_splits = {}  # {dataset_name: [splits]}
                
                for dataset in comb['datasets']:
                    node_dataset_splits = self.split_dataset_for_node_tasks(dataset)
                    dataset_abbr = dataset['abbr']
                    all_dataset_splits[dataset_abbr] = node_dataset_splits
                
                # 按GPU任务索引分组，每个GPU一个任务
                for task_idx in range(self.tasks_per_node):
                    gpu_datasets = []
                    
                    # 收集当前GPU索引对应的所有数据集片段
                    for dataset_abbr, splits in all_dataset_splits.items():
                        if task_idx < len(splits) and splits[task_idx]:  # 如果该GPU有对应的数据片段
                            gpu_datasets.append(splits[task_idx])
                    
                    if gpu_datasets:  # 如果当前GPU有数据集片段
                        dataset_names = [ds["abbr"] for ds in gpu_datasets]
                        logger.info(f'节点 {self.rank} GPU {task_idx}: 为模型 {model["abbr"]} 合并数据集片段 {dataset_names}')
                        
                        task = Config({
                            'models': [model],
                            'datasets': [gpu_datasets],  # 将当前GPU的所有数据集片段合并到一个任务中
                            'work_dir': work_dir,
                            'node_rank': self.rank,
                            'task_idx': task_idx,  # 每个GPU一个任务索引
                            'world_size': self.world_size,
                            **add_cfg
                        })
                        tasks.append(task)
        
        logger.info(f'节点 {self.rank}: 生成了 {len(tasks)} 个任务')
        return tasks

    def split_dataset_for_node_tasks(self, dataset_cfg: ConfigDict) -> List[ConfigDict]:
        """为当前节点创建多个任务的数据集分片."""
        dataset_size = self.get_size(dataset_cfg)
        split_configs = []
        abbr = dataset_abbr_from_cfg(dataset_cfg)
        
        logger = get_logger()
        logger.info(f'分片数据集 {abbr}: 总大小={dataset_size}, 节点数={self.world_size}, 每节点任务数={self.tasks_per_node}')
        
        # 计算总任务数和每个任务的大小
        total_tasks = self.world_size * self.tasks_per_node
        task_size = max(math.ceil(dataset_size / total_tasks), self.min_task_size)
        
        # 计算当前节点的任务范围
        node_start_task = self.rank * self.tasks_per_node
        node_end_task = (self.rank + 1) * self.tasks_per_node
        
        logger.info(f'节点 {self.rank}: 负责任务 {node_start_task} 到 {node_end_task-1}')
        
        # 为当前节点的每个任务创建数据分片
        for local_task_idx in range(self.tasks_per_node):
            global_task_idx = node_start_task + local_task_idx
            
            start_idx = global_task_idx * task_size
            end_idx = min(start_idx + task_size, dataset_size)
            
            if start_idx >= dataset_size:
                # 数据已经分配完，创建空任务
                cfg = copy.deepcopy(dataset_cfg)
                cfg['abbr'] = f'{abbr}_node{self.rank}_gpu{local_task_idx}_empty'
                cfg['reader_cfg']['test_range'] = f'[{dataset_size}:{dataset_size}]'
                split_configs.append(None)  # 空任务
                logger.info(f'节点 {self.rank} 任务 {local_task_idx}: 空数据集片段')
            else:
                cfg = copy.deepcopy(dataset_cfg)
                cfg['abbr'] = f'{abbr}_node{self.rank}_gpu{local_task_idx}'
                test_range = cfg['reader_cfg'].get('test_range', '')
                cfg['reader_cfg']['test_range'] = f'{test_range}[{start_idx}:{end_idx}]'
                split_configs.append(cfg)
                logger.info(f'节点 {self.rank} 任务 {local_task_idx}: 数据集片段 [{start_idx}:{end_idx}] (大小: {end_idx - start_idx})')
        
        return split_configs

    def get_size(self, dataset: ConfigDict) -> int:
        """获取数据集大小."""
        dataset_abbr = dataset_abbr_from_cfg(dataset)
        test_range = dataset.reader_cfg.get('test_range', '')

        if dataset_abbr in self.dataset_size:
            actual_size = eval(f'len(range(self.dataset_size[dataset_abbr]){test_range})')
            return actual_size

        # 构建数据集以获取大小
        dataset_obj = build_dataset_from_cfg(dataset)
        self.dataset_size[dataset_abbr] = len(dataset_obj.test)

        # 保存缓存
        mmengine.mkdir_or_exist('.cache/')
        mmengine.dump(self.dataset_size,
                      self.dataset_size_path,
                      indent=4,
                      ensure_ascii=False)

        actual_size = eval(f'len(range(self.dataset_size[dataset_abbr]){test_range})')
        return actual_size

