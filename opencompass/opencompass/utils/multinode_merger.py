import json
import os
import os.path as osp
from typing import Dict, List, Optional

import mmengine

from opencompass.utils import get_logger, get_infer_output_path, model_abbr_from_cfg, dataset_abbr_from_cfg


class MultiNodePredictionMerger:
    """多节点预测结果合并器.
    
    收集各个节点的预测结果，合并成完整的预测文件，
    然后进行统一评估。
    """
    
    def __init__(self, work_dir: str, world_size: int):
        self.work_dir = work_dir
        self.world_size = world_size
        self.logger = get_logger()
        
    def merge_predictions(self, model_cfg: Dict, dataset_cfg: Dict, 
                         output_dir: str) -> Optional[str]:
        """合并多个节点的预测结果.
        
        Args:
            model_cfg: 模型配置
            dataset_cfg: 数据集配置  
            output_dir: 输出目录
            
        Returns:
            合并后的预测文件路径，如果合并失败返回 None
        """
        model_abbr = model_abbr_from_cfg(model_cfg)
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
        
        # 移除可能已有的节点后缀，获取原始数据集名称
        original_dataset_abbr = dataset_abbr.rsplit('_node', 1)[0] if '_node' in dataset_abbr else dataset_abbr
        
        self.logger.info(f'开始合并预测结果: {model_abbr}/{original_dataset_abbr}')
        
        # 收集各节点的预测文件
        node_predictions = []
        missing_nodes = []
        
        # 跨时间戳目录搜索预测文件
        import glob
        
        for node_rank in range(self.world_size):
            # 构造节点特定的数据集配置
            node_dataset_cfg = dataset_cfg.copy()
            node_dataset_cfg['abbr'] = f'{original_dataset_abbr}_node{node_rank}'
            
            found_prediction = False
            
            # 在所有可能的时间戳目录中搜索
            search_patterns = [
                # 当前工作目录
                osp.join(self.work_dir, 'predictions'),
                # 父目录下的所有时间戳目录
                osp.join(osp.dirname(self.work_dir), '*/predictions'),
                # 全局搜索模式
                './outputs/*/predictions'
            ]
            
            for search_pattern in search_patterns:
                if found_prediction:
                    break
                    
                if '*' in search_pattern:
                    # 使用glob搜索
                    pred_dirs = glob.glob(search_pattern)
                else:
                    pred_dirs = [search_pattern] if osp.exists(search_pattern) else []
                
                for pred_dir in pred_dirs:
                    node_pred_path = get_infer_output_path(
                        model_cfg, node_dataset_cfg, pred_dir
                    )
                    
                    if osp.exists(node_pred_path):
                        self.logger.info(f'找到节点 {node_rank} 的预测文件: {node_pred_path}')
                        try:
                            with open(node_pred_path, 'r', encoding='utf-8') as f:
                                node_pred = json.load(f)
                            node_predictions.append((node_rank, node_pred))
                            found_prediction = True
                            break
                        except Exception as e:
                            self.logger.error(f'读取节点 {node_rank} 预测文件失败: {e}')
                            continue
            
            if not found_prediction:
                self.logger.warning(f'节点 {node_rank} 的预测文件未找到')
                missing_nodes.append(node_rank)
        
        if missing_nodes:
            self.logger.error(f'缺少节点 {missing_nodes} 的预测结果，无法合并')
            return None
            
        if not node_predictions:
            self.logger.error('没有找到任何节点的预测结果')
            return None
        
        # 合并预测结果
        merged_predictions = {}
        current_idx = 0
        
        # 按节点顺序合并预测
        for node_rank, node_pred in sorted(node_predictions):
            self.logger.info(f'合并节点 {node_rank} 的 {len(node_pred)} 条预测')
            for key, value in node_pred.items():
                # 重新编号键值，确保连续性
                merged_predictions[str(current_idx)] = value
                current_idx += 1
        
        # 保存合并后的预测结果，使用原始数据集名称
        merged_dataset_cfg = dataset_cfg.copy()
        merged_dataset_cfg['abbr'] = original_dataset_abbr
        merged_output_path = get_infer_output_path(
            model_cfg, merged_dataset_cfg, 
            osp.join(output_dir, 'predictions')
        )
        
        # 确保输出目录存在
        mmengine.mkdir_or_exist(osp.dirname(merged_output_path))
        
        try:
            with open(merged_output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_predictions, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f'成功合并 {len(merged_predictions)} 条预测结果到: {merged_output_path}')
            return merged_output_path
            
        except Exception as e:
            self.logger.error(f'保存合并预测结果失败: {e}')
            return None
    
    def merge_all_predictions(self, model_dataset_combinations: List[Dict], 
                            output_dir: str) -> List[str]:
        """合并所有模型-数据集组合的预测结果.
        
        Args:
            model_dataset_combinations: 模型-数据集组合列表
            output_dir: 输出目录
            
        Returns:
            成功合并的预测文件路径列表
        """
        merged_files = []
        
        for comb in model_dataset_combinations:
            for model_cfg in comb['models']:
                for dataset_cfg in comb['datasets']:
                    merged_path = self.merge_predictions(
                        model_cfg, dataset_cfg, output_dir
                    )
                    if merged_path:
                        merged_files.append(merged_path)
        
        self.logger.info(f'总共合并了 {len(merged_files)} 个预测文件')
        return merged_files
    
    def cleanup_node_predictions(self, model_dataset_combinations: List[Dict]):
        """清理各节点的临时预测文件.
        
        Args:
            model_dataset_combinations: 模型-数据集组合列表
        """
        for comb in model_dataset_combinations:
            for model_cfg in comb['models']:
                for dataset_cfg in comb['datasets']:
                    dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
                    
                    for node_rank in range(self.world_size):
                        node_dataset_cfg = dataset_cfg.copy()
                        node_dataset_cfg['abbr'] = f'{dataset_abbr}_node{node_rank}'
                        
                        node_pred_path = get_infer_output_path(
                            model_cfg, node_dataset_cfg,
                            osp.join(self.work_dir, 'predictions')
                        )
                        
                        if osp.exists(node_pred_path):
                            try:
                                os.remove(node_pred_path)
                                self.logger.info(f'清理节点 {node_rank} 预测文件: {node_pred_path}')
                            except Exception as e:
                                self.logger.warning(f'清理节点 {node_rank} 预测文件失败: {e}')


def merge_multinode_predictions(work_dir: str, model_dataset_combinations: List[Dict], 
                               world_size: int, output_dir: str = None,
                               cleanup: bool = True) -> List[str]:
    """合并多节点预测结果的便捷函数.
    
    Args:
        work_dir: 工作目录
        model_dataset_combinations: 模型-数据集组合列表
        world_size: 节点数量
        output_dir: 输出目录，默认为 work_dir
        cleanup: 是否清理节点临时文件
        
    Returns:
        成功合并的预测文件路径列表
    """
    if output_dir is None:
        output_dir = work_dir
    
    merger = MultiNodePredictionMerger(work_dir, world_size)
    merged_files = merger.merge_all_predictions(model_dataset_combinations, output_dir)
    
    if cleanup:
        merger.cleanup_node_predictions(model_dataset_combinations)
    
    return merged_files

