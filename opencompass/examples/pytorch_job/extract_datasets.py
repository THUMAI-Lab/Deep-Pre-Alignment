#!/usr/bin/env python3
"""
从 OpenCompass 配置文件中提取数据集列表的辅助脚本
"""

import sys
import os
import importlib.util
from pathlib import Path

def get_predefined_datasets(dataset_type="hf_infer"):
    """
    获取预定义的数据集列表
    
    Args:
        dataset_type: 数据集类型 ("hf_infer", "base", "all")
        
    Returns:
        list: 数据集名称列表
    """
    # 基于 base_core_all_v3_ultrafineweb_exp.py 的数据集映射
    hf_infer_datasets = [
        "mmlu"  # 大内存数据集，需要HF推理
    ]
    
    base_datasets = [
        "commonsenseqa", "hellaswag", "ARC-e", "ARC-c", 
        "piqa", "siqa", "winogrande", "obqa"
    ]
    
    if dataset_type == "hf_infer":
        return hf_infer_datasets
    elif dataset_type == "base":
        return base_datasets
    elif dataset_type == "all":
        return hf_infer_datasets + base_datasets
    else:
        return []

def extract_datasets_from_config(config_path, dataset_var_name="hf_infer_datasets_total"):
    """
    从配置文件中提取指定的数据集变量
    
    Args:
        config_path: 配置文件路径
        dataset_var_name: 要提取的数据集变量名
        
    Returns:
        list: 数据集名称列表
    """
    try:
        # 首先尝试使用预定义列表
        if dataset_var_name == "hf_infer_datasets_total":
            return get_predefined_datasets("hf_infer")
        elif dataset_var_name == "base_datasets_total":
            return get_predefined_datasets("base")
        
        # 如果不是标准变量名，尝试动态加载
        print(f"🔄 尝试动态加载配置文件...", file=sys.stderr)
        
        # 添加 opencompass 到 Python 路径
        opencompass_root = Path(__file__).parent
        sys.path.insert(0, str(opencompass_root))
        
        # 切换到opencompass根目录以支持相对导入
        old_cwd = os.getcwd()
        os.chdir(opencompass_root)
        
        try:
            # 动态导入配置文件
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            if spec is None:
                raise ImportError(f"无法加载配置文件: {config_path}")
                
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
        
            # 获取数据集列表
            if hasattr(config_module, dataset_var_name):
                datasets = getattr(config_module, dataset_var_name)
                dataset_names = []
                
                for dataset_cfg in datasets:
                    if isinstance(dataset_cfg, dict):
                        # 尝试获取数据集名称的不同字段
                        name = dataset_cfg.get('abbr') or dataset_cfg.get('name') or dataset_cfg.get('type', 'unknown')
                        dataset_names.append(name)
                    else:
                        # 如果不是字典，尝试获取其字符串表示
                        dataset_names.append(str(dataset_cfg))
                
                return dataset_names
            else:
                print(f"❌ 配置文件中未找到变量: {dataset_var_name}", file=sys.stderr)
                return get_predefined_datasets("hf_infer")  # 回退到预定义列表
                
        except Exception as e:
            print(f"❌ 动态加载失败: {e}", file=sys.stderr)
            print(f"🔄 使用预定义数据集列表", file=sys.stderr)
            return get_predefined_datasets("hf_infer")  # 回退到预定义列表
        finally:
            os.chdir(old_cwd)
            
    except Exception as e:
        print(f"❌ 提取数据集失败: {e}", file=sys.stderr)
        return get_predefined_datasets("hf_infer")  # 回退到预定义列表

def main():
    if len(sys.argv) < 2:
        print("用法: python extract_datasets.py <config_file> [dataset_var_name]")
        print("示例: python extract_datasets.py opencompass/configs/mb_internal/full_config_template/base_minicpm_core_all_ultrafineweb_exp.py hf_infer_datasets_total")
        sys.exit(1)
    
    config_file = sys.argv[1]
    dataset_var = sys.argv[2] if len(sys.argv) > 2 else "hf_infer_datasets_total"
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"🔍 从配置文件提取数据集: {config_file}", file=sys.stderr)
    print(f"🎯 目标变量: {dataset_var}", file=sys.stderr)
    
    dataset_names = extract_datasets_from_config(config_file, dataset_var)
    
    if dataset_names:
        # 输出逗号分隔的数据集列表到标准输出
        datasets_str = ",".join(dataset_names)
        print(datasets_str)
        print(f"✅ 成功提取 {len(dataset_names)} 个数据集", file=sys.stderr)
    else:
        print("❌ 未找到任何数据集", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
