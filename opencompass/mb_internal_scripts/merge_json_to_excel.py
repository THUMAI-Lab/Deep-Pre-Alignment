#!/usr/bin/env python3
"""
JSON文件合并并转换为Excel的脚本

该脚本可以：
1. 扫描指定目录中的JSON文件
2. 按文件名前缀分组，合并有数字后缀(_0, _1, _2等)的同名文件
3. 将JSON数据转换为Excel格式，包含prompt和模型名两列
4. 支持可选的参考文件，在现有Excel基础上添加新列

使用示例:
# 简化用法 - 只需指定到实验目录，脚本会自动找到predictions和模型目录
python merge_json_to_excel.py --input_dir "outputs/MiniCPM4-8B/20250607_191108-vllm_general_t0.6_sft-mbgaokao2025_gen_20250506"

# 传统用法 - 直接指定到JSON文件目录
python merge_json_to_excel.py --input_dir "outputs/MiniCPM4-8B/20250607_191108-vllm_general_t0.6_sft-mbgaokao2025_gen_20250506/predictions/MiniCPM4-8B-general-t0.6-sft-vllm"

# 使用参考文件
python merge_json_to_excel.py --input_dir "path/to/experiment/dir" --reference_file "data/mb_internal/mb_arena/0819体感纯文本题库.xlsx"
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


def find_predictions_and_model_dir(input_dir):
    """自动查找predictions目录和模型目录，返回(json_files_dir, model_name)"""
    input_path = Path(input_dir)
    
    # 情况1: 输入目录就是JSON文件所在目录
    json_files = list(input_path.glob("*.json"))
    if json_files:
        model_name = extract_model_name_from_path(str(input_path))
        return str(input_path), model_name
    
    # 情况2: 自动查找predictions目录
    predictions_dir = input_path / "predictions"
    if predictions_dir.exists():
        # 在predictions目录下查找子目录
        model_dirs = [d for d in predictions_dir.iterdir() if d.is_dir()]
        
        if len(model_dirs) == 1:
            # 只有一个子目录，使用它
            model_dir = model_dirs[0]
            model_name = extract_model_name_from_dir(model_dir.name)
            return str(model_dir), model_name
        elif len(model_dirs) > 1:
            # 多个子目录，选择包含JSON文件最多的那个
            best_dir = None
            max_json_count = 0
            
            for model_dir in model_dirs:
                json_count = len(list(model_dir.glob("*.json")))
                if json_count > max_json_count:
                    max_json_count = json_count
                    best_dir = model_dir
            
            if best_dir and max_json_count > 0:
                model_name = extract_model_name_from_dir(best_dir.name)
                return str(best_dir), model_name
    
    # 情况3: 在当前目录下递归查找包含JSON文件的目录
    for subdir in input_path.rglob("*"):
        if subdir.is_dir():
            json_files = list(subdir.glob("*.json"))
            if json_files:
                model_name = extract_model_name_from_dir(subdir.name)
                return str(subdir), model_name
    
    # 都找不到，返回原目录
    model_name = extract_model_name_from_path(str(input_path))
    return str(input_path), model_name


def extract_model_name_from_dir(dir_name):
    """从目录名中提取模型名称"""
    # 尝试从目录名中提取模型名
    if 'MiniCPM' in dir_name:
        return dir_name.split('-')[0] if '-' in dir_name else dir_name
    elif dir_name.startswith(('gpt', 'claude', 'llama', 'qwen', 'baichuan')):
        return dir_name.split('-')[0] if '-' in dir_name else dir_name
    else:
        # 使用完整的目录名作为模型名，但去掉一些常见后缀
        cleaned_name = dir_name
        # 移除常见的后缀
        suffixes_to_remove = ['-general', '-t0.6', '-sft', '-vllm', '-inference']
        for suffix in suffixes_to_remove:
            if cleaned_name.endswith(suffix):
                cleaned_name = cleaned_name[:-len(suffix)]
        return cleaned_name


def extract_model_name_from_path(input_dir):
    """从路径中提取模型名称（备用方法）"""
    path_parts = input_dir.split('/')
    for part in path_parts:
        if 'MiniCPM' in part or part.startswith(('gpt', 'claude', 'llama')):
            return part
    # 如果找不到明显的模型名，使用倒数第二个目录名
    if len(path_parts) >= 2:
        return path_parts[-2].split('-')[0]
    return "Model"


def natural_sort_key(text):
    """自然排序键，正确处理数字排序(0,1,2,3,4而不是0,1,10,11,12,2,3,4)"""
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', text)]


def scan_and_group_files(input_dir):
    """扫描目录并按文件名前缀分组"""
    json_files = []
    input_path = Path(input_dir)
    
    # 查找所有JSON文件
    for json_file in input_path.glob("*.json"):
        json_files.append(json_file)
    
    # 按文件名前缀分组
    groups = defaultdict(list)
    for json_file in json_files:
        filename = json_file.stem
        
        # 检查是否有数字后缀
        match = re.match(r'^(.+)_(\d+)$', filename)
        if match:
            prefix, number = match.groups()
            groups[prefix].append((int(number), json_file))
        else:
            # 没有数字后缀的文件
            groups[filename].append((0, json_file))
    
    # 对每个组内的文件进行排序
    for prefix in groups:
        groups[prefix].sort(key=lambda x: x[0])
    
    return dict(groups)


def clean_excel_text(text):
    """清理文本中不兼容Excel的字符"""
    if not isinstance(text, str):
        return text
    
    # 移除或替换可能导致Excel错误的字符
    # 这些字符在Excel工作表中不被支持
    problematic_chars = [
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', 
        '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12',
        '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
        '\x1b', '\x1c', '\x1d', '\x1e', '\x1f'
    ]
    
    for char in problematic_chars:
        text = text.replace(char, '')
    
    # 限制文本长度，Excel单元格最大支持32,767个字符
    # if len(text) > 32000:
    #     text = text[:32000] + "..."
    
    return text


def extract_data_from_json(json_file):
    """从JSON文件中提取数据"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        extracted_data = []
        
        # 处理不同的JSON结构
        if isinstance(data, dict):
            for key, item in data.items():
                if isinstance(item, dict) and 'origin_prompt' in item and 'prediction' in item:
                    # 提取prompt文本
                    prompt = ""
                    if isinstance(item['origin_prompt'], list):
                        for prompt_item in item['origin_prompt']:
                            if isinstance(prompt_item, dict) and 'prompt' in prompt_item:
                                prompt += prompt_item['prompt']
                    elif isinstance(item['origin_prompt'], str):
                        prompt = item['origin_prompt']
                    
                    # 清理文本
                    cleaned_prompt = clean_excel_text(prompt.strip())
                    cleaned_prediction = clean_excel_text(item.get('prediction', '').strip())
                    
                    extracted_data.append({
                        'prompt': cleaned_prompt,
                        'prediction': cleaned_prediction
                    })
        
        return extracted_data
    
    except Exception as e:
        print(f"读取文件 {json_file} 时出错: {e}")
        return []


def merge_json_files_to_excel(grouped_files, output_dir, model_name, reference_file=None):
    """将分组的JSON文件合并为Excel文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for prefix, file_list in grouped_files.items():
        print(f"处理文件组: {prefix}")
        
        all_data = []
        
        # 按顺序合并所有文件的数据
        for _, json_file in file_list:
            file_data = extract_data_from_json(json_file)
            all_data.extend(file_data)
        
        if not all_data:
            print(f"  警告: 文件组 {prefix} 中没有找到有效数据")
            continue
        
        # 创建DataFrame
        df = pd.DataFrame(all_data)
        df = df.rename(columns={'prediction': model_name})
        
        # 处理参考文件
        if reference_file and os.path.exists(reference_file):
            try:
                print(f"  使用参考文件: {reference_file}")
                ref_df = pd.read_excel(reference_file)
                
                # 合并参考文件和新数据
                # 如果参考文件的行数与新数据不同，以较短的为准
                min_rows = min(len(ref_df), len(df))
                ref_df = ref_df.iloc[:min_rows].copy()
                df = df.iloc[:min_rows].copy()
                
                # 将新的prompt和模型列添加到参考文件
                ref_df['prompt'] = df['prompt']
                ref_df[model_name] = df[model_name]
                final_df = ref_df
                
            except Exception as e:
                print(f"  处理参考文件时出错: {e}，将只使用新数据")
                final_df = df
        else:
            final_df = df
        
        # 生成输出文件名
        output_filename = f"{model_name}_{prefix}.xlsx"
        output_file = output_path / output_filename
        
        # 保存Excel文件
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            print(f"  成功生成: {output_file}")
            print(f"  数据行数: {len(final_df)}")
            
        except Exception as e:
            print(f"  保存文件 {output_file} 时出错: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="合并JSON文件并转换为Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 简化用法（推荐） - 只需指定到实验目录，脚本会自动找到predictions和模型目录
  python merge_json_to_excel.py --input_dir "outputs/MiniCPM4-8B/20250607_191108-vllm_general_t0.6_sft-mbgaokao2025_gen_20250506"
  
  # 传统用法 - 直接指定到JSON文件目录
  python merge_json_to_excel.py --input_dir "outputs/.../predictions/model-name"
  
  # 使用参考文件
  python merge_json_to_excel.py --input_dir "path/to/experiment" --reference_file "data/mb_internal/mb_arena/0819体感纯文本题库.xlsx"
  
  # 指定输出目录和模型名
  python merge_json_to_excel.py --input_dir "path/to/experiment" --output_dir "custom/output" --model_name "CustomModel"
        """)
    
    parser.add_argument('--input_dir', required=True, 
                        help='输入目录路径（可以是实验目录或直接的JSON文件目录，脚本会自动查找）')
    parser.add_argument('--output_dir', 
                        help='Excel文件输出目录（默认为输入目录）')
    parser.add_argument('--model_name', 
                        help='模型名称（用于Excel列名，默认从目录名自动提取）')
    parser.add_argument('--reference_file', 
                        help='可选的参考Excel文件路径，将在现有文件基础上添加prompt和模型列')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return 1
    
    # 智能查找JSON文件目录和模型名
    json_files_dir, auto_model_name = find_predictions_and_model_dir(args.input_dir)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 默认输出到原始输入目录（用户提供的目录）
        output_dir = args.input_dir
    
    # 设置模型名（用户指定的优先，否则使用自动检测的）
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = auto_model_name
    
    print(f"输入目录: {args.input_dir}")
    print(f"JSON文件目录: {json_files_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型名称: {model_name}")
    if args.reference_file:
        print(f"参考文件: {args.reference_file}")
    print("-" * 50)
    
    # 扫描和分组文件
    grouped_files = scan_and_group_files(json_files_dir)
    
    if not grouped_files:
        print("错误: 在指定目录中未找到JSON文件")
        return 1
    
    print(f"找到 {len(grouped_files)} 个文件组:")
    for prefix, file_list in grouped_files.items():
        file_names = [f[1].name for f in file_list]
        print(f"  {prefix}: {file_names}")
    print("-" * 50)
    
    # 合并文件并生成Excel
    merge_json_files_to_excel(grouped_files, output_dir, model_name, args.reference_file)
    
    print("处理完成!")
    return 0


if __name__ == "__main__":
    exit(main())
