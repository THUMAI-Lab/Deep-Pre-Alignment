import argparse
import glob
import os
import os.path as osp
from datetime import datetime, timedelta, timezone
import io

import pandas as pd

# 定义完整的表头列表 - 包含所有可能的字段
FULL_HEADER = [
    'model_name', 'datasets', 'metric', 'mode', 'score', 'version',
    'start_time', 'record_time', '显卡型号', '运行卡数', '耗时分钟', 'JOB_ID', 'work_dir'
]

# 定义默认字段值 - 可以在这里添加更多默认字段
DEFAULT_FIELDS = {
    '显卡型号': 'H100-80G',
    '运行卡数': '8', 
    'JOB_ID': '-',
    'work_dir': '-'
}

# 默认搜索路径
DEFAULT_OUTPUT_PATHS = [
    '/data/user/opencompass_results/outputs',
    '/user/opencompass_results/outputs',
    './outputs',
]

def parse_time(time_str):
    """解析时间字符串,格式为'20250504_133954'."""
    try:
        # 处理 nan 值
        if pd.isna(time_str) or time_str == 'nan' or str(time_str).lower() == 'nan':
            return None
        # 解析时间字符串并设置为北京时区
        dt = datetime.strptime(str(time_str), '%Y%m%d_%H%M%S')
        # 将解析出的时间设置为北京时区(UTC+8)
        return dt.replace(tzinfo=timezone(timedelta(hours=8)))
    except (ValueError, TypeError) as e:
        print(f'时间解析错误 "{time_str}": {e}')
        return None


def normalize_dataframe(df):
    """标准化DataFrame，确保包含所有必需字段，缺失字段用默认值填充."""
    # 创建包含所有字段默认值的字典
    defaults = DEFAULT_FIELDS.copy()
    defaults.update({field: '' for field in FULL_HEADER if field not in defaults})
    
    # 批量添加缺失的列
    for field, default_value in defaults.items():
        if field not in df.columns:
            df[field] = default_value
    
    # 按照标准顺序重新排列列
    return df[FULL_HEADER]


def parse_csv_with_full_header(csv_file):
    """使用完整表头解析CSV文件，自动补全缺失字段."""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 获取原始表头
        lines = content.strip().split('\n')
        if not lines:
            return None
            
        original_header = [field.strip() for field in lines[0].split(',')]
        
        # 找出缺失的字段
        missing_fields = [field for field in FULL_HEADER if field not in original_header]
        
        if missing_fields:
            # 构造新的表头
            new_header = original_header + missing_fields
            
            # 重新构造CSV内容
            new_lines = [','.join(new_header)]
            
            # 处理数据行，为缺失字段添加默认值
            for line in lines[1:]:
                if line.strip():
                    # 原始数据
                    fields = [field.strip() for field in line.split(',')]
                    # 为缺失字段添加默认值
                    for field in missing_fields:
                        default_value = DEFAULT_FIELDS.get(field, '')
                        fields.append(default_value)
                    new_lines.append(','.join(fields))
            
            # 用新内容创建StringIO对象
            csv_content = '\n'.join(new_lines)
            df = pd.read_csv(io.StringIO(csv_content))
        else:
            # 如果没有缺失字段，直接读取
            df = pd.read_csv(csv_file)
        
        # 标准化DataFrame
        if 'record_time' in df.columns:
            df = normalize_dataframe(df)
            return df
        
        return None
        
    except Exception as e:
        print(f'解析失败: {e}')
        return None


def collect_csv_data(output_paths):
    """从指定路径收集所有CSV数据."""
    # 处理输入路径，确保是列表格式
    if isinstance(output_paths, str):
        path_list = [output_paths]
    else:
        path_list = output_paths
    
    all_results = []
    
    # 遍历所有路径
    for output_path in path_list:
        if not os.path.exists(output_path):
            print(f'路径不存在，跳过: {output_path}')
            continue
            
        print(f'正在处理路径: {output_path}')
        
        # 获取当前路径下的所有csv文件
        csv_files = glob.glob(os.path.join(output_path, '*_summary.csv'))
        csv_files.sort()
        
        if not csv_files:
            print(f'  未找到CSV文件: {output_path}')
            continue
            
        print(f'  找到 {len(csv_files)} 个CSV文件')

        # 读取当前路径下的每个csv文件
        for csv_file in csv_files:
            df = None
            try:
                # 首先尝试正常读取
                df = pd.read_csv(csv_file)
                if 'record_time' in df.columns:
                    df = normalize_dataframe(df)
                else:
                    df = None
            except Exception as e:
                print(f'常规读取失败 {csv_file}: {e}，尝试完整表头解析')
                # 如果常规读取失败，使用完整表头解析
                df = parse_csv_with_full_header(csv_file)
                if df is not None:
                    print(f'完整表头解析成功: {csv_file}, 记录数: {len(df)}')
            
            if df is not None:
                all_results.append(df)
    
    return all_results


def filter_recent_results(output_paths='outputs', hours=48, save_path=None):
    """过滤最近hours小时内的评测结果.
    
    Args:
        output_paths: 单个路径字符串或路径列表
        hours: 过滤的小时数
        save_path: 保存路径，如果为None则保存到第一个output_path下
    """
    # 获取当前时间
    now = datetime.now(timezone(timedelta(hours=8)))

    # 处理输入路径，确保是列表格式
    if isinstance(output_paths, str):
        path_list = [output_paths]
    else:
        path_list = output_paths
    
    # 确定保存路径
    if save_path is None:
        save_path = path_list[0]  # 默认保存到第一个路径下

    # 收集所有CSV数据
    all_results = collect_csv_data(output_paths)

    if not all_results:
        print('No results found in any path')
        return

    print(f'总共收集到 {len(all_results)} 个数据文件')

    # 合并并处理数据
    recent_df = process_and_filter_data(all_results, hours, now)
    if recent_df is None:
        return

    # 保存结果
    output_file = os.path.join(save_path, 'recent_results.csv')
    recent_df.to_csv(output_file, index=False)
    print(f'保存了 {len(recent_df)} 条最近结果到 {osp.abspath(output_file)}')


def process_and_filter_data(all_results, hours, now):
    """处理和过滤数据."""
    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f'合并后总记录数: {len(combined_df)}')

    # 转换record_time为datetime
    combined_df['datetime'] = combined_df['record_time'].apply(parse_time)
    
    # 过滤掉解析失败的记录
    valid_records = len(combined_df)
    valid_mask = combined_df['datetime'].notna() & (combined_df['datetime'] != None)
    combined_df = combined_df[valid_mask]
    invalid_records = valid_records - len(combined_df)
    if invalid_records > 0:
        print(f'过滤掉 {invalid_records} 条时间解析失败的记录')
    
    if len(combined_df) == 0:
        print('没有有效的时间记录，无法进行过滤')
        return None

    # 过滤最近指定小时数的结果
    cutoff_time = now - timedelta(hours=hours)
    recent_df = combined_df[combined_df['datetime'] >= cutoff_time]
    print(f'过滤后（最近{hours}小时）记录数: {len(recent_df)}')

    # 去重
    before_dedup = len(recent_df)
    recent_df = recent_df.drop_duplicates()
    after_dedup = len(recent_df)
    if before_dedup != after_dedup:
        print(f'去重：{before_dedup} -> {after_dedup}')

    # 删除datetime列
    recent_df = recent_df.drop('datetime', axis=1)
    
    return recent_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='过滤最近指定小时数内的评测结果')
    # 位置参数：小时数
    parser.add_argument('hours',
                        type=int,
                        nargs='?',
                        default=48,
                        help='需要过滤的小时数范围（默认：48小时）')
    parser.add_argument('--paths',
                        type=str,
                        nargs='+',
                        default=None,
                        help='要处理的路径列表，用空格分隔')
    parser.add_argument('--save-path',
                        type=str,
                        default=None,
                        help='结果保存路径（默认保存到第一个输入路径）')
    args = parser.parse_args()

    # 使用hours参数
    hours = args.hours
    
    # 处理路径参数
    if args.paths:
        output_paths = args.paths
    else:
        output_paths = DEFAULT_OUTPUT_PATHS
    
    filter_recent_results(output_paths, hours, args.save_path)
