import os
import csv
import json
from pathlib import Path
from abc import ABC, abstractmethod

def has_header(first_row):
    """判断是否包含表头 如果首行去除空格后为A,B,C,D则认为是表头."""
    if not first_row or len(first_row) < 4:
        return False
    
    # 检查前4列是否为A,B,C,D
    if 'A,B,C,D' in ','.join(first_row).replace(' ', '').upper():
        return True
    return False

def get_dataset_name(path):
    """从路径中提取数据集名称
    例如: data/mmlu -> mmlu
         data/ceval/formal_ceval -> ceval
    """
    parts = Path(path).parts
    for part in parts:
        if part in {'mmlu', 'ceval', 'gaokao'}:  # 可以添加更多数据集名称
            return part
    return parts[-1]  # 如果没找到匹配的,返回最后一个部分

class BaseProcessor(ABC):
    """基础处理器类."""
    @staticmethod
    @abstractmethod
    def process_row(row, csv_name='', header=None):
        """处理单行数据."""
        pass
    
    @classmethod
    def process_file(cls, csv_path):
        """处理单个CSV文件."""
        data = []
        csv_name = csv_path.stem  # 获取文件名(不含扩展名)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = list(csv.reader(f))
            if not reader:
                return []
                
            start_idx = 0
            header = None
            # 检查是否有表头
            if has_header(reader[0]) or cls.has_column_header(reader[0]):
                header = reader[0]
                start_idx = 1
                
            # 处理每一行数据
            for row in reader[start_idx:]:
                item = cls.process_row(row, csv_name, header)
                if item:
                    data.append(item)
                    
        return data
    
    @staticmethod
    def has_column_header(row):
        """检查是否有列名表头."""
        return False
    
    @classmethod
    def process_directory(cls, base_dir, output_dir):
        """处理整个目录,包括子目录的处理和输出."""
        # 获取所有子目录及其文件
        subdir_files = {}
        for item in Path(base_dir).rglob('*.csv'):
            subdir = item.parent.name
            if subdir not in subdir_files:
                subdir_files[subdir] = []
            subdir_files[subdir].append(item)
        
        # 获取数据集名称
        dataset_name = get_dataset_name(base_dir)
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 按子目录处理文件
        for subdir in sorted(subdir_files.keys()):
            all_data = []
            # 对每个子目录中的文件排序后处理
            for csv_file in sorted(subdir_files[subdir]):
                try:
                    data = cls.process_file(csv_file)
                    # print(f"处理文件 {csv_file.name} 得到 {len(data)} 条数据")  # 添加调试信息
                    all_data.extend(data)
                    print(f'已处理文件: {csv_file.relative_to(base_dir)}')
                except Exception as e:
                    print(f'处理文件 {csv_file} 时出错: {e}')
            
            # 写入 JSONL 文件
            print(f'子目录 {subdir} 收集到 {len(all_data)} 条数据')  # 添加调试信息
            if len(all_data) > 0:  # 修改判断条件
                # 确保每个子目录的数据都写入对应的jsonl文件
                output_file = dataset_output_dir / f'{subdir}.jsonl'
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in all_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f'已生成 {output_file} 文件,包含 {len(all_data)} 条记录')

class MMLUProcessor(BaseProcessor):
    """MMLU格式处理器."""
    @staticmethod
    def process_row(row, csv_name='', header=None):
        """处理单行数据."""
        if len(row) < 6:
            return None
            
        question = f'{row[0]}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}'
        if len(row) > 4:
            question += f'\nD. {row[4]}'
        answer = f'Answer: {row[-1]}'
        
        return {
            'question': question,
            'answer': answer,
            'source': csv_name
        }

class CEvalProcessor(BaseProcessor):
    """CEval格式处理器."""
    @staticmethod
    def has_column_header(row):
        return has_header(row)
    
    @staticmethod
    def process_row(row, csv_name='', header=None):
        """处理单行数据."""
        if not header or len(row) < len(header):
            return None
            
        # 创建列名到索引的映射
        col_map = {col.strip().lower(): idx for idx, col in enumerate(header)}
        
        # 获取必要的字段
        try:
            question = row[col_map['question']]
            option_a = row[col_map['a']]
            option_b = row[col_map['b']]
            option_c = row[col_map['c']]
            option_d = row[col_map['d']]
            
            question = f'{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}'
            
            # 如果有answer列,添加答案
            if 'answer' in col_map:
                answer = row[col_map['answer']]
                answer = f'Answer: {answer}'
            else:
                answer = ''
            
            result = {
                'question': question,
                'answer': answer,
                'source': csv_name
            }
            
            
            # 如果有解释字段,添加到结果中
            if 'explanation' in col_map and row[col_map['explanation']].strip():
                result['explanation'] = row[col_map['explanation']]
                
            return result
            
        except (KeyError, IndexError):
            return None

def main():
    # 定义需要处理的数据集路径和对应的处理器
    datasets = [
        ['data/mmlu', MMLUProcessor],
        ['data/ceval/formal_ceval', CEvalProcessor],
        # 可以添加更多数据集和对应的处理器
    ]
    
    # 创建输出目录
    output_dir = Path('data/mb_internal/benbench')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个数据集
    for dataset_info in datasets:
        dataset_path = dataset_info[0]
        processor_class = dataset_info[1]
        
        base_dir = Path(dataset_path)
        if not base_dir.exists():
            print(f'警告: 数据集路径 {dataset_path} 不存在,跳过处理')
            continue
            
        print(f'正在处理数据集: {base_dir.name}, 处理器: {processor_class.__name__}')
        
        # 使用处理器的目录处理方法
        processor_class.process_directory(base_dir, output_dir)
        print(f'数据集 {base_dir.name} 处理完成\n')

if __name__ == '__main__':
    main()