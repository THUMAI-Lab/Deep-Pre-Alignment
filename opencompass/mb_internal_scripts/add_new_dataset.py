import os
import sys
import argparse
from pathlib import Path


def to_camel_case(snake_str):
    """将下划线命名转换为大驼峰命名"""
    return ''.join(word.capitalize() for word in snake_str.split('_'))


def create_dataset_file(dataset_name, datasets_dir, force=False):
    """创建数据集文件"""
    dataset_file_path = os.path.join(datasets_dir, f'{dataset_name}.py')
    
    # 安全检查：如果文件已存在，询问是否覆盖
    if os.path.exists(dataset_file_path) and not force:
        try:
            response = input(f"文件 {dataset_file_path} 已存在。是否覆盖？(y/N): ")
            if response.lower() != 'y':
                print(f"跳过创建 {dataset_file_path}")
                return False
        except EOFError:
            print(f"跳过创建 {dataset_file_path}")
            return False
    
    dataset_registry = f"""import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class {to_camel_case(dataset_name)}Dataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path, local_mode=True)
        filename = osp.join(path, f'{{name}}.jsonl')
        dataset = []
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)
        return Dataset.from_list(dataset)
"""
    
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        f.write(dataset_registry)
    
    print(f"已创建数据集文件: {dataset_file_path}")
    return True


def create_config_file(dataset_name, config_dir, force=False):
    """创建配置文件"""
    # 创建数据集文件夹
    dataset_config_dir = os.path.join(config_dir, dataset_name)
    os.makedirs(dataset_config_dir, exist_ok=True)
    
    config_file_path = os.path.join(dataset_config_dir, f'{dataset_name}_gen_mb.py')
    
    # 安全检查：如果文件已存在，询问是否覆盖
    if os.path.exists(config_file_path) and not force:
        try:
            response = input(f"文件 {config_file_path} 已存在。是否覆盖？(y/N): ")
            if response.lower() != 'y':
                print(f"跳过创建 {config_file_path}")
                return False
        except EOFError:
            print(f"跳过创建 {config_file_path}")
            return False
    
    config_template = f"""from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithFullDetailsEvaluator
from opencompass.datasets import {to_camel_case(dataset_name)}Dataset
import os

QUERY_TEMPLATE = '''    
你回答的最后一行**必须**是以下格式 '答案：$选项' (不带引号), 其中选项是ABCD之一。请在回答之前一步步思考。

{{question}}

A) {{A}}
B) {{B}}
C) {{C}}
D) {{D}}
'''.strip()

{dataset_name}_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='answer'
)

{dataset_name}_sets = ['']
{dataset_name}_datasets = []

for name in {dataset_name}_sets:
    {dataset_name}_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    {dataset_name}_eval_cfg = dict(
        evaluator=dict(type=AccwithFullDetailsEvaluator),
    )

    {dataset_name}_datasets.append(
        dict(
            abbr=f'{dataset_name}_{{name}}',
            type={to_camel_case(dataset_name)}Dataset,
            # path='opencompass/{dataset_name}',
            path=os.path.join(os.environ.get('COMPASS_DATA_CACHE', './'), 'data/mb_internal/{dataset_name}'),
            name=name,
            reader_cfg={dataset_name}_reader_cfg,
            infer_cfg={dataset_name}_infer_cfg,
            eval_cfg={dataset_name}_eval_cfg,
        ))
"""
    
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(config_template)
    
    print(f"已创建配置文件: {config_file_path}")
    return True


def validate_dataset_name(name):
    """验证数据集名称是否符合Python变量命名规范"""
    if not name.isidentifier():
        return False
    if name.startswith('_'):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='快速添加新数据集')
    parser.add_argument('dataset_name', nargs='?', help='数据集名称（使用下划线命名）')
    parser.add_argument('--datasets-dir', default='opencompass/datasets', help='数据集目录路径')
    parser.add_argument('--config-dir', default='opencompass/configs/mb_internal/datasets', help='配置文件目录路径')
    parser.add_argument('--force', action='store_true', help='强制覆盖现有文件')
    
    args = parser.parse_args()
    
    # 如果没有提供数据集名称，则交互式输入
    if not args.dataset_name:
        try:
            dataset_name = input("请输入数据集名称（使用下划线命名，如 my_dataset）: ").strip()
        except EOFError:
            print("错误：需要提供数据集名称")
            sys.exit(1)
    else:
        dataset_name = args.dataset_name
    
    # 验证数据集名称
    if not validate_dataset_name(dataset_name):
        print("错误：数据集名称必须是有效的Python标识符，且不能以下划线开头")
        sys.exit(1)
    
    # 检查目录是否存在
    datasets_dir = Path(args.datasets_dir)
    config_dir = Path(args.config_dir)
    
    if not datasets_dir.exists():
        print(f"错误：数据集目录 {datasets_dir} 不存在")
        sys.exit(1)
    
    if not config_dir.exists():
        print(f"错误：配置目录 {config_dir} 不存在")
        sys.exit(1)
    
    print(f"正在为数据集 '{dataset_name}' 创建文件...")
    print(f"数据集类名将使用大驼峰命名: {to_camel_case(dataset_name)}Dataset")
    
    if args.force:
        print("⚠️  强制覆盖模式已启用")
    
    # 创建数据集文件
    dataset_created = create_dataset_file(dataset_name, datasets_dir, args.force)
    
    # 创建配置文件
    config_created = create_config_file(dataset_name, config_dir, args.force)
    
    if dataset_created and config_created:
        print(f"\n✅ 成功创建数据集 '{dataset_name}' 的所有文件！")
        print(f"📁 数据集文件: {datasets_dir / f'{dataset_name}.py'}")
        print(f"📁 配置文件: {config_dir / dataset_name / f'{dataset_name}_gen_mb.py'}")
        print(f"\n下一步：")
        print(f"1. 修改数据集文件中的数据加载逻辑")
        print(f"2. 根据需要调整配置文件中的模板和参数")
        print(f"3. 在 opencompass/datasets/__init__.py 中添加导入")
    else:
        print(f"⚠️  部分文件创建失败或被跳过")
    
    # # 清理测试文件
    # if dataset_name == 'test_dataset':
    #     print("\n🧹 清理测试文件...")
    #     test_dataset_file = datasets_dir / 'test_dataset.py'
    #     test_config_dir = config_dir / 'test_dataset'
        
    #     try:
    #         if test_dataset_file.exists():
    #             test_dataset_file.unlink()
    #             print(f"已删除: {test_dataset_file}")
            
    #         if test_config_dir.exists():
    #             import shutil
    #             shutil.rmtree(test_config_dir)
    #             print(f"已删除: {test_config_dir}")
    #     except Exception as e:
    #         print(f"清理失败: {e}")


if __name__ == "__main__":
    main()

