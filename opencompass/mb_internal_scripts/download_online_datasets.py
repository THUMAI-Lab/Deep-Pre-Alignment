#!/usr/bin/env python3
"""
自动下载数据集脚本
支持传统数据集(wget+zip)和HuggingFace数据集(git clone)
包括智能更新检测功能
"""

import os
import subprocess
import sys
import argparse
import zipfile
import glob

# 配置
is_update = 0  # 设置为1启用更新检查
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def load_datasets_url():
    """从文件中动态加载DATASETS_URL"""
    try:
        # 尝试从相对路径读取
        datasets_info_path = os.path.join(os.path.dirname(
            __file__), '..', 'opencompass', 'utils', 'datasets_info.py')
        if not os.path.exists(datasets_info_path):
            # 如果相对路径不存在，尝试当前目录的相对路径
            datasets_info_path = 'opencompass/utils/datasets_info.py'

        if not os.path.exists(datasets_info_path):
            print(f"❌ 找不到datasets_info.py文件")
            print(f"🔍 请确保在opencompass项目根目录下运行此脚本")
            return {}

        # 读取文件内容
        with open(datasets_info_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 找到DATASETS_URL的定义
        start_marker = "DATASETS_URL = {"
        end_marker = "\n}"

        start_idx = content.find(start_marker)
        if start_idx == -1:
            print("❌ 在datasets_info.py中找不到DATASETS_URL定义")
            return {}

        # 找到对应的结束位置（需要处理嵌套的大括号）
        brace_count = 0
        current_idx = start_idx + len(start_marker) - 1  # 回到第一个'{'

        for i, char in enumerate(content[current_idx:], current_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        else:
            print("❌ 无法找到DATASETS_URL的结束位置")
            return {}

        # 提取DATASETS_URL部分
        datasets_url_str = content[start_idx:end_idx]

        # 使用exec执行代码片段来获取字典
        local_vars = {}
        exec(datasets_url_str, {}, local_vars)

        datasets_url = local_vars.get('DATASETS_URL', {})
        print(f"✅ 成功从文件加载 {len(datasets_url)} 个传统数据集配置")

        return datasets_url

    except Exception as e:
        print(f"❌ 加载DATASETS_URL失败: {e}")
        print("💡 将使用基本的数据集配置")
        # 返回基本配置作为备用
        return {}


# 动态加载传统数据集配置
DATASETS_URL = load_datasets_url()

# HuggingFace数据集配置
HF_DATASETS = {
    "code_generation_lite": {
        "hf_repo": "livecodebench/code_generation_lite",
        "local_path": "data/code_generation_lite"
    },
    # "humaneval": {
    #     "hf_repo": "openai/openai_humaneval",
    #     "local_path": "data/humaneval_hf"
    # },
    # 可以继续添加更多HF数据集
}


def run_command_with_output(cmd, shell=True):
    """执行命令并实时显示输出"""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=shell,
        text=True,
        universal_newlines=True
    )

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            sys.stdout.write(output)
            sys.stdout.flush()

    return process.poll()


def run_wget_dataset(url):
    """下载传统数据集"""
    return run_command_with_output(f'wget {url} -P ./data -N')


def check_git_updates(local_path):
    """检查git仓库是否有更新"""
    try:
        # 获取远程信息
        result = subprocess.run(["git", "fetch", "origin"], cwd=local_path,
                                capture_output=True, text=True)
        if result.returncode != 0:
            return False, "fetch失败"

        # 获取本地和远程commit hash
        local_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=local_path,
                                    capture_output=True, text=True).stdout.strip()

        # 获取当前分支
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=local_path,
                                capture_output=True, text=True).stdout.strip() or "main"

        # 获取远程hash
        remote_hash = subprocess.run([f"git", "rev-parse", f"origin/{branch}"],
                                     cwd=local_path, capture_output=True, text=True).stdout.strip()

        if not remote_hash:  # 尝试main分支
            remote_hash = subprocess.run(["git", "rev-parse", "origin/main"],
                                         cwd=local_path, capture_output=True, text=True).stdout.strip()

        return local_hash != remote_hash, (local_hash[:8], remote_hash[:8])
    except Exception as e:
        return False, str(e)


def run_git_clone_hf_dataset(hf_repo, local_path):
    """下载HuggingFace数据集，支持智能更新检测"""
    hf_url = f"{os.environ['HF_ENDPOINT']}/datasets/{hf_repo}"

    if os.path.exists(local_path):
        print(f"📁 目录 {local_path} 已存在")

        # 检查是否是git仓库
        if not os.path.exists(os.path.join(local_path, '.git')):
            print(f"⚠️  目录不是git仓库，请手动处理或删除后重新运行")
            return 1

        if is_update:
            print(f"🔍 检查远程更新...")
            has_update, info = check_git_updates(local_path)

            if not has_update:
                if isinstance(info, tuple):
                    print(f"✅ 数据集 {hf_repo} 已是最新版本")
                    print(f"📋 当前版本: {info[0]}")
                else:
                    print(f"❌ 检查更新失败: {info}")
                return 0
            else:
                local_ver, remote_ver = info
                print(f"🆕 发现远程更新!")
                print(f"📋 本地版本: {local_ver}")
                print(f"📋 远程版本: {remote_ver}")
                print(f"🔄 正在更新...")

                result = subprocess.run(["git", "pull"], cwd=local_path,
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ 成功更新 {hf_repo}")
                    # 显示新版本
                    new_hash = subprocess.run(["git", "rev-parse", "HEAD"],
                                              cwd=local_path, capture_output=True, text=True).stdout.strip()
                    print(f"📋 更新后版本: {new_hash[:8]}")
                else:
                    print(f"❌ 更新失败: {result.stderr}")
                return result.returncode
        else:
            print(f"💡 如需检查更新，请设置 is_update = 1 或使用 --update 参数")
            return 0

    # 全新下载
    print(f"🚀 开始下载 HF 数据集: {hf_repo}")
    print(f"📥 目标路径: {local_path}")

    # 确保父目录存在
    parent_dir = os.path.dirname(local_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # 执行git clone
    return_code = run_command_with_output(f'git clone {hf_url} {local_path}')

    if return_code == 0:
        print(f"✅ 成功下载 {hf_repo}")
        # 显示版本信息
        try:
            hash_result = subprocess.run(["git", "rev-parse", "HEAD"],
                                         cwd=local_path, capture_output=True, text=True)
            if hash_result.returncode == 0:
                print(f"📋 版本: {hash_result.stdout.strip()[:8]}")
        except:
            pass
    else:
        print(f"❌ 下载失败 {hf_repo}")

    return return_code


def run_unzip():
    """解压zip文件"""
    if not os.path.exists('data'):
        return
    
    # 使用Python的zipfile模块解压，避免依赖系统unzip命令
    data_dir = 'data'
    zip_files = glob.glob(os.path.join(data_dir, '*.zip'))
    
    if not zip_files:
        print("📦 没有找到需要解压的zip文件")
        return
        
    print(f"📦 找到 {len(zip_files)} 个zip文件，开始解压...")
    
    for zip_path in zip_files:
        try:
            zip_filename = os.path.basename(zip_path)
            print(f"📂 正在解压: {zip_filename}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 获取解压目标目录名（去掉.zip扩展名）
                extract_dir = os.path.join(data_dir, os.path.splitext(zip_filename)[0])
                
                # 如果目标目录已存在，跳过解压
                if os.path.exists(extract_dir):
                    print(f"  ⏭️  目录 {os.path.basename(extract_dir)} 已存在，跳过解压")
                    continue
                
                # 解压到对应目录
                zip_ref.extractall(extract_dir)
                print(f"  ✅ 成功解压到: {os.path.basename(extract_dir)}")
                
        except zipfile.BadZipFile:
            print(f"  ❌ {zip_filename} 不是有效的zip文件")
        except Exception as e:
            print(f"  ❌ 解压 {zip_filename} 失败: {e}")
    
    print("📦 解压完成")


def main():
    parser = argparse.ArgumentParser(description="自动下载数据集工具")
    parser.add_argument("--skip-traditional",
                        action="store_true", help="跳过传统数据集下载")
    parser.add_argument("--skip-hf", action="store_true",
                        help="跳过HuggingFace数据集下载")
    parser.add_argument("--update", "-u", action="store_true", help="启用更新检查")
    parser.add_argument("--hf-only", action="store_true",
                        help="仅下载HuggingFace数据集")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有配置的数据集")

    args = parser.parse_args()

    global is_update
    if args.update:
        is_update = 1

    if args.list:
        print("📋 传统数据集 (zip格式):")
        for name in DATASETS_URL.keys():
            print(f"  - {name}")
        print(f"\n📋 HuggingFace数据集 (git格式):")
        for name, info in HF_DATASETS.items():
            print(f"  - {name}: {info['hf_repo']}")
        return

    # 创建data目录
    os.makedirs('./data', exist_ok=True)

    # 下载传统数据集
    if not args.skip_traditional and not args.hf_only:
        if DATASETS_URL:
            print("🌐 开始下载传统数据集（wget方式）...")
            for dataset, dataset_info in DATASETS_URL.items():
                url = dataset_info['url']
                dataset_name = os.path.basename(url)
                print(f"⬇️  {dataset_name}")
                run_wget_dataset(url)

            print("\n📦 解压zip文件...")
            run_unzip()
        else:
            print("⚠️  没有加载到传统数据集配置，跳过下载")

    # 下载HF数据集
    if not args.skip_hf:
        if HF_DATASETS:
            print("\n🤗 开始下载HuggingFace数据集（git clone方式）...")
            for dataset_name, dataset_info in HF_DATASETS.items():
                hf_repo = dataset_info['hf_repo']
                local_path = dataset_info['local_path']
                print(f"\n📂 {dataset_name}: {hf_repo}")
                run_git_clone_hf_dataset(hf_repo, local_path)
        else:
            print("⚠️  没有配置HuggingFace数据集")

    print("\n🎉 所有数据集下载完成！")


if __name__ == "__main__":
    main()
