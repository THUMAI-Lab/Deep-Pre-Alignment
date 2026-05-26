import os
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='运行模型评测')
parser.add_argument('-c', '--config_path', type=str, default='', help='配置文件路径')
args = parser.parse_args()

config_path = args.config_path

"""
以gsm_hard为例，将gsmhard_gen_8a1400.py复制为gsmhard_gen_8a1400_new.py

opencompass/configs/datasets/gsm_hard/gsmhard_gen_8a1400.py

mkdir opencompass/configs/mb_internal/datasets/gsm_hard

cp opencompass/configs/datasets/gsm_hard/gsmhard_gen_8a1400.py opencompass/configs/mb_internal/datasets/gsm_hard/gsmhard_gen_8a1400_mb.py
"""

def safe_mkdir(path):
    """安全创建目录"""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        return True
    except Exception as e:
        print(f"创建目录 {path} 失败: {str(e)}")
        return False

def safe_copy_file(src, dst):
    """安全复制文件"""
    try:
        if not os.path.exists(src):
            print(f"源文件不存在: {src}")
            return False
            
        # 确保目标目录存在
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        # 读取源文件并写入目标文件
        with open(src, 'r', encoding='utf-8') as f_src:
            content = f_src.read()
        with open(dst, 'w', encoding='utf-8') as f_dst:
            f_dst.write(content)
            
        return True
    except Exception as e:
        print(f"复制文件失败 {src} -> {dst}: {str(e)}")
        return False

def main():
    if not config_path:
        print("请提供配置文件路径")
        return
        
    # 解析配置文件路径
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    config_name_no_ext = os.path.splitext(config_name)[0]
    
    # 构建目标路径
    mb_config_dir = config_dir.replace('/configs/datasets/', '/configs/mb_internal/datasets/')
    mb_config_path = os.path.join(mb_config_dir, 
                                 config_name_no_ext + '_mb.py')
    
    # 创建目录
    if not safe_mkdir(mb_config_dir):
        return
        
    # 复制文件
    if not safe_copy_file(config_path, mb_config_path):
        return
        
    print(f"成功创建MB配置文件: {mb_config_path}")

if __name__ == '__main__':
    main()




