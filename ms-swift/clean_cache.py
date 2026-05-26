#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""清理数据集缓存文件"""
import os
import glob
from modelscope.hub.utils.utils import get_cache_dir

def clean_cache():
    cache_dir = get_cache_dir()
    print(f'Cache dir: {cache_dir}')
    
    # 查找所有相关的缓存目录
    dataset_cache = os.path.join(cache_dir, 'datasets')
    map_cache = os.path.join(cache_dir, 'datasets', 'map_cache')
    
    print(f'\nDataset cache: {dataset_cache}')
    print(f'Map cache: {map_cache}')
    
    # 查找所有 .arrow 文件
    arrow_files = []
    if os.path.exists(dataset_cache):
        for root, dirs, files in os.walk(dataset_cache):
            for f in files:
                if f.endswith('.arrow'):
                    arrow_files.append(os.path.join(root, f))
    
    print(f'\nFound {len(arrow_files)} .arrow cache files')
    
    # 查找与 mammoth_ov 相关的缓存
    mammoth_files = []
    for f in arrow_files:
        if 'mammoth' in f.lower() or 'ov' in f.lower():
            mammoth_files.append(f)
    
    print(f'Found {len(mammoth_files)} files related to mammoth_ov')
    
    if mammoth_files:
        print('\nFiles to delete:')
        for f in mammoth_files[:10]:  # 只显示前10个
            print(f'  {f}')
        if len(mammoth_files) > 10:
            print(f'  ... and {len(mammoth_files) - 10} more files')
        
        # 删除文件
        print('\nDeleting cache files...')
        deleted = 0
        for f in mammoth_files:
            try:
                os.remove(f)
                deleted += 1
            except Exception as e:
                print(f'Error deleting {f}: {e}')
        
        print(f'Deleted {deleted} cache files')
    else:
        print('\nNo mammoth_ov related cache files found')
    
    # 也清理整个 map_cache 目录（如果存在）
    if os.path.exists(map_cache):
        map_files = glob.glob(os.path.join(map_cache, '*'))
        print(f'\nFound {len(map_files)} files in map_cache')
        if map_files:
            print('Cleaning map_cache...')
            for f in map_files:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        import shutil
                        shutil.rmtree(f)
                except Exception as e:
                    print(f'Error deleting {f}: {e}')
            print('map_cache cleaned')

if __name__ == '__main__':
    clean_cache()



















