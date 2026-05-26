import os
import json
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoConfig


def find_weight_files(model_dir: str):
    """查找模型权重文件"""
    p = Path(model_dir)
    # 优先分片 bin
    index_json = p / "pytorch_model.bin.index.json"
    if index_json.exists():
        with open(index_json, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]  # key -> shard filename
        files = sorted({p / fname for fname in weight_map.values()})
        return "index_bin", files
    # 兼容 safetensors
    st_index = p / "model.safetensors.index.json"
    if st_index.exists():
        with open(st_index, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        files = sorted({p / fname for fname in weight_map.values()})
        return "index_safetensors", files
    single_st = p / "model.safetensors"
    if single_st.exists():
        return "single_safetensors", [single_st]
    pt_bin = p / "pytorch_model.bin"
    if pt_bin.exists():
        return "bin", [pt_bin]
    raise FileNotFoundError(f"未找到权重文件: {model_dir}")


def load_llava_state_dict(model_dir: str):
    """加载 LLaVA 模型的 state_dict"""
    fmt, files = find_weight_files(model_dir)
    state = {}
    if fmt in ("index_safetensors", "single_safetensors"):
        from safetensors import safe_open
        for fp in files:
            with safe_open(fp, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
    elif fmt == "index_bin":
        for fp in files:
            shard = torch.load(fp, map_location="cpu")
            state.update(shard)
    else:
        state = torch.load(files[0], map_location="cpu")
    return state


def map_key(k: str) -> str:
    """映射 LLaVA 键名到 Qwen 键名"""
    if k.startswith("model.vlm.visual."):
        return k.replace("model.vlm.visual.", "model.visual.")
    if k.startswith("model.vlm.language_model."):
        return k.replace("model.vlm.language_model.", "model.language_model.")
    return k


def convert_state_dict(llava_sd: dict):
    """转换 state_dict 的键名"""
    return {map_key(k): v for k, v in llava_sd.items()}


def merge_into_qwen(qwen_model, converted_sd: dict):
    """将转换后的权重合并到 Qwen 模型"""
    qwen_sd = qwen_model.state_dict()
    load_sd = {}
    matched = 0
    skipped_shape = []
    missing_in_llava = []
    for k in qwen_sd.keys():
        if k in converted_sd:
            if qwen_sd[k].shape != converted_sd[k].shape:
                skipped_shape.append((k, qwen_sd[k].shape, converted_sd[k].shape))
            else:
                load_sd[k] = converted_sd[k]
                matched += 1
        else:
            missing_in_llava.append(k)
    # 填充未匹配部分
    for k, v in qwen_sd.items():
        if k not in load_sd:
            load_sd[k] = v
    qwen_model.load_state_dict(load_sd, strict=False)
    return {
        "matched": matched,
        "total_qwen_keys": len(qwen_sd),
        "skipped_shape": skipped_shape,
        "missing_in_llava": missing_in_llava
    }


def copy_missing_files_and_replace_config(reference_dir: str, output_dir: str):
    """
    从参考目录复制缺失的文件到输出目录，并替换 config.json
    
    Args:
        reference_dir: 参考目录路径（包含完整文件的目录）
        output_dir: 输出目录路径（转换后的模型目录）
    """
    ref_path = Path(reference_dir)
    out_path = Path(output_dir)
    
    if not ref_path.exists():
        print(f"警告：参考目录不存在: {reference_dir}")
        return
    
    if not out_path.exists():
        print(f"警告：输出目录不存在: {output_dir}")
        return
    
    # 获取参考目录中的所有文件
    ref_files = set()
    for item in ref_path.iterdir():
        if item.is_file():
            ref_files.add(item.name)
    
    # 获取输出目录中已有的文件
    out_files = set()
    for item in out_path.iterdir():
        if item.is_file():
            out_files.add(item.name)
    
    # 复制缺失的文件（排除 config.json，因为需要单独处理）
    missing_files = ref_files - out_files - {"config.json"}
    if missing_files:
        print(f"发现 {len(missing_files)} 个缺失文件，开始复制...")
        for filename in missing_files:
            src = ref_path / filename
            dst = out_path / filename
            try:
                shutil.copy2(src, dst)
                print(f"  已复制: {filename}")
            except Exception as e:
                print(f"  复制 {filename} 时出错: {e}")
    else:
        print("没有发现缺失的文件。")
    
    # 单独处理 config.json：如果输出目录中没有，则复制；如果已有，则替换
    config_src = ref_path / "config.json"
    config_dst = out_path / "config.json"
    if config_src.exists():
        try:
            shutil.copy2(config_src, config_dst)
            if config_dst.exists():
                print(f"已替换 config.json")
            else:
                print(f"已复制 config.json")
        except Exception as e:
            print(f"处理 config.json 时出错: {e}")
    else:
        print(f"警告：参考目录中不存在 config.json")


def convert_llava_to_qwen(
    model_path: str,
    output_dir: str,
    qwen_config_dir: str,
    reference_dir: str,
    save_safetensors: bool = True
):
    """
    将 LLaVA checkpoint 转换为 Qwen 格式，并复制必要的文件
    
    Args:
        model_path: LLaVA checkpoint 目录路径
        output_dir: 输出目录路径
        qwen_config_dir: Qwen 配置目录路径
        reference_dir: 参考目录路径（用于复制缺失的文件）
        save_safetensors: 是否保存为 safetensors 格式
    """
    print(f"开始转换 LLaVA checkpoint: {model_path} -> {output_dir}")
    
    # 1. 加载 LLaVA 权重
    print("步骤 1/4: 加载 LLaVA Baseline 分片权重...")
    llava_sd = load_llava_state_dict(model_path)
    print(f"LLaVA state_dict 张量数: {len(llava_sd)}")
    
    # 2. 键名映射转换
    print("步骤 2/4: 键名映射转换...")
    converted = convert_state_dict(llava_sd)
    print(f"转换后键数量: {len(converted)}")
    print(qwen_config_dir)
    # 3. 加载目标 Qwen 配置并实例化模型
    print("步骤 3/4: 加载目标 Qwen 配置并实例化模型...")
    config = AutoConfig.from_pretrained(
        qwen_config_dir,
        trust_remote_code=True
    )
    qwen_model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True
    )
    
    # 4. 合并权重到 Qwen
    print("合并权重到 Qwen...")
    report = merge_into_qwen(qwen_model, converted)
    print(f"匹配键: {report['matched']} / {report['total_qwen_keys']}")
    if report["skipped_shape"]:
        print(f"形状不匹配: {len(report['skipped_shape'])} (前10)")
        for k, s1, s2 in report["skipped_shape"][:10]:
            print(f"  - {k}: Qwen {s1} vs LLaVA {s2}")
    if report["missing_in_llava"]:
        print(f"LLaVA 缺失键: {len(report['missing_in_llava'])} (前10)")
        for k in report["missing_in_llava"][:10]:
            print(f"  - {k}")
    
    # 5. 保存到输出目录
    print("步骤 4/4: 保存到输出目录 (Transformers 接口)...")
    os.makedirs(output_dir, exist_ok=True)
    qwen_model.save_pretrained(output_dir, safe_serialization=save_safetensors)
    print(f"已保存转换后的模型: {output_dir}")
    
    # 6. 复制缺失的文件并替换 config.json
    print("\n复制缺失的文件并替换 config.json...")
    copy_missing_files_and_replace_config(reference_dir, output_dir)
    
    # 7. 清理
    del qwen_model
    torch.cuda.empty_cache()
    
    print(f"\n转换完成！输出目录: {output_dir}")


def main():
    """提供命令行接口"""
    import argparse
    parser = argparse.ArgumentParser(
        description="将 LLaVA checkpoint 转换为 Qwen 格式"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="LLaVA checkpoint 目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录路径"
    )
    parser.add_argument(
        "--qwen_config_dir",
        type=str,
        help="Qwen 配置目录路径"
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="参考目录路径（用于复制缺失的文件）"
    )
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        default=True,
        help="是否保存为 safetensors 格式"
    )
    args = parser.parse_args()
    
    convert_llava_to_qwen(
        model_path=args.model_path,
        output_dir=args.output_dir,
        qwen_config_dir=args.qwen_config_dir,
        reference_dir=args.reference_dir,
        save_safetensors=args.save_safetensors
    )


if __name__ == "__main__":
    main()

