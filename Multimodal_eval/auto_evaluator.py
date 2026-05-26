import os
import time
import argparse
import subprocess
import re
from typing import Set, Tuple, List, Optional

# 從您之前創建的函數文件中導入模型分離函數
try:
    from seperate_model_mlp_func import separate_duplex_model
except ImportError:
    print("錯誤：無法找到 'seperate_model_mlp_func.py'。請確保它與 auto_evaluator.py 在同一個目錄下。")
    exit(1)

def merge_lora_checkpoint(checkpoint_path: str, merge_script_path: str = "merge_lora_weights.sh") -> Optional[str]:
    """
    使用 merge script 來合併 LoRA checkpoint。
    
    Args:
        checkpoint_path: 未合併的 checkpoint 路徑 (例如 checkpoint-4000)
        merge_script_path: merge_lora_weights.sh 腳本的路徑
    
    Returns:
        合併後的 checkpoint 路徑 (checkpoint-XXXX-merged)，如果失敗則返回 None
    """
    if not os.path.exists(merge_script_path):
        print(f"警告：找不到 merge script: {merge_script_path}")
        return None
    
    # 期望的合併後路徑
    merged_path = f"{checkpoint_path}-merged"
    
    # 如果已經存在，直接返回
    if os.path.exists(merged_path):
        print(f"  合併後的 checkpoint 已存在: {merged_path}")
        return merged_path
    
    print(f"  正在合併 LoRA checkpoint: {checkpoint_path}")
    print(f"  目標路徑: {merged_path}")
    
    try:
        # 運行 merge script
        result = subprocess.run(
            ["bash", merge_script_path, checkpoint_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 分鐘超時
        )
        
        # 檢查合併後的目錄是否創建成功
        if os.path.exists(merged_path):
            print(f"  ✓ 成功合併 checkpoint: {merged_path}")
            return merged_path
        else:
            print(f"  ✗ 合併腳本執行完成，但找不到輸出目錄: {merged_path}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ 合併 checkpoint 超時: {checkpoint_path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 合併 checkpoint 失敗: {e}")
        print(f"  標準輸出: {e.stdout}")
        print(f"  標準錯誤: {e.stderr}")
        return None
    except Exception as e:
        print(f"  ✗ 合併過程發生異常: {e}")
        return None

# 導入 LLaVA 到 Qwen 轉換函數
try:
    from convert_llava_to_qwen_func import convert_llava_to_qwen
except ImportError:
    print("錯誤：無法找到 'convert_llava_to_qwen_func.py'。請確保它與 auto_evaluator.py 在同一個目錄下。")
    exit(1)

def find_new_checkpoints(model_path: str, eval_step: int, processed_checkpoints: Set[str], start_step: int, is_lora=False, merge_script_path: str = "merge_lora_weights.sh") -> List[Tuple[int, str]]:
    """
    遞歸掃描模型路徑，找出符合步長要求且尚未處理的新 checkpoint。
    跳過所有 'results' 目錄以避免找到評估結果中的 checkpoint。
    
    Args:
        processed_checkpoints: Set of checkpoint paths that have been processed
        is_lora: Whether to search for LoRA merged checkpoints (checkpoint-XXXX-merged pattern)
    
    Returns:
        list[tuple[int, str]]: 列表包含 (step, checkpoint_full_path) 元組
    """
    new_checkpoints = []
    checkpoint_type = "LoRA merged checkpoints" if is_lora else "checkpoints"
    print(f"正在遞歸掃描 '{model_path}' 下的 {checkpoint_type} (從 step {start_step} 開始)...")
    
    if not os.path.isdir(model_path):
        print(f"警告：目錄 '{model_path}' 不存在。")
        return []

    # 根據 is_lora 參數選擇不同的 checkpoint 模式
    if is_lora:
        checkpoint_pattern_merged = re.compile(r"^checkpoint-(\d+)-merged$")
        checkpoint_pattern_unmerged = re.compile(r"^checkpoint-(\d+)$")
    else:
        checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")

    def should_skip_directory(dirpath: str, dirname: str) -> bool:
        """檢查是否應該跳過此目錄"""
        # 跳過名為 'results' 的目錄
        if dirname == 'results':
            return True
        # 跳過路徑中包含 'results' 的目錄
        full_path = os.path.join(dirpath, dirname)
        if '/results/' in full_path or '\\results\\' in full_path:
            return True
        return False

    # 使用 os.walk 遞歸掃描
    for dirpath, dirnames, filenames in os.walk(model_path, topdown=True):
        # 修改 dirnames 來跳過不需要的目錄（topdown=True 時這會影響遞歸行為）
        dirnames[:] = [d for d in dirnames if not should_skip_directory(dirpath, d)]
        
        # 檢查當前目錄下的所有子目錄
        for dirname in dirnames[:]:  # 使用副本遍歷，因為我們可能修改 dirnames
            if is_lora:
                # LoRA 模式：優先尋找 merged checkpoint，如果沒有則嘗試合併
                match_merged = checkpoint_pattern_merged.match(dirname)
                match_unmerged = checkpoint_pattern_unmerged.match(dirname)
                
                if match_merged:
                    step = int(match_merged.group(1))
                    checkpoint_full_path = os.path.join(dirpath, dirname)
                elif match_unmerged:
                    step = int(match_unmerged.group(1))
                    
                    # 先檢查是否符合評估條件
                    if not (step >= start_step and step > 0 and step % eval_step == 0):
                        continue
                    
                    unmerged_checkpoint_path = os.path.join(dirpath, dirname)
                    merged_checkpoint_path = f"{unmerged_checkpoint_path}-merged"
                    
                    # 檢查 merged 版本是否存在
                    if not os.path.exists(merged_checkpoint_path):
                        # 嘗試合併
                        print(f"  發現未合併的 LoRA checkpoint: {dirname} (step {step})")
                        merged_path = merge_lora_checkpoint(unmerged_checkpoint_path, merge_script_path)
                        if merged_path:
                            checkpoint_full_path = merged_path
                        else:
                            print(f"  無法合併 checkpoint，跳過: {dirname}")
                            continue
                    else:
                        checkpoint_full_path = merged_checkpoint_path
                else:
                    continue
            else:
                # 非 LoRA 模式：使用原始邏輯
                match = checkpoint_pattern.match(dirname)
                if not match:
                    continue
                step = int(match.group(1))
                checkpoint_full_path = os.path.join(dirpath, dirname)
            
            if step >= start_step and step > 0 and step % eval_step == 0 and checkpoint_full_path not in processed_checkpoints:
                    try:
                        files_in_ckpt = set(os.listdir(checkpoint_full_path))
                    except OSError:
                        files_in_ckpt = set()
                    is_complete = "pytorch_model.bin" in files_in_ckpt or \
                                any(re.match(r"pytorch_model-(\d+)-of-\1\.bin", f) for f in files_in_ckpt)
                    if is_complete:
                        relative_path = os.path.relpath(checkpoint_full_path, model_path)
                        new_checkpoints.append((step, checkpoint_full_path))
                        print(f"  找到 checkpoint: {relative_path} (step {step})")
                    else:
                        relative_path = os.path.relpath(checkpoint_full_path, model_path)
                        print(f"  發現 {relative_path}，但它似乎不完整，暫時跳過。")

    new_checkpoints.sort(key=lambda x: x[0])
    return new_checkpoints

def generate_and_run_eval_script(gpus:str, results_root_dir: str, model_name: str, step: int, separated_model_path: str, prefix: str = None, visual_bandwidth: int = 64, max_token: int = 4096, temperature: float = 0.01, benchmarks: list[str] | None = None, hidden_size: int = 2560, max_image_resolution: int = None):
    default_benchmarks = "SEEDBench2_Plus MMVet MMStar MMMU_DEV_VAL MathVista_MINI MathVision OCRBench AI2D_TEST"
    benchmark_str = " ".join(benchmarks) if benchmarks else default_benchmarks

    eval_configs = [
        {"gpu": gpus, "datasets": benchmark_str}
    ]

    vlmevalkit_dir = "vlmevalkit"
    if not os.path.isdir(vlmevalkit_dir):
        print(f"Error: {vlmevalkit_dir} not found")
        return

    abs_results_root = os.path.abspath(results_root_dir)
    abs_separated_model_path = os.path.abspath(separated_model_path)
    save_prefix = f"checkpoint-{step}"
    work_dir = os.path.join(abs_results_root, save_prefix, "output")
    os.makedirs(work_dir, exist_ok=True)

    base_env = os.environ.copy()
    cuda_path = "/usr/local/cuda/bin"
    base_env["PATH"] = f"{cuda_path}:{base_env['PATH']}" if "PATH" in base_env else cuda_path
    base_env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["FORCE_LOCAL"] = "True"
    base_env["VLLM_USE_V1"] = "0"
    base_env["VISUAL_BANDWIDTH"] = str(visual_bandwidth)
    base_env["MAX_TOKEN"] = str(max_token)
    base_env["TEMPERATURE"] = str(temperature)
    base_env["SAVE_ROOT"] = abs_results_root
    base_env["MODEL_PATH"] = abs_separated_model_path
    base_env["SAVE_PREFIX"] = save_prefix
    base_env["CUSTOM_PREFIX"] = prefix
    base_env["HIDDEN_SIZE_OF_MODEL"] = str(hidden_size)
    if max_image_resolution is not None:
        base_env["MAX_IMAGE_RESOLUTION"] = str(max_image_resolution)

    # infer_batch_size = 1 if model_name == "DuplexThinkerS2vLLMPrefixCustom" else 4
    infer_batch_size = 4

    try:
        for i, config in enumerate(eval_configs, start=1):
            datasets = config["datasets"].split()
            env_with_gpu = base_env.copy()
            env_with_gpu["CUDA_VISIBLE_DEVICES"] = config["gpu"]

            for dataset in datasets:
                infer_cmd = [
                    "python",
                    "run.py",
                    "--data",
                    dataset,
                    "--model",
                    model_name,
                    "--work-dir",
                    work_dir,
                    "--mode",
                    "infer",
                    "--verbose",
                    "--batch-size",
                    str(infer_batch_size),
                ]
                subprocess.run(infer_cmd, cwd=vlmevalkit_dir, env=env_with_gpu, check=True)

                eval_cmd = [
                    "python",
                    "run.py",
                    "--data",
                    dataset,
                    "--model",
                    model_name,
                    "--work-dir",
                    work_dir,
                    "--nproc",
                    "1",
                    "--verbose",
                    "--judge",
                    "gpt-4o" if dataset != 'MMVet' else "gpt-4-turbo",
                ]
                subprocess.run(eval_cmd, cwd=vlmevalkit_dir, env=env_with_gpu, check=True)

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="自動監控模型訓練並通過生成 Bash 腳本來執行評估。"
    )
    # (您的所有參數定義都保留不變)
    parser.add_argument("--model_path", type=str, required=True, help="模型訓練的運行路徑，其中包含 'checkpoint-*' 子目錄。")
    parser.add_argument("--eval_step", type=int, required=True, help="評估的步長間隔。")
    parser.add_argument("--model_name", type=str, required=True, help="傳遞給評估腳本的模型名稱。")
    parser.add_argument("--start_step", type=int, default=0, help="評估的起始步數。默認為 0。")
    parser.add_argument("--poll_interval", type=int, default=300, help="掃描新 checkpoint 的時間間隔（秒）。")
    parser.add_argument("--save_path", type=str, default=None, help="結果保存的路徑。如果未指定，將使用 model_path 下的 'results' 目錄。")
    parser.add_argument("--prefix", type=str, default="<think>\n\n</think>\n\n", help="自定義前綴字串。")
    parser.add_argument("--max_token", type=int, default=4096, help="最大 token 數量參數，可選。")
    parser.add_argument("--visual_bandwidth", type=int, default=64, help="視覺帶寬參數，可選。")
    parser.add_argument("--temperature", type=float, default=0.01, help="溫度參數，可選。")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None, help="指定要評估的 benchmark 名稱，使用空格分隔，例如: --benchmarks MMMU_DEV_VAL MMVet")
    parser.add_argument("--hidden_size", type=int, default=2560, help="隱藏層大小。")
    parser.add_argument("--max_image_resolution", type=int, default=None, help="最大圖像分辨率（像素數），例如 1024*1024=1048576。如果設置，將作為 MAX_IMAGE_RESOLUTION 環境變量傳遞。")
    parser.add_argument("--qwen_config_dir", type=str, help="Qwen 配置目錄路徑（用於 Qwen25VLCustomvLLM 模型轉換）。")
    parser.add_argument("--reference_dir", type=str, help="參考目錄路徑（用於複製缺失文件，用於 Qwen25VLCustomvLLM 模型轉換）。")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="指定要評估的 GPU 列表，使用逗號分隔，例如: --gpus 0,1,2,3")
    parser.add_argument("--is_lora", action="store_true", help="是否為 LoRA 模型。如果設置，將搜索 'checkpoint-XXXX-merged' 格式的 checkpoint。")
    parser.add_argument("--merge_script_path", type=str, default="merge_lora_weights.sh", help="LoRA 合併腳本的路徑。")
    args = parser.parse_args()
    
    vlmevalkit_path = "vlmevalkit"
    if not os.path.isdir(vlmevalkit_path):
        print(f"錯誤：找不到評估目錄 '{vlmevalkit_path}'。請確保此腳本在與 'vlmevalkit' 目錄相同的層級運行。")
        exit(1)

    processed_checkpoints: Set[str] = set()
    try:
        while True:
            new_checkpoints = find_new_checkpoints(
                args.model_path, 
                args.eval_step, 
                processed_checkpoints,
                args.start_step,
                is_lora=args.is_lora,
                merge_script_path=args.merge_script_path
            )
            print(new_checkpoints, flush=True)

            if not new_checkpoints:
                print(f"未發現新的待評估 checkpoint。將在 {args.poll_interval} 秒後重試...", flush=True)
            else:
                for step, source_checkpoint_path in new_checkpoints:
                    print(f"\n{'='*60}\n發現並準備處理新的 Checkpoint: checkpoint-{step}\n路徑: {source_checkpoint_path}\n{'='*60}", flush=True)
                    
                    # 確定此 checkpoint 的父目錄，並在該目錄下創建 results 子目錄
                    checkpoint_parent_dir = os.path.dirname(source_checkpoint_path)
                    
                    # 如果指定了 save_path，使用它；否則在 checkpoint 父目錄下創建 results
                    if args.save_path:
                        results_dir = args.save_path
                    else:
                        results_dir = os.path.join(checkpoint_parent_dir, "results")
                    
                    separated_models_root_dir = os.path.join(results_dir, "separated_models")
                    os.makedirs(results_dir, exist_ok=True)
                    os.makedirs(separated_models_root_dir, exist_ok=True)
                    
                    separated_model_dest_path = os.path.join(separated_models_root_dir, f"checkpoint-{step}")
                    
                    print(f"結果將保存在: {results_dir}", flush=True)
                    print(f"分離後的模型將保存在: {separated_model_dest_path}", flush=True)

                    print(f"--- [Checkpoint-{step}] 步驟 1/2: 處理模型 ---")
                    # 根據模型類型選擇不同的處理方式
                    if args.model_name == "DuplexThinkerS2" or args.model_name == "Qwen25VLLlamaCustom":
                        print("模型名稱為 'DuplexThinkerS2'，跳過分離步驟，直接使用原始 checkpoint。")
                        separated_model_dest_path = source_checkpoint_path
                    elif args.model_name == "Qwen25VLCustomPrefixCustomvLLM":
                        print(f"模型名稱為 'Qwen25VLCustomPrefixCustomvLLM'，開始轉換 LLaVA checkpoint 到 Qwen 格式...")
                        try:
                            if os.path.exists(separated_model_dest_path):
                                print(f"轉換後的模型目錄已存在，跳過轉換步驟: {separated_model_dest_path}")
                            else:
                                convert_llava_to_qwen(
                                    model_path=source_checkpoint_path,
                                    output_dir=separated_model_dest_path,
                                    qwen_config_dir=args.qwen_config_dir,
                                    reference_dir=args.reference_dir,
                                    save_safetensors=True
                                )
                        except Exception as e:
                            print(f"錯誤：在轉換 checkpoint-{step} 時發生異常: {e}")
                            print("跳過此 checkpoint 的評估。")
                            continue
                    else:
                        # 其他模型使用分離邏輯
                        try:
                            if os.path.exists(separated_model_dest_path):
                                print(f"分離後的模型目錄已存在，跳過分離步驟: {separated_model_dest_path}")
                            else:
                                separate_duplex_model(
                                    model_path=source_checkpoint_path,
                                    output_dir=separated_model_dest_path
                                )
                        except Exception as e:
                            print(f"錯誤：在分離 checkpoint-{step} 時發生異常: {e}")
                            print("跳過此 checkpoint 的評估。")
                            continue

                    print(f"\n--- [Checkpoint-{step}] 步驟 2/2: 執行評估 ---")
                    # --- MODIFIED: 呼叫新的 bash 腳本生成與執行函數 ---
                    generate_and_run_eval_script(
                        gpus=args.gpus,
                        results_root_dir=results_dir,
                        model_name=args.model_name,
                        step=step,
                        separated_model_path=separated_model_dest_path,
                        prefix=args.prefix,
                        visual_bandwidth=args.visual_bandwidth,
                        max_token=args.max_token,
                        temperature=args.temperature,
                        benchmarks=args.benchmarks,
                        hidden_size=args.hidden_size,
                        max_image_resolution=args.max_image_resolution
                    )

                    processed_checkpoints.add(source_checkpoint_path)
                    print(f"\n{'*'*60}\n成功完成對 Checkpoint-{step} 的所有處理。\n{'*'*60}")
            
            if args.poll_interval <= 0:
                print(f'Stop Evaluation', flush=True)
                return
            time.sleep(args.poll_interval)
            

    except KeyboardInterrupt:
        print("\n監控已手動停止。")

if __name__ == "__main__":
    main()