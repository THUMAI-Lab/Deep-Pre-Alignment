import torch
import argparse
import os
from transformers import AutoModel
# # 确保 DuplexPTS2ForCausalLM 类的定义可以被访问到
# from duplex_pts2_model_bw64_mlp.modeling_duplex_pts2_mlp import DuplexPTS2ForCausalLM

def separate_duplex_model(model_path: str, output_dir: str):
    """
    加载一个 DuplexPTS2ForCausalLM 复合模型，并将其分离成 Perceiver、Thinker、对齐层和residual MLP等部分，
    然后分别保存到指定的目录。

    Args:
        model_path (str): 完整的 DuplexPTS2ForCausalLM 模型检查点路径。
        output_dir (str): 用于保存分离后模型组件的根目录。
    """
    # --- 1. 加载完整模型 ---
    print(f"Loading the full composite model from {model_path}...")
    try:
        # full_model = DuplexPTS2ForCausalLM.from_pretrained(model_path)
        full_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        full_model.eval()
        print("Full model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # --- 2. 创建输出目录 ---
    perceiver_path = os.path.join(output_dir, "perceiver")
    thinker_path = os.path.join(output_dir, "thinker")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(perceiver_path, exist_ok=True)
    os.makedirs(thinker_path, exist_ok=True)
    print(f"Output directories created at {output_dir}")

    if hasattr(full_model, 'model'):
        model = full_model.model
    else:
        model = full_model

    # --- 3. 保存 Perceiver ---
    print(f"Saving Perceiver model to {perceiver_path}...")
    perceiver_module = model.perceiver
    perceiver_module.save_pretrained(perceiver_path, safe_serialization=False)
    # 同时保存对应的 processor
    if hasattr(model, 'p_processor'):
        model.p_processor.save_pretrained(perceiver_path)
        print("Perceiver model and processor saved.")
    else:
        print("Perceiver model saved (p_processor not found).")

    # --- 4. 保存 Thinker ---
    print(f"Saving Thinker model to {thinker_path}...")
    thinker_module = model.thinker
    thinker_module.save_pretrained(thinker_path)
    # 同时保存对应的 tokenizer
    if hasattr(model, 't_tokenizer'):
        model.t_tokenizer.save_pretrained(thinker_path)
        print("Thinker model and tokenizer saved.")
    else:
        print("Thinker model saved (t_tokenizer not found).")

    # --- 5. 保存对齐层的权重 ---
    alignment_layer_path = os.path.join(output_dir, "linear_align_dim.pth")
    print(f"Saving alignment layer weights to {alignment_layer_path}...")
    alignment_layer_state_dict = model.linear_align_dim.state_dict()
    torch.save(alignment_layer_state_dict, alignment_layer_path)
    print("Alignment layer saved.")
    
    print(f"model: {model}")
    # --- 6. 保存normalization层的权重 ---
    if hasattr(model, 'norm'):
        normalization_layer_path = os.path.join(output_dir, "norm_weights.pth")
        print(f"Saving normalization layer weights to {normalization_layer_path}...")
        normalization_layer_state_dict = model.norm.state_dict()
        torch.save(normalization_layer_state_dict, normalization_layer_path)
        print("Normalization layer saved.")
    print("model: ", model)
    
    # --- 6. 保存residual_add_mlp的权重（如果存在）---
    if hasattr(model, 'residual_add_mlp'):
        residual_mlp_path = os.path.join(output_dir, "residual_add_mlp.pth")
        print(f"Saving residual_add_mlp weights to {residual_mlp_path}...")
        residual_mlp_state_dict = model.residual_add_mlp.state_dict()
        torch.save(residual_mlp_state_dict, residual_mlp_path)
        print("Residual MLP saved.")
    else:
        print("Residual MLP not found in model, skipping.")
    if hasattr(model, 'residual_concat_mlp'):
        residual_concat_mlp_path = os.path.join(output_dir, "residual_concat_mlp.pth")
        print(f"Saving residual_concat_mlp weights to {residual_concat_mlp_path}...")
        residual_concat_mlp_state_dict = model.residual_concat_mlp.state_dict()
        torch.save(residual_concat_mlp_state_dict, residual_concat_mlp_path)
        print("Residual Concat MLP saved.")
    else:
        print("Residual Concat MLP not found in model, skipping.")
    
    print("\nSeparation complete.")
    print(f"All components saved in: {output_dir}")

    # --- 7. 清理 ---
    del full_model
    torch.cuda.empty_cache()


def main():
    """
    提供命令行接口，用于执行模型分离操作。
    """
    parser = argparse.ArgumentParser(
        description="Separate the DuplexPTS2ForCausalLM model into its Perceiver, Thinker, alignment layer, and residual MLP components."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the full, fine-tuned unified model checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the separated models and weights will be saved."
    )
    args = parser.parse_args()

    # 调用核心函数
    separate_duplex_model(model_path=args.model_path, output_dir=args.output_dir)


if __name__ == "__main__":
    # 当这个文件作为主脚本运行时，执行 main 函数
    main()