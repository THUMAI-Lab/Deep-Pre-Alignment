#!/bin/bash

#===============================================================================
# OpenCompass 多节点 PytorchJob 双模型测试脚本
# 
# 使用方法：在每个节点上执行相同的脚本
# bash run_multinode_pytorchjob_dual_models.sh
# 
# 脚本会根据 PytorchJob 环境变量自动：
# 1. 检测当前节点角色 (Master/Worker)
# 2. 生成对应的配置文件（包含两个模型）
# 3. 启动相应的评测流程
#===============================================================================

set -e  # 遇到错误立即退出

echo "🚀 OpenCompass 多节点 PytorchJob 多模型评测自动化脚本启动..."
echo "========================================================"

# 1. 检查必要的环境变量
echo "🔍 检查 PytorchJob 环境变量..."
required_vars=("WORLD_SIZE" "RANK" "MASTER_ADDR" "MASTER_PORT" "MODELS_TO_EVAL")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "❌ 错误: 缺少环境变量 $var"
        exit 1
    fi
    echo "   $var=${!var}"
done

# 检查数据集配置 - 支持两种方式
if [[ -n "$DATASETS_TO_EVAL" ]]; then
    echo "   DATASETS_TO_EVAL=${DATASETS_TO_EVAL} (直接指定)"
elif [[ -n "$DATASET_CONFIG_PATH" ]]; then
    echo "   DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH} (从配置文件提取)"
    echo "   DATASET_VAR_NAME=${DATASET_VAR_NAME:-hf_infer_datasets_total}"
else
    echo "❌ 错误: 必须设置 DATASETS_TO_EVAL 或 DATASET_CONFIG_PATH 环境变量"
    exit 1
fi

# 检查可选的环境变量
echo "🔍 检查可选环境变量..."
echo "   OC_RESULT_ROOT=${OC_RESULT_ROOT:-./outputs}"

# 2. 设置必要的环境变量
# 注意: PytorchJob环境中WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT已由K8s自动设置
export PYTHONPATH=$(pwd)  # 仅设置Python路径

# 获取每节点GPU数量
# PET_NPROC_PER_NODE: PytorchJob中的标准环境变量，值可能是"auto"或具体数字
if [[ "${PET_NPROC_PER_NODE:-auto}" == "auto" ]]; then
    GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "🎮 自动检测到 $GPUS_PER_NODE 个GPU"
else
    GPUS_PER_NODE=${PET_NPROC_PER_NODE}
    echo "🎮 使用配置的GPU数量: $GPUS_PER_NODE"
fi

# GPU数量校验
if [[ $GPUS_PER_NODE -eq 0 ]]; then
    echo "⚠️  警告: 未检测到GPU，使用默认值8"
    GPUS_PER_NODE=8
fi

# 3. 确定节点角色
if [[ $RANK -eq 0 ]]; then
    NODE_ROLE="master"
    echo "🎯 当前节点: Master (RANK=0)"
else
    NODE_ROLE="worker"
    echo "🎯 当前节点: Worker (RANK=$RANK)"
fi

# 4. 处理环境变量并设置配置文件和工作目录
# 解析逗号分隔的模型列表
IFS=',' read -ra MODELS_ARRAY <<< "$MODELS_TO_EVAL"

# 处理数据集列表 - 支持两种方式
if [[ -n "$DATASETS_TO_EVAL" ]]; then
    # 方式1: 直接从环境变量获取
    IFS=',' read -ra DATASETS_ARRAY <<< "$DATASETS_TO_EVAL"
    echo "📊 使用直接指定的数据集: ${DATASETS_ARRAY[@]}"
else
    # 方式2: 从配置文件提取
    echo "🔍 从配置文件提取数据集列表..."
    DATASET_VAR_NAME="${DATASET_VAR_NAME:-hf_infer_datasets_total}"
    
    # 使用提取脚本获取数据集列表
    EXTRACTED_DATASETS=$(python examples/pytorch_job/extract_datasets.py "$DATASET_CONFIG_PATH" "$DATASET_VAR_NAME" 2>/dev/null)
    
    if [[ $? -eq 0 && -n "$EXTRACTED_DATASETS" ]]; then
        IFS=',' read -ra DATASETS_ARRAY <<< "$EXTRACTED_DATASETS"
        echo "📊 从配置文件提取的数据集: ${DATASETS_ARRAY[@]}"
        echo "   配置文件: $DATASET_CONFIG_PATH"
        echo "   变量名: $DATASET_VAR_NAME"
        
        # 将提取的数据集设置为环境变量，供后续使用
        export DATASETS_TO_EVAL="$EXTRACTED_DATASETS"
    else
        echo "❌ 从配置文件提取数据集失败，使用默认数据集"
        DATASETS_ARRAY=("ARC-c" "ARC-e")  # 回退到默认数据集
        export DATASETS_TO_EVAL="${DATASETS_ARRAY[0]},${DATASETS_ARRAY[1]}"
    fi
fi

echo "🤖 要评测的模型: ${MODELS_ARRAY[@]}"
echo "📊 最终数据集列表: ${DATASETS_ARRAY[@]}"

# SHARED_CONFIG="examples/eval_pytorchjob_dual_models.py"

# 使用OC_RESULT_ROOT作为输出根目录
RESULT_ROOT="${OC_RESULT_ROOT:-./outputs}"
# 使用JOB_ID组织目录结构
# 所有节点使用相同的work_dir和reuse_name（真正的共享存储模式）
SHARED_WORK_DIR="${RESULT_ROOT}/job_dual_models_${JOB_ID:-default}"
SHARED_CONFIG="${RESULT_ROOT}/eval_pytorchjob_dual_models.py"
WORK_DIR="$SHARED_WORK_DIR"  # 所有节点使用共享工作目录
REUSE_NAME="job_dual_models_${JOB_ID:-default}"  # 双模型实验目录

echo "📁 共享工作目录: $WORK_DIR (所有节点共享)"
echo "🔄 实验复用名称: $REUSE_NAME (所有节点共享)"
echo "⚙️ 共享配置: $SHARED_CONFIG"

# 5. 生成配置文件 (只有Master节点生成)
if [[ $RANK -eq 0 ]]; then
    echo "🛠️ Master节点生成多模型共享配置文件..."
    # ✅ 使用 max-workers-per-gpu=1 确保GPU资源严格按顺序分配，避免冲突
    echo "✅ 使用 max-workers-per-gpu=1 确保GPU任务调度的稳定性"
    echo "   优势: 避免GPU资源竞争，防止OOM，确保任务按顺序执行"
    
    # 构建动态参数列表
    MODELS_ARGS=()
    for model in "${MODELS_ARRAY[@]}"; do
        MODELS_ARGS+=("$model")
    done
    
    DATASETS_ARGS=()
    for dataset in "${DATASETS_ARRAY[@]}"; do
        DATASETS_ARGS+=("$dataset")
    done
    
    echo "🔧 生成配置参数:"
    echo "   模型参数: ${MODELS_ARGS[@]}"
    echo "   数据集参数: ${DATASETS_ARGS[@]}"
    
    python examples/pytorch_job/generate_multinode_config.py \
        --models "${MODELS_ARGS[@]}" \
        --datasets "${DATASETS_ARGS[@]}" \
        --output "$SHARED_CONFIG" \
        --max-tasks-per-node $GPUS_PER_NODE \
        --max-workers-per-gpu 1 \
        --force
    
    if [[ $? -ne 0 ]]; then
        echo "❌ 配置文件生成失败"
        exit 1
    fi
    echo "✅ 多模型共享配置文件生成成功: $SHARED_CONFIG"
else
    # Worker节点等待配置文件生成
    echo "⏳ Worker节点等待配置文件..."
    timeout 30 bash -c "while [[ ! -f '$SHARED_CONFIG' ]]; do sleep 1; done"
    if [[ ! -f "$SHARED_CONFIG" ]]; then
        echo "❌ 等待配置文件超时"
        exit 1
    fi
    echo "✅ 找到共享配置文件: $SHARED_CONFIG"
fi

# 6. 节点同步 (确保所有节点都准备就绪)
echo "🔄 节点同步等待..."
sleep 3

# 7. 启动多节点评测
echo "🚀 启动多模型多节点评测..."
echo "   节点角色: $NODE_ROLE"
echo "   世界大小: $WORLD_SIZE"
echo "   节点排名: $RANK"
echo "   Master地址: $MASTER_ADDR:$MASTER_PORT"
echo "   每节点GPU: $GPUS_PER_NODE"
echo "   模型数量: ${#MODELS_ARRAY[@]} (${MODELS_ARRAY[@]})"

# 运行评测命令
timeout 6000 python run.py "$SHARED_CONFIG" \
    --work-dir "$WORK_DIR" \
    --reuse "$REUSE_NAME" \
    --pytorchjob \
    --max-num-workers $(($GPUS_PER_NODE * 2)) || {
    
    echo "⚠️ 评测运行超时或失败，检查日志..."
    
    # 输出部分日志用于调试
    if [[ -d "$WORK_DIR" ]]; then
        echo "📋 最近的日志文件:"
        find "$WORK_DIR" -name "*.out" -type f -newer "$SHARED_CONFIG" | head -3
        echo "📄 日志片段:"
        find "$WORK_DIR" -name "*.out" -type f -newer "$SHARED_CONFIG" | head -1 | xargs tail -10 2>/dev/null || echo "无可用日志"
    fi
    
    exit 1
}

# 8. 结果检查和输出
echo "🎉 多模型评测完成！检查结果..."

# 检查共享工作目录的结果
if [[ -d "$WORK_DIR" ]]; then
    if [[ $RANK -eq 0 ]]; then
        echo "📊 Master节点汇总共享目录结果:"
        
        # 统计所有预测文件
        TOTAL_PREDICTION_FILES=$(find "$WORK_DIR" -name "*.json" -path "*/predictions/*" | wc -l)
        echo "   📁 共享目录总预测文件: $TOTAL_PREDICTION_FILES"
        
        # 按模型统计预测文件
        echo "   📊 各模型预测文件统计:"
        for model_path in "${MODELS_ARRAY[@]}"; do
            # 提取模型名称（去掉路径，只保留最后一部分）
            model_name=$(basename "$model_path")
            model_files=$(find "$WORK_DIR" -name "*.json" -path "*/predictions/$model_name/*" | wc -l)
            echo "     $model_name: $model_files 个文件"
        done
        
        # 统计各类型文件
        NODE_FILES=$(find "$WORK_DIR" -name "*_node*.json" -path "*/predictions/*" | wc -l)
        MERGED_FILES=$(find "$WORK_DIR" -name "*.json" -path "*/predictions/*" ! -name "*_node*" ! -name "*_gpu*" | wc -l)
        echo "   📄 节点预测文件: $NODE_FILES"
        echo "   🔗 最终合并文件: $MERGED_FILES"
        
        # 检查摘要报告
        SUMMARY_FILES=$(find "$WORK_DIR" -name "summary_*.md" -o -name "summary_*.txt" | head -1)
        if [[ -n "$SUMMARY_FILES" ]]; then
            echo "📋 评测摘要:"
            cat "$SUMMARY_FILES" | tail -15
        fi
        
    else
        echo "📊 Worker节点 $RANK 完成，结果已保存到共享目录"
        PREDICTION_FILES=$(find "$WORK_DIR" -name "*_node${RANK}_*.json" -path "*/predictions/*" | wc -l)
        echo "   📄 本节点生成的预测文件: $PREDICTION_FILES"
    fi
else
    echo "⚠️ 共享工作目录不存在或为空"
fi

# 9. 清理和完成
echo "🧹 清理临时文件..."
# 保留配置文件和结果，只清理可能的临时文件
find tmp/ -name "*_params.py" -mmin +10 -delete 2>/dev/null || true

echo "✅ $NODE_ROLE 节点 (RANK=$RANK) 多模型评测任务完成!"
echo "📁 共享工作目录: $WORK_DIR"
echo "🔄 节点复用名称: $REUSE_NAME"
echo "⚙️ 共享配置: $SHARED_CONFIG"
echo "🎯 测试模型: ${MODELS_ARRAY[@]} (数量: ${#MODELS_ARRAY[@]})"
echo "========================================================"
