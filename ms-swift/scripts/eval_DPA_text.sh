cd $(cd $(dirname ${BASH_SOURCE});pwd)
conda init
conda activate texteval

BASE_DIR="$1"
shift
CKPT_IDS="$@"        # 例如 3000 5000 7000

OPENCOMPASS_DIR="../../opencompass"

if [ -z "$BASE_DIR" ] || [ -z "$CKPT_IDS" ]; then
  echo "Usage: bash run_all_checkpoints.sh <BASE_DIR> <ckpt_id1> <ckpt_id2> ..."
  exit 1
fi

cd "$OPENCOMPASS_DIR"

for CKPT_ID in $CKPT_IDS; do
  CKPT_NAME="checkpoint-${CKPT_ID}"
  CKPT_PATH="$BASE_DIR/results/separated_models/$CKPT_NAME"

  MODEL_PATH="$CKPT_PATH/thinker"
  SAVE_DIR="$BASE_DIR/results/${CKPT_NAME}-text"

  if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️ thinker 模型不存在，跳过: $MODEL_PATH"
    continue
  fi

  mkdir -p "$SAVE_DIR"

  OPENCOMPASS_RUN_DIR="$SAVE_DIR" \
  BATCH_SIZE=128 \
  bash run_model.sh \
    --dataset mmlu_redux_0shot_simple_evals_gen_fe2877,\
gpqa_openai_simple_evals_gen_5aeece,\
math_prm800k_500_0shot_cot_v2_gen_11c4b5\
    --model_path "$MODEL_PATH" \
    --infer_type vllm_qwen3_nothink_sft_reproduce \
    --max-num-workers 1 \
    --reuse

done

echo "All checkpoints finished."
