conda init
conda activate sft_msswift
cd $(cd $(dirname ${BASH_SOURCE});pwd)
cd ..

MODEL_PATH="../output/4B_DPA/4B-open-DPA-PT/vx-xxxxxxxx-xxxxxx/checkpoint-2180"
OUTPUT_DIR="../output/4B_DPA"
CUSTOM_REGISTER_PATH="register_DPA.py"

run_name=4B-open-DPA-SFT
OUTPUT_DIR=$OUTPUT_DIR/$run_name

nproc_per_node=8
export MAX_PIXELS=262144
export KEEP_FIRST_TURN_ONLY=false
export TRIME_PERCEIVER_AFTER_IMAGE=false
export ROOT_IMAGE_DIR="/path/to/your/MAmmoTH-VL-Instruct-12M/single_image_data"

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-12348}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=80

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
swift sft \
    --model "$MODEL_PATH" \
    --custom_register_path "$CUSTOM_REGISTER_PATH" \
    --dataset /path/to/your/mammoth_ov_single_1M.jsonl \
    --val_dataset /path/to/your/mammoth_ov_single_1M_val_last1000.jsonl \
    --load_from_cache_file true \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --vit_lr 2e-6 \
    --aligner_lr 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 16 \
    --dataset_num_proc 64 \
    --report_to wandb \
    --run_name $run_name \
    --push_to_hub false \
    --include_tokens_per_second true \
    --include_num_input_tokens_seen true \
    --overwrite_output_dir true \
    --save_safetensors false \
    --ddp_timeout 180000000 \
    --deepspeed zero3 \
    --streaming false \
    --max_steps 8000 \
    --warmup_steps 500 \
    --freeze_llm False \
    --freeze_vit False \
    --freeze_aligner False \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --weight_decay 0.01 \
    --adam_beta2 0.98 \
    --average_tokens_across_devices true  \
    --max_length 8192 \
    --attn_impl flash_attn



echo "Training completed! Model saved to: $OUTPUT_DIR"
