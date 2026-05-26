conda init
conda activate sft_msswift
cd $(cd $(dirname ${BASH_SOURCE});pwd)
cd ..

MODEL_PATH="../DPA-4B-init"
OUTPUT_DIR="../output/4B_DPA"
CUSTOM_REGISTER_PATH="register_DPA.py"

run_name=4B-open-DPA-PT
OUTPUT_DIR=$OUTPUT_DIR/$run_name

nproc_per_node=8
export MAX_PIXELS=262144
export KEEP_FIRST_TURN_ONLY=true
export TRIME_PERCEIVER_AFTER_IMAGE=true
export ROOT_IMAGE_DIR="/path/to/your/LLaVA-Pretrain"

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
    --dataset /path/to/your/blip_laion_cc_sbu_558k_shuffled.json \
    --template duplex_forward_visual_tokens_arch \
    --load_from_cache_file true \
    --split_dataset_ratio 0.002 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 100 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 8 \
    --dataset_num_proc 32 \
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
    --max_steps 2180 \
    --freeze_llm true \
    --freeze_vit true \
    --freeze_aligner false \
    --lr_scheduler_type warmup_stable_decay \
    --lr_scheduler_kwargs '{"num_decay_steps":1170}' \
    --warmup_steps 440 \
    --weight_decay 0.01 \
    --adam_beta2 0.98 \
    --average_tokens_across_devices true

echo "Training completed! Model saved to: $OUTPUT_DIR"
