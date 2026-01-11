#! /bin/bash

MODEL_NAME=$1
# get the model id from the model name
MODEL_ID=$(echo $MODEL_NAME | cut -d '/' -f 2)

NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
# get device id from 0 to NPROC_PER_NODE - 1 with comma separated
DEVICE_IDS=$(seq 0 $((NPROC_PER_NODE - 1)) | tr '\n' ',' | sed 's/,$//')
DEVICE_IDS=3
GLOBAL_BATCH_SIZE=128
BATCH_SIZE=32
# set grad acc as global batch size / batch size
GRAD_ACC_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_SIZE * NPROC_PER_NODE)))
# check if divisible
if [ $((GLOBAL_BATCH_SIZE % (BATCH_SIZE * NPROC_PER_NODE))) -ne 0 ]; then
    echo "GLOBAL_BATCH_SIZE is not divisible by (BATCH_SIZE * NPROC_PER_NODE), global batch size: $GLOBAL_BATCH_SIZE, batch size: $BATCH_SIZE, nproc per node: $NPROC_PER_NODE"
    echo "exiting..."
    exit 1
fi

TRAIN_DATASET_PATH=data_v2/sft/${2}_train.jsonl
PREFIX=$3


echo "Model: $MODEL_NAME"
echo "Model ID: $MODEL_ID"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "DEVICE_IDS: $DEVICE_IDS"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACC_STEPS: $GRAD_ACC_STEPS"
echo "PREFIX: $PREFIX"
echo "Train dataset path: $TRAIN_DATASET_PATH"

# 35GiB
WANDB_ENTITY=collab-srd \
WANDB_PROJECT=GanitLLM-train \
WANDB_TAGS=sft_v4,full,${1},${2},${3} \
NPROC_PER_NODE=$NPROC_PER_NODE \
CUDA_VISIBLE_DEVICES=$DEVICE_IDS \
swift sft \
    --use_hf \
    --model_type qwen3 \
    --dataset $TRAIN_DATASET_PATH \
    --model $MODEL_NAME \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 50 \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-7}' \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --max_grad_norm 1.0 \
    --save_strategy epoch \
    --eval_strategy no \
    --save_steps 1 \
    --logging_steps 1 \
    --max_length 4096 \
    --packing true \
    --deepspeed zero1 \
    --use_liger_kernel true \
    --attn_impl flash_attention_2 \
    --run_name ${PREFIX}_v4_sft_full_${MODEL_ID} \
    --create_checkpoint_symlink true \
    --load_from_cache_file false \
    --logging_first_step true \
    --output_dir /umbc/rs/pi_ferraro/users/sroydip1/GanitLLM_checkpoints/sft_v4/${PREFIX}_full_${MODEL_ID} \
    --report_to wandb


