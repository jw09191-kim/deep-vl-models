#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

export USE_HF=1
export HF_HUB_OFFLINE=0
export FPS_MAX_FRAMES=16

export PYTHONPATH="./deep-vl-models:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

AUTO_GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

STAGE1_MODEL="${1:?Usage: $0 <stage1_checkpoint_path> [encoder]}"
ENCODER="${2:-vitl}"

if [ ! -d "$STAGE1_MODEL" ]; then
    echo "Stage 1 checkpoint not found: $STAGE1_MODEL"
    exit 1
fi

MODEL_TYPE="vora-qwen35-${ENCODER}"

# Extract model size for naming
MODEL_SIZE=$(echo "$STAGE1_MODEL" | grep -oP '(0\.8B|2B)' | head -1)
MODEL_SIZE=${MODEL_SIZE:-"0.8B"}

OUTPUT_DIR=${OUTPUT_DIR:-"output"}
OUTPUT_DIR=$OUTPUT_DIR/"Qwen3.5-${MODEL_SIZE}-${ENCODER}-instruction"

if [ -d "/tensorboard" ]; then
    export HF_HOME=/group-volume/.cache/huggingface
    export TORCH_HOME=/group-volume/.cache/torch
    mkdir -p "$HF_HOME" "$TORCH_HOME"
fi

echo "=========================================="
echo "  VoRA Stage 2: Instruction Tuning"
echo "=========================================="
echo "STAGE1_MODEL: $STAGE1_MODEL"
echo "MODEL_TYPE  : $MODEL_TYPE"
echo "ENCODER     : $ENCODER"
echo "OUTPUT_DIR  : $OUTPUT_DIR"
echo "GPU_COUNT   : $AUTO_GPU_COUNT"
echo "=========================================="

NPROC_PER_NODE=$AUTO_GPU_COUNT \
swift sft \
    --model "$STAGE1_MODEL" \
    --model_type "$MODEL_TYPE" \
    --external_plugins 'my_vora_omni' \
    --system "You are a helpful assistant that can understand and analyze images and videos." \
    --dataset './datasets/LLaVA-OneVision-Data/llava_onevision.jsonl#150000' \
              './datasets/LLaVA-Video-178K/sources/youtube_video_2024.jsonl#50000' \
              './datasets/LLaVA-Video-178K/sources/Charades.jsonl' \
              './datasets/LLaVA-Video-178K/sources/activitynet.jsonl' \
              './datasets/LLaVA-Video-178K/sources/hdvila.jsonl#20000' \
              './datasets/LLaVA-Video-178K/sources/ego4d.jsonl' \
              './datasets/LLaVA-Video-178K/sources/train_val.jsonl' \
              './datasets/LLaVA-Video-178K/sources/train.jsonl' \
              './datasets/LLaVA-Video-178K/sources/videos.jsonl' \
              './datasets/LLaVA-Video-178K/sources/others.jsonl#20000' \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl "flash_attn" \
    --deepspeed zero2 \
    --padding_free false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner false \
    --modules_to_save "model.visual.merger" \
    --gradient_accumulation_steps 4 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --split_dataset_ratio 0.01 \
    --save_strategy "steps" \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 16 \
    --gradient_checkpointing true \
    --lazy_tokenize true \
    --ddp_find_unused_parameters true \
