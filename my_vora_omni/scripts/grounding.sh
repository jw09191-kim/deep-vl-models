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

STAGE2_MODEL="${1:?Usage: $0 <stage2_checkpoint_path> [encoder]}"
ENCODER="${2:-vitl}"

if [ ! -d "$STAGE2_MODEL" ]; then
    echo "Stage 2 checkpoint not found: $STAGE2_MODEL"
    exit 1
fi

# Extract model size for naming
MODEL_SIZE=$(echo "$STAGE2_MODEL" | grep -oP '(0\.8B|2B)' | head -1)
MODEL_SIZE=${MODEL_SIZE:-"0.8B"}

BASE_MODEL="Qwen/Qwen3.5-${MODEL_SIZE}"
MODEL_TYPE="vora-qwen35-${ENCODER}"

OUTPUT_DIR=${OUTPUT_DIR:-"output"}
OUTPUT_DIR=$OUTPUT_DIR/"Qwen3.5-${MODEL_SIZE}-${ENCODER}-grounding"

if [ -d "/tensorboard" ]; then
    export HF_HOME=/group-volume/.cache/huggingface
    export TORCH_HOME=/group-volume/.cache/torch
    mkdir -p "$HF_HOME" "$TORCH_HOME"
fi

echo "=========================================="
echo "  VoRA Stage 3: Grounding Tuning"
echo "=========================================="
echo "STAGE2_MODEL: $STAGE2_MODEL"
echo "BASE_MODEL  : $BASE_MODEL"
echo "MODEL_TYPE  : $MODEL_TYPE"
echo "ENCODER     : $ENCODER"
echo "OUTPUT_DIR  : $OUTPUT_DIR"
echo "GPU_COUNT   : $AUTO_GPU_COUNT"
echo "=========================================="

NPROC_PER_NODE=$AUTO_GPU_COUNT \
swift sft \
    --model "$BASE_MODEL" \
    --adapters "$STAGE2_MODEL" \
    --model_type "$MODEL_TYPE" \
    --external_plugins 'my_vora_omni' \
    --dataset './datasets/LLaVA-OneVision-Data/llava_onevision.jsonl#100000' \
              './datasets/LLaVA-Video-178K/llava_video_clean.jsonl#100000' \
              './datasets/LLaVA-Video-178K/grounding_all.jsonl' \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl "flash_attn" \
    --deepspeed zero2 \
    --padding_free false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
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
    --lazy_tokenize true
