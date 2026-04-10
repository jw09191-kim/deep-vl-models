#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

export USE_HF=1
export HF_HUB_OFFLINE=0
export FPS_MAX_FRAMES=16

export PYTHONPATH="./deep-vl-models:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

AUTO_GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

# MODEL_ID 선택:
#   Qwen3.5: Qwen/Qwen3.5-0.8B / Qwen/Qwen3.5-2B
MODEL_ID="${1:-Qwen/Qwen3.5-0.8B}"
ENCODER="${2:-vitl}"

if echo "$MODEL_ID" | grep -q "Qwen3.5"; then
    MODEL_TYPE="vora-qwen35-${ENCODER}"
else
    echo "Unknown model type in MODEL_ID: $MODEL_ID"
    exit 1
fi

MODEL_ID_CLEAN=$(basename $MODEL_ID)

OUTPUT_DIR=${OUTPUT_DIR:-"output"}
OUTPUT_DIR="$OUTPUT_DIR/${MODEL_ID_CLEAN}-${ENCODER}-align"

if [ -d "/tensorboard" ]; then
    export HF_HOME=/group-volume/.cache/huggingface
    export TORCH_HOME=/group-volume/.cache/torch
    mkdir -p "$HF_HOME" "$TORCH_HOME"
fi

echo "=========================================="
echo "  VoRA Stage 1: Alignment"
echo "=========================================="
echo "MODEL_ID  : $MODEL_ID"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "ENCODER   : $ENCODER"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "GPU_COUNT : $AUTO_GPU_COUNT"
echo "=========================================="

NPROC_PER_NODE=$AUTO_GPU_COUNT \
swift sft \
    --model "$MODEL_ID" \
    --model_type "$MODEL_TYPE" \
    --external_plugins 'my_vora_omni' \
    --dataset './datasets/LLaVA-OneVision-Data/llava_onevision.jsonl#200000' \
              './datasets/LLaVA-Video-178K/llava_video_clean.jsonl#100000' \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl "flash_attn" \
    --padding_free false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --modules_to_save "model.visual.merger" \
    --gradient_accumulation_steps 2 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 3 \
    --save_strategy "steps" \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 16 \
    --ddp_find_unused_parameters true \
    --gradient_checkpointing true \
    --lazy_tokenize true


