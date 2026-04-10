#!/bin/bash

export USE_HF=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="./deep-vl-models:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=32

CHECKPOINT="${1:?Usage: $0 <checkpoint> [encoder] [dataset]}"
ENCODER="${2:-vitl}"
VAL_DATASET="${3:-my_vora_omni/scripts/jsonl/infer.jsonl}"

MODEL_SIZE=$(echo "$CHECKPOINT" | grep -oP '(0\.8B|2B)' | head -1)
MODEL_SIZE=${MODEL_SIZE:-"0.8B"}

BASE_MODEL="Qwen/Qwen3.5-${MODEL_SIZE}"
MODEL_TYPE="vora-qwen35-${ENCODER}"

echo "=========================================="
echo "  VoRA Inference"
echo "=========================================="
echo "CHECKPOINT : $CHECKPOINT"
echo "BASE_MODEL : $BASE_MODEL"
echo "MODEL_TYPE : $MODEL_TYPE"
echo "ENCODER    : $ENCODER"
echo "DATASET    : $VAL_DATASET"

if [ -f "$CHECKPOINT/adapter_config.json" ]; then
    echo "MODE       : LoRA (Stage 2/3)"
    echo "=========================================="
    swift infer \
        --model "$BASE_MODEL" \
        --adapters "$CHECKPOINT" \
        --model_type "$MODEL_TYPE" \
        --external_plugins 'my_vora_omni' \
        --torch_dtype bfloat16 \
        --max_new_tokens 256 \
        --val_dataset "$VAL_DATASET" \
        --attn_impl sdpa \
        --enable_thinking false
else
    echo "MODE       : Full (Stage 1)"
    echo "=========================================="
    swift infer \
        --model "$CHECKPOINT" \
        --model_type "$MODEL_TYPE" \
        --external_plugins 'my_vora_omni' \
        --torch_dtype bfloat16 \
        --max_new_tokens 256 \
        --val_dataset "$VAL_DATASET" \
        --attn_impl sdpa \
        --enable_thinking false
fi