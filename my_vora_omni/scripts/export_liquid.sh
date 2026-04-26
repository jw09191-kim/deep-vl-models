#!/bin/bash

export USE_HF=1
export PYTHONPATH="./deep-vl-models:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CHECKPOINT="${1:?Usage: $0 <lora_checkpoint_path> [encoder] [output_name]}"
ENCODER="${2:-vitl}"
OUTPUT_NAME="${3:-}"

if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

BASE_MODEL="LiquidAI/LFM2.5-VL-450M"
MODEL_TYPE="vora-lfm2-${ENCODER}"

# Export는 LoRA 병합 전용 (Stage 1 full 체크포인트는 infer에 경로만 넘기면 됨)
if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
    echo "LoRA checkpoint required: missing $CHECKPOINT/adapter_config.json"
    echo "Stage 1 (full) align outputs are already a complete model; use:"
    echo "  bash scripts/infer_liquid.sh $CHECKPOINT $ENCODER"
    exit 1
fi

DEFAULT_NAME="vora-lfm2-450M-${ENCODER}-lora-merged"

MERGED_PATH="output/${OUTPUT_NAME:-$DEFAULT_NAME}"

echo "=========================================="
echo "  VoRA Export (LFM2)"
echo "=========================================="
echo "CHECKPOINT  : $CHECKPOINT"
echo "ENCODER     : $ENCODER"
echo "MODEL_TYPE  : $MODEL_TYPE"
echo "MERGED_PATH : $MERGED_PATH"
echo "=========================================="

echo ""
echo "Merging LoRA with swift export..."
swift export \
    --model "$BASE_MODEL" \
    --adapters "$CHECKPOINT" \
    --model_type "$MODEL_TYPE" \
    --external_plugins 'my_vora_omni/src/register.py' \
    --merge_lora true \
    --output_dir "$MERGED_PATH"

if [ $? -ne 0 ]; then
    echo "swift export failed"
    exit 1
fi
echo "LoRA merged by swift."

echo ""
echo "=========================================="
echo "Export complete: $MERGED_PATH"
echo ""
echo "Test with:"
echo "  bash scripts/infer_liquid.sh $MERGED_PATH $ENCODER"
echo "=========================================="
