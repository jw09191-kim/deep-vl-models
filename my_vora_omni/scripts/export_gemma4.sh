#!/bin/bash

export USE_HF=1
export PYTHONPATH="./deep-vl-models:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [encoder] [output_name]}"
ENCODER="${2:-vitl}"
OUTPUT_NAME="${3:-}"

if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

MODEL_TYPE="vora-gemma4-${ENCODER}"
if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE=$(echo "$STAGE1_MODEL" | grep -oP '(E2B(?:-it)?|E4B(?:-it)?)' | head -1)
    MODEL_SIZE=${MODEL_SIZE:-"E4B"}
fi

BASE_MODEL="google/gemma-4-${MODEL_SIZE}"

# 자동 stage 감지 + output 이름
if [ -f "$CHECKPOINT/adapter_config.json" ]; then
    STAGE="lora"
    DEFAULT_NAME="vora-gemma4-${MODEL_SIZE}-${ENCODER}-lora-merged"
else
    STAGE="full"
    DEFAULT_NAME="vora-gemma4-${MODEL_SIZE}-${ENCODER}-merged"
fi

MERGED_PATH="output/${OUTPUT_NAME:-$DEFAULT_NAME}"

echo "=========================================="
echo "  VoRA Export (Gemma-4)"
echo "=========================================="
echo "CHECKPOINT  : $CHECKPOINT"
echo "ENCODER     : $ENCODER"
echo "MODEL_TYPE  : $MODEL_TYPE"
echo "STAGE       : $STAGE"
echo "MERGED_PATH : $MERGED_PATH"
echo "=========================================="

if [ "$STAGE" = "lora" ]; then
    echo ""
    echo "LoRA checkpoint detected -> merging..."
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
    else
        echo "LoRA merged by swift."
    fi
fi

# ── Re-save with encoder weights ───────────────────────────────
echo ""
echo "Saving full model with encoder weights..."

ENCODER="$ENCODER" CHECKPOINT="$CHECKPOINT" MERGED_PATH="$MERGED_PATH" STAGE="$STAGE" BASE_MODEL="$BASE_MODEL" \
python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import torch
from pathlib import Path
from peft import PeftModel
from my_vora_omni.src.register import *

encoder_name = os.environ["ENCODER"]
checkpoint   = os.environ["CHECKPOINT"]
merged_path  = os.environ["MERGED_PATH"]
stage        = os.environ["STAGE"]

ENCODER_MAP = {
    "vitl":     (Gemma4VJEPALModel,    Gemma4VJepa2LProcessor),
    "vitg":     (Gemma4VJEPAGModel,    Gemma4VJepa2GProcessor),
    "vjepa21b": (Gemma4VJEPA21BModel,  Gemma4VJEPA21BProcessor),
    "vjepa21l": (Gemma4VJEPA21LModel,  Gemma4VJEPA21LProcessor),
    "vjepa21g": (Gemma4VJEPA21GModel,  Gemma4VJEPA21GProcessor),
}

ModelCls, ProcCls = ENCODER_MAP[encoder_name]

base_model_id = os.environ["BASE_MODEL"]

# ── 1. Base model 로드 ────────────────────────────────────────
print(f"Loading base: {base_model_id}")
model = ModelCls.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

if stage == "lora":
    print(f"Merging LoRA from: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.merge_and_unload()
    print("LoRA merged.")

# ── 2. Save ──────────────────────────────────────────────────
# Gemma4: visual은 outer class에 직접 부착 → key prefix = "visual."
# Qwen과 달리 key 수정 불필요
model.save_pretrained(merged_path)

# config + processor 저장
model.config.save_pretrained(merged_path)

try:
    processor = ProcCls.from_pretrained(merged_path)
except Exception:
    processor = ProcCls.from_pretrained(base_model_id)
processor.save_pretrained(merged_path)

print(f"\nSaved to: {merged_path}")
PYEOF

if [ $? -ne 0 ]; then
    echo "Export failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Export complete: $MERGED_PATH"
echo ""
echo "Test with:"
echo "  bash scripts/infer_gemma4.sh $MERGED_PATH $ENCODER"
echo "=========================================="
