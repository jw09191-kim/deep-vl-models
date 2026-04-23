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

MODEL_TYPE="vora-qwen35-${ENCODER}"

if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE=$(echo "$STAGE1_MODEL" | grep -oP '(0\.8B|2B)' | head -1)
    MODEL_SIZE=${MODEL_SIZE:-"0.8B"}
fi

BASE_MODEL="Qwen/Qwen3.5-${MODEL_SIZE}"

# 자동 stage 감지 + output 이름
if [ -f "$CHECKPOINT/adapter_config.json" ]; then
    STAGE="lora"
    DEFAULT_NAME="vora-qwen35-${MODEL_SIZE}-${ENCODER}-lora-merged"
else
    STAGE="full"
    DEFAULT_NAME="vora-qwen35-${MODEL_SIZE}-${ENCODER}-merged"
fi

MERGED_PATH="output/${OUTPUT_NAME:-$DEFAULT_NAME}"

echo "=========================================="
echo "  VoRA Export"
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
        echo "swift export failed, trying manual merge..."
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

import json
import torch
from pathlib import Path
from peft import PeftModel
from my_vora_omni.src.register import *

encoder_name = os.environ["ENCODER"]
checkpoint   = os.environ["CHECKPOINT"]
merged_path  = os.environ["MERGED_PATH"]
stage        = os.environ["STAGE"]

ENCODER_MAP = {
    "vitl":     (Qwen3_5VJEPALModel,    Qwen3VLVJepa2LProcessor),
    "vitg":     (Qwen3_5VJEPAGModel,    Qwen3VLVJepa2GProcessor),
    "vjepa21b": (Qwen3_5VJEPA21BModel,  Qwen3VLVJepa21BProcessor),
    "vjepa21l": (Qwen3_5VJEPA21LModel,  Qwen3VLVJepa21LProcessor),
    "vjepa21g": (Qwen3_5VJEPA21GModel,  Qwen3VLVJepa21GProcessor),
}

ModelCls, ProcCls = ENCODER_MAP[encoder_name]

# ── Base model 찾기 ───────────────────────────────────────────
args_file = Path(checkpoint) / "args.json"
if not args_file.exists():
    for parent in Path(checkpoint).parents:
        if (parent / "args.json").exists():
            args_file = parent / "args.json"
            break

base_model_id = os.environ["BASE_MODEL"]

# ── 1. Base model 로드 (Stage 1 checkpoint = trained merger) ──
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

# ── Save ─────────────────────────────────────────────────────
model.save_pretrained(merged_path)

# language_model.visual.* → visual.* key 수정
from safetensors.torch import load_file, save_file
st = load_file(os.path.join(merged_path, "model.safetensors"))
fixed = {}
for k, v in st.items():
    if 'language_model.visual.' in k:
        new_k = k.replace('language_model.visual.', 'visual.')
        fixed[new_k] = v
        print(f"  Key fixed: {k} → {new_k}")
    else:
        fixed[k] = v
save_file(fixed, os.path.join(merged_path, "model.safetensors"))

# config + processor 저장
model.config.save_pretrained(merged_path)

try:
    processor = ProcCls.from_pretrained(merged_path)
except:
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
echo "  bash infer.sh $MERGED_PATH $ENCODER"
echo "=========================================="
