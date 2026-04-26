#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

export USE_HF=1
export HF_HUB_OFFLINE=0
export FPS_MAX_FRAMES=16
export IMAGE_MAX_TILES=4

export PYTHONPATH="./deep-vl-models:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

AUTO_GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')

# MODEL_ID 선택:
#   LFM2: LiquidAI/LFM2.5-VL-450M
MODEL_ID="${1:-LiquidAI/LFM2.5-VL-450M}"
ENCODER="${2:-vitl}"

if echo "$MODEL_ID" | grep -qiE "LFM2|LiquidAI"; then
    MODEL_TYPE="vora-lfm2-${ENCODER}"
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
    export HF_HUB_OFFLINE=1
fi

echo "=========================================="
echo "  VoRA Stage 1: Alignment (LFM2)"
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
    --external_plugins 'my_vora_omni/src/register.py' \
    --dataset './datasets/LLaVA-OneVision-Data/llava_onevision.jsonl#50000' \
              './datasets/LLaVA-Video-178K/sources/ego4d.jsonl#10000' \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl "sdpa" \
    --padding_free false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --modules_to_save "model.visual.merger" \
    --gradient_accumulation_steps 4 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 1 \
    --save_strategy "steps" \
    --logging_steps 10 \
    --max_length 5120 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 16 \
    --ddp_find_unused_parameters true \
    --gradient_checkpointing true \
    --lazy_tokenize true
