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
#   Gemma4: google/gemma-4-E2B / google/gemma-4-E2B-it
#           google/gemma-4-E4B / google/gemma-4-E4B-it
MODEL_ID="${1:-google/gemma-4-E4B}"
ENCODER="${2:-vitl}"

if echo "$MODEL_ID" | grep -q "gemma-4"; then
    MODEL_TYPE="vora-gemma4-${ENCODER}"
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
echo "  VoRA Stage 1: Alignment (Gemma-4)"
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
    --dataset './datasets/LLaVA-OneVision-Data/llava_onevision.jsonl#150000' \
              './datasets/LLaVA-Video-178K/llava_video_clean.jsonl#150000' \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl "flash_attn" \
    --padding_free false \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --modules_to_save "visual.merger" \
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
