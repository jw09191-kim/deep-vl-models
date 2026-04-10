#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Video-MME Benchmark Runner for VoRA
#
# Usage:
#   bash src/tests/run_video_mme.sh <model_path> <model_type> <video_dir> [num_frames]
#
# Examples:
#   # Short test (first 100 samples)
#   bash src/tests/run_video_mme.sh output/Qwen3.5-0.8B-vitl-instruct vitl /data/video-mme/videos
#
#   # Full evaluation
#   bash src/tests/run_video_mme.sh output/Qwen3.5-0.8B-vitl-instruct vitl /data/video-mme/videos 16
#
#   # Use local annotation JSON instead of HuggingFace
#   DATA_PATH=/data/video-mme/videomme.json \
#   bash src/tests/run_video_mme.sh output/Qwen3.5-0.8B-vitl-instruct vitl /data/video-mme/videos
# ──────────────────────────────────────────────────────────────

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ── Arguments ──────────────────────────────────
MODEL_PATH="${1:?Usage: $0 <model_path> <model_type> <video_dir> [num_frames]}"
MODEL_TYPE="${2:?model_type required (vitl|vitg|vjepa21b|vjepa21l|vjepa21g)}"
VIDEO_DIR="${3:?video_dir required}"
NUM_FRAMES="${4:-16}"

# Optional overrides via env
DATA_PATH="${DATA_PATH:-}"         # local JSON; leave empty to load from HF
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT="${LIMIT:-}"                 # set to e.g. 100 for quick smoke-test

export CUDA_VISIBLE_DEVICES

# ── Derived paths ───────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="results/video_mme"
OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}_${MODEL_TYPE}_${NUM_FRAMES}f_${TIMESTAMP}.json"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  Video-MME Evaluation"
echo "=========================================="
echo "  MODEL_PATH : $MODEL_PATH"
echo "  MODEL_TYPE : $MODEL_TYPE"
echo "  VIDEO_DIR  : $VIDEO_DIR"
echo "  NUM_FRAMES : $NUM_FRAMES"
echo "  OUTPUT     : $OUTPUT_PATH"
echo "  GPU        : $CUDA_VISIBLE_DEVICES"
if [ -n "$LIMIT" ]; then
    echo "  LIMIT      : $LIMIT  (partial run)"
fi
echo "=========================================="

# ── Build python command ────────────────────────
CMD=(
    python -m my_vora_omni.tests.eval_video_mme
    --model_path  "$MODEL_PATH"
    --model_type  "$MODEL_TYPE"
    --video_dir   "$VIDEO_DIR"
    --output_path "$OUTPUT_PATH"
    --num_frames  "$NUM_FRAMES"
    --skip_missing
)

if [ -n "$DATA_PATH" ]; then
    CMD+=(--data_path "$DATA_PATH")
fi

if [ -n "$LIMIT" ]; then
    CMD+=(--limit "$LIMIT")
fi

# ── Run ─────────────────────────────────────────
"${CMD[@]}"

echo ""
echo "Done. Results saved to: $OUTPUT_PATH"