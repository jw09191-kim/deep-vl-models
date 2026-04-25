#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Video-MME Benchmark Runner for VoRA
#
# Run from anywhere; the script cds to the repo root (parent of my_vora_omni).
#
# Usage:
#   bash my_vora_omni/scripts/run_video_mme.sh <model_path> <model_type> <video_dir> [num_frames]
#
# Examples (cwd = repository root that contains my_vora_omni/):
#   LIMIT=100 bash my_vora_omni/scripts/run_video_mme.sh \
#       output/Qwen3.5-0.8B-vitl-instruct vitl /data/video-mme/videos
#
#   bash my_vora_omni/scripts/run_video_mme.sh \
#       output/gemma-4-E4B-vitl-instruct vitl /data/video-mme/videos 16
#
#   DATA_PATH=/data/video-mme/videomme.json \
#   bash my_vora_omni/scripts/run_video_mme.sh \
#       output/Qwen3.5-0.8B-vitl-instruct vitl /data/video-mme/videos
#
# Optional env (passed to eval_video_mme.py when set):
#   DATA_PATH, LIMIT, MAX_TILES, MAX_NEW_TOKENS, DEVICE, CUDA_VISIBLE_DEVICES
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# my_vora_omni/scripts -> repo root (directory that contains my_vora_omni/)
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ── Arguments ──────────────────────────────────
MODEL_PATH="${1:?Usage: $0 <model_path> <model_type> <video_dir> [num_frames]}"
MODEL_TYPE="${2:?model_type required (vitl|vitg|vjepa21b|vjepa21l|vjepa21g)}"
VIDEO_DIR="${3:?video_dir required}"
NUM_FRAMES="${4:-16}"

# Optional overrides via env
DATA_PATH="${DATA_PATH:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT="${LIMIT:-}"
MAX_TILES="${MAX_TILES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
DEVICE="${DEVICE:-}"

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
echo "  REPO_ROOT  : $REPO_ROOT"
echo "  MODEL_PATH : $MODEL_PATH"
echo "  MODEL_TYPE : $MODEL_TYPE"
echo "  VIDEO_DIR  : $VIDEO_DIR"
echo "  NUM_FRAMES : $NUM_FRAMES"
echo "  OUTPUT     : $OUTPUT_PATH"
echo "  GPU        : $CUDA_VISIBLE_DEVICES"
if [ -n "${DEVICE}" ]; then
    echo "  DEVICE     : $DEVICE"
fi
if [ -n "$LIMIT" ]; then
    echo "  LIMIT      : $LIMIT  (partial run)"
fi
if [ -n "$MAX_TILES" ]; then
    echo "  MAX_TILES  : $MAX_TILES"
fi
if [ -n "$MAX_NEW_TOKENS" ]; then
    echo "  MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
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

if [ -n "$MAX_TILES" ]; then
    CMD+=(--max_tiles "$MAX_TILES")
fi

if [ -n "$MAX_NEW_TOKENS" ]; then
    CMD+=(--max_new_tokens "$MAX_NEW_TOKENS")
fi

if [ -n "$DEVICE" ]; then
    CMD+=(--device "$DEVICE")
fi

# ── Run ─────────────────────────────────────────
"${CMD[@]}"

echo ""
echo "Done. Results saved to: $OUTPUT_PATH"
