# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VoRA** (Vision-Language Representation Alignment) — a framework for training Qwen3.5 language models with VJEPA2 (Vision Joint Embedding Predictive Architecture) vision encoders. The project is a plugin for the [MS-Swift](https://github.com/modelscope/ms-swift) training framework (`swift` CLI).

All source lives under `my_vora_omni/`. Scripts are run from the repo root. Swift must be installed and `my_vora_omni` must be on `PYTHONPATH`.

## Commands

### Training

Three-stage pipeline. Scripts auto-detect GPU count; override with `CUDA_VISIBLE_DEVICES`.

**Stage 1 — Visual Alignment** (freezes encoder and LLM, trains merger only):
```bash
cd my_vora_omni
bash scripts/align.sh [model_id] [encoder]
# e.g.: bash scripts/align.sh Qwen/Qwen3.5-0.8B vitl
```

**Stage 2 — Instruction Tuning** (LoRA on LLM + merger/aligner trainable):
```bash
bash scripts/instruction.sh <stage1_checkpoint> [encoder]
# e.g.: bash scripts/instruction.sh ./output/Qwen3.5-0.8B-vitl-align vitl
```

**Stage 3 — Export LoRA to full weights**:
```bash
bash scripts/export.sh <lora_checkpoint> [encoder]
```

Encoder options: `vitl`, `vitg`, `vjepa21b`, `vjepa21l`, `vjepa21g`

### Inference
```bash
bash scripts/infer.sh <checkpoint> [encoder] [dataset_jsonl]
# e.g.: bash scripts/infer.sh ./output/Qwen3.5-0.8B-vitl-instruct vitl ./scripts/jsonl/infer.jsonl
```
Accepts both LoRA (Stage 2) and full model (Stage 1/3) checkpoints.

### Benchmark (Video-MME)
```bash
bash scripts/run_video_mme.sh <model_path> <model_type> <video_dir> [num_frames]
```

### Underlying CLI
All scripts wrap `swift sft` / `swift infer` with `--external_plugins 'my_vora_omni'`. You can call swift directly with that flag to use the registered models/templates.

## Architecture

### Training Pipeline
```
VJEPA2 Encoder (frozen) → Merger (LayerNorm→Linear→GELU→Linear) → Qwen3.5 LLM
```
The merger is the only component trained in Stage 1. Stage 2 adds LoRA adapters to the LLM.

### Key Source Files

| File | Purpose |
|---|---|
| `src/model/model.py` | `VJEPA2VisualModule` wraps encoder + merger. `Qwen3_5VJEPAInnerModel` extends Qwen3.5 to accept visual features. One class per encoder variant (L/G/2.1B/L/G). |
| `src/processor/processor.py` | `VJEPAImageProcessor` (384×384, normalized), `VJEPAVideoProcessor` (frame-based), `Qwen3VLVJEPAProcessor` combines both. One processor class per encoder variant. |
| `src/template/template.py` | `Qwen3_5VJEPATemplate` handles prompt formatting. `_post_encode` injects visual embeddings into the token stream and computes mrope position IDs. |
| `src/register.py` | Registers 5 model variants and the `vora_qwen35` template with swift. Custom loaders extend `Qwen3_5Loader`. |
| `src/example.py` | Reference implementation using Qwen2.5-Omni (not part of VoRA pipeline). |

### Model Variants Registered with Swift

| Swift Model ID | Encoder |
|---|---|
| `vora-qwen35-vitl` | VJEPA2 ViT-L (HuggingFace) |
| `vora-qwen35-vitg` | VJEPA2 ViT-G (HuggingFace) |
| `vora-qwen35-vjepa21b` | VJEPA2.1 Base (torch.hub) |
| `vora-qwen35-vjepa21l` | VJEPA2.1 Large (torch.hub) |
| `vora-qwen35-vjepa21g` | VJEPA2.1 Giant (torch.hub) |

### Data Format (JSONL)

Input files for training/inference follow this schema:
```json
{"messages": [{"role": "user", "content": "<image>\nDescribe this."}], "images": ["path/to/img.jpg"]}
{"messages": [{"role": "user", "content": "<video>\nWhat happens?"}], "videos": ["path/to/video.mp4"]}
```

### Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` (infer) / `0,1,2,3` (train) | GPU selection |
| `FPS_MAX_FRAMES` | 16–32 | Max video frames |
| `VIDEO_MAX_PIXELS` | 50176 | Max pixels per frame |
| `HF_HUB_OFFLINE` | `0` | Set to `1` for air-gapped environments |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | GPU memory management |

Training uses DeepSpeed ZeRO-2 and Flash-Attention. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are set to `1` in scripts.
