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
VJEPA2 Encoder (frozen) → Merger (LN→Linear→GELU→Linear→GELU→LN→Linear→LN) → Qwen3.5 LLM
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

## Known Issues

### Gemma4 비디오 입력 시 채널 차원 워닝 (미해결)

**발생 조건**: Gemma4 모델 변형(`Gemma4VJEPA*`)으로 비디오 추론 시

**워닝 메시지**:
```
The channel dimension is ambiguous. Got image shape torch.Size([0, 3, 1, 1]).
Assuming channels are the first dimension.
```

**원인 분석**:
- `VJEPAVideoProcessor._preprocess()`에서 `T < tubelet_size`일 때 `grid_t = T // tubelet_size = 0`이 되어 `[0, ...]` 텐서가 생성됨
- `VJEPAVideoProcessor._preprocess()`의 T 패딩 수정(`T % tubelet_size != 0`이면 마지막 프레임 반복)으로 직접 호출 경로는 수정됨
- **그러나 워닝이 여전히 발생**: `Gemma4VJEPATemplate._post_encode()` 또는 부모 클래스 `Qwen3VLVideoProcessor.__call__()`이 프레임을 개별 이미지로 처리하는 경로에서 발생하는 것으로 추정
- `H=1, W=1`은 정상 비디오 프레임 크기가 아니므로 부모 클래스의 프레임 변환 로직이 관여하는 것으로 보임

**조사 필요 사항**:
- Qwen3.5에서는 발생하지 않고 Gemma4에서만 발생하는 것으로 확인됨
- 가설: Gemma4 비디오 프로세서 호출 시 `pixel_values`(이미지용)가 빈 값으로 함께 전달되어 HF 이미지 처리 유틸리티가 `[0, 3, 1, 1]` 형태의 빈 텐서를 처리하려다 워닝 발생 가능성
- `Qwen3VLVideoProcessor.__call__()`이 내부적으로 이미지 처리 경로를 함께 실행하는 경우 확인 필요
- `-W error::UserWarning`으로 정확한 스택 트레이스 확인 권장
