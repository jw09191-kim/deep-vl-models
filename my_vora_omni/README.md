# my_vora_omni

MS-Swift external plugin package for **VoRA** (Vision-Language Representation Alignment). Loaded via `--external_plugins 'my_vora_omni'` when running `swift sft` / `swift infer`.

When `__init__.py` is loaded, `src/register.py` runs and registers all models and templates with Swift.

---

## Package Structure

| Path | Description |
|---|---|
| `src/model/model.py` | Vision module (`VJEPA2VisualModule`) and LLM integration for each backbone |
| `src/processor/processor.py` | Image/video preprocessors (VJEPA normalization, grid metadata) |
| `src/template/template.py` | Training templates — visual embedding injection via `_post_encode` |
| `src/register.py` | Registers all model variants and templates with Swift |
| `scripts/` | Shell scripts for training (3 stages), inference, benchmark, and export |

> All scripts must be run from the **repo root** (`deep-vl-models/`) with `./deep-vl-models` on `PYTHONPATH`.

---

## Supported Backbones

### Qwen3.5 + VJEPA2

| Swift Model ID | Vision Encoder | LLM |
|---|---|---|
| `vora-qwen35-vitl` | VJEPA2 ViT-L (HuggingFace) | Qwen3.5-0.8B / 2B |
| `vora-qwen35-vitg` | VJEPA2 ViT-G (HuggingFace) | Qwen3.5-0.8B / 2B |
| `vora-qwen35-vjepa21b` | VJEPA2.1 Base (torch.hub) | Qwen3.5-0.8B / 2B |
| `vora-qwen35-vjepa21l` | VJEPA2.1 Large (torch.hub) | Qwen3.5-0.8B / 2B |
| `vora-qwen35-vjepa21g` | VJEPA2.1 Giant (torch.hub) | Qwen3.5-0.8B / 2B |

**Scripts:** `align.sh`, `instruction.sh`, `infer.sh`, `export.sh`

### Gemma-4 + VJEPA2

| Swift Model ID | Vision Encoder | LLM |
|---|---|---|
| `vora-gemma4-vitl` | VJEPA2 ViT-L (HuggingFace) | Gemma-4 E2B / E4B |
| `vora-gemma4-vitg` | VJEPA2 ViT-G (HuggingFace) | Gemma-4 E2B / E4B |
| `vora-gemma4-vjepa21b` | VJEPA2.1 Base (torch.hub) | Gemma-4 E2B / E4B |
| `vora-gemma4-vjepa21l` | VJEPA2.1 Large (torch.hub) | Gemma-4 E2B / E4B |
| `vora-gemma4-vjepa21g` | VJEPA2.1 Giant (torch.hub) | Gemma-4 E2B / E4B |

**Scripts:** `align_gemma4.sh`, `instruction_gemma4.sh`, `infer_gemma4.sh`, `export_gemma4.sh`

> Requires `transformers>=4.53`

---

## Training Pipeline

Three-stage pipeline. Scripts auto-detect GPU count; override with `CUDA_VISIBLE_DEVICES`.

### Stage 1 — Visual Alignment

Freezes the LLM and vision encoder. Trains the merger MLP only.

**Qwen3.5:**
```bash
cd my_vora_omni
bash scripts/align.sh [model_id] [encoder]
# e.g.: bash scripts/align.sh Qwen/Qwen3.5-0.8B vitl
```

**Gemma-4:**
```bash
bash scripts/align_gemma4.sh [model_id] [encoder]
# e.g.: bash scripts/align_gemma4.sh google/gemma-4-E4B vitl
#       bash scripts/align_gemma4.sh google/gemma-4-E2B-it vitl
```

### Stage 2 — Instruction Tuning

LoRA on the LLM + merger/aligner trainable.

**Qwen3.5:**
```bash
bash scripts/instruction.sh <stage1_checkpoint> [encoder]
# e.g.: bash scripts/instruction.sh ./output/Qwen3.5-0.8B-vitl-align vitl
```

**Gemma-4:**
```bash
bash scripts/instruction_gemma4.sh <stage1_checkpoint> [encoder]
# e.g.: bash scripts/instruction_gemma4.sh ./output/gemma-4-E4B-vitl-align vitl
```

### Stage 3 — Export LoRA to Full Weights

**Qwen3.5:**
```bash
bash scripts/export.sh <lora_checkpoint> [encoder]
```

**Gemma-4:**
```bash
bash scripts/export_gemma4.sh <lora_checkpoint> [encoder]
```

---

## Inference

**Qwen3.5:**
```bash
bash scripts/infer.sh <checkpoint> [encoder] [dataset_jsonl]
# e.g.: bash scripts/infer.sh ./output/Qwen3.5-0.8B-vitl-instruct vitl ./scripts/jsonl/infer.jsonl
```

**Gemma-4:**
```bash
bash scripts/infer_gemma4.sh <checkpoint> [encoder] [dataset_jsonl]
# e.g.: bash scripts/infer_gemma4.sh ./output/gemma-4-E4B-vitl-align vitl ./scripts/jsonl/infer.jsonl
```

Accepts both LoRA (Stage 2) and full model (Stage 1/3) checkpoints.

---

## Architecture

```
VJEPA2 Encoder (frozen) → Merger (LayerNorm→Linear→GELU→Linear→GELU→LayerNorm→Linear→LayerNorm) → LLM
```

The merger is the only component trained in Stage 1. Stage 2 adds LoRA adapters to the LLM.

### Backbone Differences

| | Qwen3.5 | Gemma-4 |
|---|---|---|
| `visual` location | `model.model.visual` (inner model) | `model.visual` (outer model) |
| Checkpoint key prefix | `model.visual.*` | `visual.*` |
| Position encoding | mRoPE (multi-dimensional) | Standard RoPE |
| Template `_data_collator` | mRoPE position_ids splitting | Delegates to parent |
| `modules_to_save` | `model.visual.merger` | `visual.merger` |

### Encoder Options

| Encoder flag | Model ID | Image size | Tokens/image |
|---|---|---|---|
| `vitl` | `facebook/vjepa2-vitl-fpc64-256` | 256×256 | 64 |
| `vitg` | `facebook/vjepa2-vitg-fpc64-256` | 256×256 | 64 |
| `vjepa21b` | `vjepa2_1_vit_base_384` (torch.hub) | 384×384 | 144 |
| `vjepa21l` | `vjepa2_1_vit_large_384` (torch.hub) | 384×384 | 144 |
| `vjepa21g` | `vjepa2_1_vit_giant_384` (torch.hub) | 384×384 | 144 |

---

## Data Format (JSONL)

```json
{"messages": [{"role": "user", "content": "<image>\nDescribe this."}], "images": ["path/to/img.jpg"]}
{"messages": [{"role": "user", "content": "<video>\nWhat happens?"}], "videos": ["path/to/video.mp4"]}
```

---

## Testing

모든 테스트는 repo root(`deep-vl-models/`)에서 실행한다.

### 단위 테스트 (오프라인 가능)

전처리 로직(shape, token 수, padding 등)을 HF Hub 없이 검증한다.

```bash
PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/test_processor.py -v
```

### 통합 테스트 (HF Hub 필요)

`from_pretrained` 경로를 통해 실제 사용 경로를 검증한다.
`HF_HUB_OFFLINE=1` 환경에서는 자동으로 skip된다.

```bash
PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/test_processor_integration.py -v -m integration
```

### 전체 실행

```bash
# 온라인 (integration 포함)
PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/ -v

# 오프라인 (integration 자동 skip)
HF_HUB_OFFLINE=1 PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/ -v
```

### 테스트 파일 구조

| 파일 | 종류 | 내용 |
|---|---|---|
| `tests/test_processor.py` | 단위 | 전처리 shape/token/padding 검증, 직접 인스턴스화 |
| `tests/test_processor_integration.py` | 통합 (`@pytest.mark.integration`) | `from_pretrained` 경로, sub-processor 타입, 처리 결과 검증 |

---

## Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` (infer) / `0,1,2,3` (train) | GPU selection |
| `FPS_MAX_FRAMES` | 16–32 | Max video frames |
| `VIDEO_MAX_PIXELS` | 50176 | Max pixels per frame |
| `HF_HUB_OFFLINE` | `0` | Set to `1` for air-gapped environments |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | GPU memory management |
