# VoRA 성능 개선 TODO

## 문제
VoRA (VJEPA2 + Qwen3.5-0.8B)가 base Qwen3.5-0.8B보다 video description 품질이 낮음.
- VoRA: "The dog is sniffing around a blue bowl filled with food" (잘못된 객체)
- Base: "A brown poodle in a kitchen, looking at a broken blue vase" (정확)

---

## 우선순위 1 — 코드/스크립트 변경 (재학습 필요)

- [x] **1-A. Post-Merger LayerNorm 추가** (`src/model/model.py` 68–71줄)
  - merger 출력 뒤 `nn.LayerNorm(out_dim)` 추가
  - visual token scale을 LLM embedding distribution에 맞춤

- [x] **1-B. Merger 3-layer로 깊게** (`src/model/model.py` 61–73줄)
  - 현재: LayerNorm → Linear → GELU → Linear
  - 변경: LayerNorm → Linear → GELU → Linear → GELU → LayerNorm → Linear → LayerNorm
  - `mid_dim = max(llm_dim, hidden_dim // 2)` 추가

- [x] **1-C. Stage 1 학습 설정 변경** (`scripts/align.sh`)
  - 데이터 균형: 이미지 200K → 150K, 영상 100K → 150K
  - LR: `1e-4` → `5e-5`
  - LR scheduler: `cosine` 추가
  - Epochs: `1` → `2`

---

## 우선순위 2 — 데이터 추가 (다운로드 필요)

- [ ] **2-A. ShareGPT4Video** (Stage 1 + Stage 2용, 40K)
  - HuggingFace: `ShareGPT4Video/ShareGPT4Video`
  - GPT-4V dense caption → object identity 학습에 효과적
  - Stage 1: merger가 semantic 정보 학습
  - Stage 2: QA 데이터만 쓰면 짧은 답변에 편향되므로 captioning 데이터로 균형 유지
  - JSONL 변환 후 `align.sh`, `instruction.sh` 둘 다 추가

- [ ] **2-B. NExT-QA** (Stage 2용)
  - 인과/시간적 reasoning QA → 현재 실패 패턴 직접 커버

- [ ] **2-C. MSVD-QA + MSRVTT-QA** (Stage 2용)
  - 기본 video QA (what/who/where) 정확도 향상

---

## 우선순위 3 — Gemma-4 백본 통합 (VJEPA2 + Gemma-4-E4B)

Gemma-4는 네이티브 SigLIP 비전을 가진 멀티모달 모델. 해당 비전 타워를 VJEPA2로 교체하는 것이 핵심.

- [ ] **3-A. Processor** (`src/processor/processor.py`)
  - `VJEPAImageProcessor._preprocess()`에 `num_soft_tokens_per_image` 필드 추가 (Gemma4Processor 호환)
  - `VJEPAVideoProcessor._preprocess()`에 `num_soft_tokens_per_video` 필드 추가
  - `Gemma4VJEPAProcessor(Gemma4Processor)` 추가: `image_processor`/`video_processor`를 VJEPA 버전으로 교체, `image_seq_length` = VJEPA 토큰 수 (vitl 기준 144)
  - Variants: `Gemma4VJepa2LProcessor`, `Gemma4VJepa2GProcessor`, `Gemma4VJEPA21B/L/GProcessor`

- [ ] **3-B. Model** (`src/model/model.py`)
  - `Gemma4VJEPAModel(Gemma4ForConditionalGeneration)` 추가
    - `get_image_features()` 오버라이드: `self.visual` (VJEPA2VisualModule) 사용, Qwen의 spatial merging 로직 그대로
    - `get_video_features()`: `get_image_features()`에 위임
    - `from_pretrained()`: Gemma4 로드 → VJEPA2 인코더 생성 → `model.visual`에 부착 (Qwen과 달리 outer class에 직접)
  - `VJEPA2VisualModule` 재사용 (변경 없음)
  - Variants: `Gemma4VJEPALModel`, `Gemma4VJEPAGModel`, `Gemma4VJEPA21B/L/GModel`
  - ⚠️ checkpoint key prefix: `"visual."` (Qwen은 `"model.visual."`)

- [ ] **3-C. Template** (`src/template/template.py`)
  - `Gemma4VJEPATemplate(Gemma4Template)` 추가
    - `_post_encode()`: VJEPA 임베딩 주입, `base_model.model.get_placeholder_mask()` 사용 (3-tuple 반환 — audio_mask 무시)
    - `_data_collator()`: Gemma4는 mrope 없음 → `super()` 위임만으로 충분
  - embed path: `base_model.model.get_input_embeddings()(input_ids)`

- [ ] **3-D. Register** (`src/register.py`)
  - `Gemma4VJEPALoaderBase(Gemma4Loader)` + 5개 variant loader
  - `register_model_arch('gemma4_vjepa')`: `vision_tower=['visual.encoder']`, `aligner=['visual.merger']`
  - `register_template('vora_gemma4')`: `Gemma4TemplateMeta` 사용
  - `register_model` × 5: `vora-gemma4-vitl/vitg/vjepa21b/l/g`, base model `google/gemma-4-E4B`

- [ ] **3-E. Scripts**
  - `scripts/align_gemma4.sh`, `scripts/instruction_gemma4.sh`, `scripts/infer_gemma4.sh` 추가
