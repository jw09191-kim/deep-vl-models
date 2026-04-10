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

## 우선순위 3 — 추후 개선

- [ ] **3-A. VJEPA2 마지막 2 블록 unfreeze** (Stage 2)
  - `model.py` freeze 로직 변경 + `instruction.sh` encoder LR 별도 설정

- [ ] **3-B. Caption-only 데이터 필터링**
  - LLaVA 데이터에서 "describe"/"caption" 포함 샘플만 추출 → Stage 1에서 2x 가중치
