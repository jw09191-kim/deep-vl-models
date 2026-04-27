# Qwen3.5+VJEPA 함수 실행 순서

## 학습 (Training) — `swift sft`

```
Dataset JSONL
  │
  ▼ Dataset.map() — 샘플별 전처리
┌─────────────────────────────────────────────────────┐
│ template.encode(sample)                             │
│   └─ _encode_truncated()                           │
│       └─ _encode()  ← Template base               │
│           ├─ _swift_prepare_inputs()               │
│           │    └─ load_image() → PIL 객체로 로딩   │
│           ├─ _swift_encode()                        │
│           │    └─ 메시지 순회 → replace_tag() 호출  │
│           │         └─ ★ 우리 replace_tag('video') │
│           │              ├─ _decode_video_to_tensor │ ← mp4→[T,C,H,W] tensor
│           │              └─ do_sample_frames=False  │
│           └─ _encode_context_list() → tokenize     │
│                → input_ids (단일 248057 포함)       │
│                                                     │
│       └─ _encode()  ← Qwen3VLTemplate              │
│           ├─ processor(videos=tensors, ...)         │
│           │    ├─ Qwen3VLProcessor.__call__         │
│           │    │    └─ video_processor(tensors, ...) │
│           │    │         └─ _vjepa_preprocess_videos │ ← pixel_values_videos, video_grid_thw
│           │    └─ 타임스탬프 텍스트 생성            │
│           │         → splited_tokens[i]             │
│           └─ _extend_tokens()                       │
│                → 248057×1 → 248057×N + timestamps  │
│                                                     │
│ 반환: {input_ids, labels, pixel_values_videos,      │
│        video_grid_thw, attention_mask}              │
└─────────────────────────────────────────────────────┘
  │
  ▼ DataCollator — 배치 묶기
┌─────────────────────────────────────────────────────┐
│ _data_collator(batch)                               │
│   ├─ pad input_ids, labels, attention_mask          │
│   ├─ _data_collator_mm_data()                       │
│   │    ├─ pixel_values_videos torch.concat(batch)   │
│   │    └─ video_grid_thw torch.concat(batch)        │
│   └─ _get_position_ids()                            │
│        └─ get_rope_index(input_ids, video_grid_thw) │ ← M-RoPE 3D position_ids 계산
└─────────────────────────────────────────────────────┘
  │
  ▼ Training Step — model.forward() 직전
┌─────────────────────────────────────────────────────┐
│ pre_forward_hook()  ← Swift가 register한 hook       │
│   └─ ★ _post_encode(model, inputs)                  │
│        └─ return inputs  (no-op, VJEPA는 model에서) │
└─────────────────────────────────────────────────────┘
  │
  ▼ Qwen3_5VJEPAModel.forward()
┌─────────────────────────────────────────────────────┐
│ super().forward()  ← Qwen3_5ForConditionalGeneration│
│   ├─ embed_tokens(input_ids) → text embeddings      │
│   ├─ get_video_features(pixel_values_videos,        │
│   │    video_grid_thw)                              │
│   │    └─ ★ get_image_features() (우리 VJEPA)       │
│   │         ├─ VJEPA encoder → patch embeddings     │
│   │         ├─ tile 재조립                           │
│   │         ├─ spatial merge (2×2)                  │
│   │         └─ Merger MLP → LLM dim                 │
│   ├─ video_token 위치에 VJEPA features 주입          │
│   └─ Transformer layers → logits                   │
│                                                     │
│ Loss 계산 + Backprop                                │
└─────────────────────────────────────────────────────┘
```

---

## 추론 (Inference) — `swift infer`

```
User Prompt
  │
  ▼ template.encode(inputs, mode='transformers')
    is_training=False → labels=None (응답 부분 제거)
    ├─ replace_tag() → 학습과 동일 (mp4 디코딩)
    ├─ _encode() Qwen3VLTemplate → 학습과 동일
    │    (단, labels 없음)
    └─ 반환: {input_ids, pixel_values_videos,
               video_grid_thw, attention_mask,
               position_ids}
  │
  ▼ model.generate()
    ┌──────────────────────────────────────────────┐
    │ Prefill (첫 번째 forward, 전체 prompt)        │
    │   Qwen3_5VJEPAModel.forward()                 │
    │     ← 학습 forward와 동일하게 VJEPA 처리      │
    │     → KV cache 생성                           │
    └──────────────────────────────────────────────┘
    ┌──────────────────────────────────────────────┐
    │ Decode (이후 forward, 토큰 1개씩)             │
    │   input_ids = [새 토큰 1개]                   │
    │   pixel_values_videos = None  ← KV cache 재사용│
    │   → logits → next token sampling              │
    └──────────────────────────────────────────────┘
```

---

## 학습 vs 추론 핵심 차이

| 단계 | 학습 | 추론 |
|------|------|------|
| `labels` | 있음 (loss 계산) | 없음 |
| `_data_collator` | 호출됨 (배치 패딩) | 호출 안됨 |
| `position_ids` | collator에서 계산 | encode 결과에 포함 |
| `_post_encode` | Swift hook으로 호출 (no-op) | 호출 안됨 (또는 no-op) |
| `pixel_values_videos` | 매 step 전달 | Prefill에만 전달, decode 시 None |
| padding | right padding | left padding |

---

## 핵심 포인트

`_post_encode`가 no-op이므로, 학습/추론 모두 VJEPA 특성 주입은
`Qwen3_5VJEPAModel.forward()` 내 `Qwen3_5ForConditionalGeneration.forward()`에서 담당.

Swift 표준 Qwen3.5VL은 `_post_encode`에서 `inputs_embeds`를 미리 계산하지만,
우리는 이를 건너뛰고 model forward에서 처리한다.

```
표준 Qwen3.5VL:  _post_encode → inputs_embeds 계산 → model.forward(inputs_embeds=...)
우리 VJEPA:      _post_encode → no-op             → model.forward(pixel_values_videos=...)
                                                      → super().forward() 내부에서 VJEPA 주입
```
