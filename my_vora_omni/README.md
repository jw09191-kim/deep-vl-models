# my_vora_omni

VoRA의 MS-Swift external plugin 패키지입니다. `swift sft` / `swift infer` 실행 시 `--external_plugins 'my_vora_omni'` 옵션으로 로드됩니다.

## 패키지 진입점

`__init__.py`가 로드되면 `src/register.py`가 실행되어 swift에 모델/템플릿을 등록합니다.

## 하위 구조

| 경로 | 역할 |
|---|---|
| `src/` | 모델, 프로세서, 템플릿 구현 및 swift 등록 |
| `scripts/` | 학습(3단계), 추론, 벤치마크, 익스포트 셸 스크립트 |

## 실행 위치

모든 스크립트는 **repo 루트**(`deep-vl-models/`)에서 실행해야 합니다. `PYTHONPATH`에 `./deep-vl-models`가 포함되어 있어야 합니다.
