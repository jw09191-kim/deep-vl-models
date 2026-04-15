"""
test_processor_integration.py
==============================
from_pretrained 경로 통합 테스트.

실제 사용 경로: register.py의 PROCESSOR_CLS.from_pretrained(model_dir) 와 동일.
HF Hub 접근이 필요하므로 HF_HUB_OFFLINE=1 환경에서는 conftest.py에 의해 자동 skip.

실행:
    cd /home/jinnwoo_kim/workspace/deep-vl-models
    PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/test_processor_integration.py -v -m integration
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.processor.processor import (
    Qwen3VLVJepa2LProcessor,
    Gemma4VJepa2LProcessor,
    VJEPAImageProcessor,
    VJEPAVideoProcessor,
    Gemma4VJEPAImageProcessor,
    Gemma4VJEPAVideoProcessor,
    VoRAVisionConfig,
)

# 각 클래스에 대응하는 HF Hub 모델 ID (가장 작은 것만)
QWEN_MODEL = "Qwen/Qwen3.5-0.8B"
GEMMA_MODEL = "google/gemma-4-E2B"

MERGE = VoRAVisionConfig.MERGE_SIZE


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def make_pil(h: int, w: int, fill: int = 128) -> Image.Image:
    return Image.fromarray(np.full((h, w, 3), fill, dtype=np.uint8))


def as_list(x) -> list:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)


# ---------------------------------------------------------------------------
# Fixture: 각 processor를 from_pretrained으로 로드 (테스트 클래스 단위 캐싱)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def qwen_proc():
    return Qwen3VLVJepa2LProcessor.from_pretrained(QWEN_MODEL)


@pytest.fixture(scope="class")
def gemma_proc():
    return Gemma4VJepa2LProcessor.from_pretrained(GEMMA_MODEL)


# ===========================================================================
# 1. from_pretrained 로드 + sub-processor 타입 검증
# ===========================================================================

@pytest.mark.integration
class TestFromPretrainedLoads:

    def test_qwen_sub_processor_types(self, qwen_proc):
        """Qwen3VLVJepa2LProcessor.from_pretrained → VJEPA sub-processor 타입 확인."""
        assert isinstance(qwen_proc.image_processor, VJEPAImageProcessor), \
            f"image_processor 타입 오류: {type(qwen_proc.image_processor)}"
        assert isinstance(qwen_proc.video_processor, VJEPAVideoProcessor), \
            f"video_processor 타입 오류: {type(qwen_proc.video_processor)}"

    def test_gemma_sub_processor_types(self, gemma_proc):
        """Gemma4VJepa2LProcessor.from_pretrained → Gemma4VJEPA sub-processor 타입 확인."""
        assert isinstance(gemma_proc.image_processor, Gemma4VJEPAImageProcessor), \
            f"image_processor 타입 오류: {type(gemma_proc.image_processor)}"
        assert isinstance(gemma_proc.video_processor, Gemma4VJEPAVideoProcessor), \
            f"video_processor 타입 오류: {type(gemma_proc.video_processor)}"

    def test_qwen_image_size(self, qwen_proc):
        """Qwen: from_pretrained 후 image_size가 VISION_MODEL_ID 기반 값(256)인지 확인."""
        assert qwen_proc.image_processor.image_size == 256
        assert qwen_proc.video_processor.image_size == 256

    def test_gemma_image_size(self, gemma_proc):
        """Gemma4: from_pretrained 후 image_size가 256인지 확인."""
        assert gemma_proc.image_processor.image_size == 256
        assert gemma_proc.video_processor.image_size == 256


# ===========================================================================
# 2. from_pretrained 후 이미지 처리 shape 검증
# ===========================================================================

@pytest.mark.integration
class TestFromPretrainedImageProcessing:

    def test_qwen_square_image_shape(self, qwen_proc):
        """Qwen: from_pretrained 후 256×256 이미지 → pixel_values shape."""
        out = qwen_proc.image_processor(images=[make_pil(256, 256)], return_tensors="pt")
        assert out["pixel_values"].shape == (1, 2, 3, 256, 256)
        assert out["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert as_list(out["num_soft_tokens_per_image"]) == [64]

    def test_gemma_square_image_shape(self, gemma_proc):
        """Gemma4: from_pretrained 후 256×256 이미지 → pixel_values shape."""
        out = gemma_proc.image_processor(images=[make_pil(256, 256)], return_tensors="pt")
        assert out["pixel_values"].shape == (1, 2, 3, 256, 256)
        assert out["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert as_list(out["num_soft_tokens_per_image"]) == [64]

    def test_qwen_token_count_matches_grid(self, qwen_proc):
        """Qwen: token 수 = (h_total // merge) * (w_total // merge)."""
        out = qwen_proc.image_processor(images=[make_pil(256, 512)], return_tensors="pt")
        _, h_total, w_total = out["image_grid_thw"][0].tolist()
        expected = (h_total // MERGE) * (w_total // MERGE)
        assert as_list(out["num_soft_tokens_per_image"])[0] == expected


# ===========================================================================
# 3. from_pretrained 후 비디오 처리 shape 검증
# ===========================================================================

@pytest.mark.integration
class TestFromPretrainedVideoProcessing:

    def test_qwen_pil_list_video_shape(self, qwen_proc):
        """Qwen: from_pretrained 후 PIL 리스트 비디오 → pixel_values_videos shape."""
        frames = [make_pil(256, 256) for _ in range(4)]
        out = qwen_proc.video_processor(
            videos=[frames], do_sample_frames=False, return_tensors="pt"
        )
        assert out["pixel_values_videos"].shape == (1, 4, 3, 256, 256)
        assert out["video_grid_thw"][0, 0].item() == 2  # grid_t = 4 // tubelet_size(2)
        assert as_list(out["num_soft_tokens_per_video"])[0] > 0

    def test_gemma_pil_list_video_shape(self, gemma_proc):
        """Gemma4: from_pretrained 후 PIL 리스트 비디오 → pixel_values_videos shape."""
        frames = [make_pil(256, 256) for _ in range(4)]
        out = gemma_proc.video_processor(
            videos=[frames], do_sample_frames=False, return_tensors="pt"
        )
        assert out["pixel_values_videos"].shape == (1, 4, 3, 256, 256)
        assert out["video_grid_thw"][0, 0].item() == 2
        assert as_list(out["num_soft_tokens_per_video"])[0] > 0

    def test_qwen_token_count_matches_formula(self, qwen_proc):
        """Qwen: video token 수 = grid_t * (h_total // merge) * (w_total // merge)."""
        frames = [make_pil(256, 256) for _ in range(4)]
        out = qwen_proc.video_processor(
            videos=[frames], do_sample_frames=False, return_tensors="pt"
        )
        grid_t, h_total, w_total = out["video_grid_thw"][0].tolist()
        formula = grid_t * (h_total // MERGE) * (w_total // MERGE)
        assert as_list(out["num_soft_tokens_per_video"])[0] == formula
