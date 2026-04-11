"""
test_processor.py
=================
Qwen3VLVJEPAProcessor / Gemma4VJEPAProcessor 래퍼 클래스 검증.

래퍼 클래스의 역할:
  - VISION_MODEL_ID 로 VJEPAImageProcessor / VJEPAVideoProcessor 를 생성
  - Gemma4 는 추가로 image_seq_length 를 계산

각 래퍼 클래스의 VISION_MODEL_ID 를 가져와 서브 프로세서를 직접 생성하고
이미지/비디오 처리 결과의 shape·토큰 수를 검증한다.
(tokenizer 없이 실행 가능)

실행:
    cd /home/jinnwoo_kim/workspace/deep-vl-models
    PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/test_processor.py -v
"""

import pytest
import torch
import numpy as np
from PIL import Image
from transformers.image_utils import SizeDict

from src.processor.processor import (
    # Qwen3 래퍼 클래스 (VISION_MODEL_ID 참조용)
    Qwen3VLVJepa2LProcessor,
    Qwen3VLVJepa2GProcessor,
    Qwen3VLVJepa21BProcessor,
    Qwen3VLVJepa21LProcessor,
    Qwen3VLVJepa21GProcessor,
    # Gemma4 래퍼 클래스 (VISION_MODEL_ID + image_seq_length 참조용)
    Gemma4VJepa2LProcessor,
    Gemma4VJepa2GProcessor,
    Gemma4VJEPA21BProcessor,
    Gemma4VJEPA21LProcessor,
    Gemma4VJEPA21GProcessor,
    # 실제 서브 프로세서
    VJEPAImageProcessor,
    VJEPAVideoProcessor,
    VoRAVisionConfig,
)

MERGE = VoRAVisionConfig.MERGE_SIZE  # 2


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def as_list(x) -> list:
    """tensor / list 모두 Python list 로 변환. return_tensors='pt' 시 tensor로 반환되는 필드에 사용."""
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)


def make_pil(h: int, w: int, fill: int = 128) -> Image.Image:
    return Image.fromarray(np.full((h, w, 3), fill, dtype=np.uint8))


def make_video_tensor(T: int, h: int, w: int) -> torch.Tensor:
    """[T, C=3, H, W] float32, 값 범위 [0, 255]"""
    return torch.full((T, 3, h, w), 128.0, dtype=torch.float32)


def img_proc_for(cls) -> VJEPAImageProcessor:
    return VJEPAImageProcessor(cls.VISION_MODEL_ID)


def vid_proc_for(cls) -> VJEPAVideoProcessor:
    return VJEPAVideoProcessor(cls.VISION_MODEL_ID)


def run_video(proc: VJEPAVideoProcessor, vid: torch.Tensor):
    return proc._preprocess(
        videos=[vid],
        do_resize=True,
        size=SizeDict(
            longest_edge=proc.image_size**2, shortest_edge=proc.image_size**2
        ),
        image_mean=VoRAVisionConfig.MEAN,
        image_std=VoRAVisionConfig.STD,
        do_rescale=True,
        rescale_factor=1 / 255.0,
    )


# ===========================================================================
# 1. VISION_MODEL_ID 매핑 검증
#    각 래퍼 클래스가 올바른 VISION_MODEL_ID 를 선언하고,
#    그 ID 로 만든 서브 프로세서의 image_size 가 맞는지 확인
# ===========================================================================


class TestVisionModelIdMapping:

    @pytest.mark.parametrize(
        "cls, expected_image_size",
        [
            (Qwen3VLVJepa2LProcessor, 256),
            (Qwen3VLVJepa2GProcessor, 256),
            (Qwen3VLVJepa21BProcessor, 384),
            (Qwen3VLVJepa21LProcessor, 384),
            (Qwen3VLVJepa21GProcessor, 384),
        ],
    )
    def test_qwen_image_size(self, cls, expected_image_size):
        img_proc = img_proc_for(cls)
        vid_proc = vid_proc_for(cls)
        assert (
            img_proc.image_size == expected_image_size
        ), f"{cls.__name__}: image_processor.image_size={img_proc.image_size}"
        assert (
            vid_proc.image_size == expected_image_size
        ), f"{cls.__name__}: video_processor.image_size={vid_proc.image_size}"

    @pytest.mark.parametrize(
        "cls, expected_image_size",
        [
            (Gemma4VJepa2LProcessor, 256),
            (Gemma4VJepa2GProcessor, 256),
            (Gemma4VJEPA21BProcessor, 384),
            (Gemma4VJEPA21LProcessor, 384),
            (Gemma4VJEPA21GProcessor, 384),
        ],
    )
    def test_gemma4_image_size(self, cls, expected_image_size):
        img_proc = img_proc_for(cls)
        vid_proc = vid_proc_for(cls)
        assert img_proc.image_size == expected_image_size
        assert vid_proc.image_size == expected_image_size

    @pytest.mark.parametrize(
        "cls",
        [
            Qwen3VLVJepa2LProcessor,
            Qwen3VLVJepa2GProcessor,
            Qwen3VLVJepa21BProcessor,
            Qwen3VLVJepa21LProcessor,
            Qwen3VLVJepa21GProcessor,
            Gemma4VJepa2LProcessor,
            Gemma4VJepa2GProcessor,
            Gemma4VJEPA21BProcessor,
            Gemma4VJEPA21LProcessor,
            Gemma4VJEPA21GProcessor,
        ],
    )
    def test_tubelet_merge_size(self, cls):
        img_proc = img_proc_for(cls)
        vid_proc = vid_proc_for(cls)
        assert img_proc.tubelet_size == 2
        assert img_proc.merge_size == MERGE
        assert vid_proc.tubelet_size == 2
        assert vid_proc.merge_size == MERGE


# ===========================================================================
# 2. Gemma4 image_seq_length 계산 검증
#    image_seq_length = (image_size // patch_size // merge_size) ** 2
# ===========================================================================


class TestGemma4ImageSeqLength:

    @pytest.mark.parametrize(
        "cls, expected_seq_len",
        [
            # (256 // 16 // 2)^2 = 8^2 = 64
            (Gemma4VJepa2LProcessor, 64),
            (Gemma4VJepa2GProcessor, 64),
            # (384 // 16 // 2)^2 = 12^2 = 144
            (Gemma4VJEPA21BProcessor, 144),
            (Gemma4VJEPA21LProcessor, 144),
            (Gemma4VJEPA21GProcessor, 144),
        ],
    )
    def test_image_seq_length_formula(self, cls, expected_seq_len):
        cfg = VoRAVisionConfig(cls.VISION_MODEL_ID)
        computed = (cfg.image_size // cfg.patch_size // cfg.MERGE_SIZE) ** 2
        assert (
            computed == expected_seq_len
        ), f"{cls.__name__}: computed={computed}, expected={expected_seq_len}"

    @pytest.mark.parametrize(
        "cls, expected_seq_len",
        [
            (Gemma4VJepa2LProcessor, 64),
            (Gemma4VJepa2GProcessor, 64),
            (Gemma4VJEPA21BProcessor, 144),
            (Gemma4VJEPA21LProcessor, 144),
            (Gemma4VJEPA21GProcessor, 144),
        ],
    )
    def test_seq_len_equals_single_tile_tokens(self, cls, expected_seq_len):
        """single-tile 이미지의 소프트 토큰 수 == image_seq_length"""
        img_proc = img_proc_for(cls)
        sz = img_proc.image_size
        out = img_proc(images=[make_pil(sz, sz)], return_tensors="pt")
        tokens = as_list(out["num_soft_tokens_per_image"])[0]
        assert (
            tokens == expected_seq_len
        ), f"{cls.__name__}: single-tile tokens={tokens}, image_seq_length={expected_seq_len}"


# ===========================================================================
# 3. image_processor — PIL 이미지 처리 shape / token 수 검증
# ===========================================================================


class TestImageProcessorOutput:

    # ── Qwen3 ViT-L / ViT-G (256 px) ──────────────────────────────────────

    @pytest.mark.parametrize(
        "cls",
        [
            Qwen3VLVJepa2LProcessor,
            Qwen3VLVJepa2GProcessor,
            Gemma4VJepa2LProcessor,
            Gemma4VJepa2GProcessor,
        ],
    )
    def test_256px_square(self, cls):
        out = img_proc_for(cls)(images=[make_pil(256, 256)], return_tensors="pt")
        assert out["pixel_values"].shape == (1, 2, 3, 256, 256)
        assert out["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert out["num_soft_tokens_per_image"] == [64]

    @pytest.mark.parametrize("cls", [Qwen3VLVJepa2LProcessor, Gemma4VJepa2LProcessor])
    def test_256px_wide_2x(self, cls):
        # 256×512 → (1,2) 타일 → 2 tiles
        out = img_proc_for(cls)(images=[make_pil(256, 512)], return_tensors="pt")
        assert out["pixel_values"].shape == (2, 2, 3, 256, 256)
        assert out["image_grid_thw"].tolist() == [[1, 16, 32]]
        assert out["num_soft_tokens_per_image"] == [128]  # 8 * 16

    @pytest.mark.parametrize("cls", [Qwen3VLVJepa2LProcessor, Gemma4VJepa2LProcessor])
    def test_256px_tall_2x(self, cls):
        # 512×256 → (2,1) 타일 → 2 tiles
        out = img_proc_for(cls)(images=[make_pil(512, 256)], return_tensors="pt")
        assert out["pixel_values"].shape == (2, 2, 3, 256, 256)
        assert out["image_grid_thw"].tolist() == [[1, 32, 16]]
        assert out["num_soft_tokens_per_image"] == [128]

    # ── Qwen3 / Gemma4 VJEPA2.1 (384 px) ──────────────────────────────────

    @pytest.mark.parametrize(
        "cls",
        [
            Qwen3VLVJepa21BProcessor,
            Qwen3VLVJepa21LProcessor,
            Qwen3VLVJepa21GProcessor,
            Gemma4VJEPA21BProcessor,
            Gemma4VJEPA21LProcessor,
            Gemma4VJEPA21GProcessor,
        ],
    )
    def test_384px_square(self, cls):
        out = img_proc_for(cls)(images=[make_pil(384, 384)], return_tensors="pt")
        assert out["pixel_values"].shape == (1, 2, 3, 384, 384)
        assert out["image_grid_thw"].tolist() == [[1, 24, 24]]
        assert out["num_soft_tokens_per_image"] == [144]

    @pytest.mark.parametrize("cls", [Qwen3VLVJepa21BProcessor, Gemma4VJEPA21BProcessor])
    def test_384px_wide_2x(self, cls):
        out = img_proc_for(cls)(images=[make_pil(384, 768)], return_tensors="pt")
        assert out["pixel_values"].shape == (2, 2, 3, 384, 384)
        assert out["image_grid_thw"].tolist() == [[1, 24, 48]]
        assert out["num_soft_tokens_per_image"] == [288]  # 12 * 24

    # ── token count == grid 공식 일관성 ────────────────────────────────────

    @pytest.mark.parametrize(
        "cls, h, w",
        [
            (Qwen3VLVJepa2LProcessor, 256, 256),
            (Qwen3VLVJepa2LProcessor, 256, 512),
            (Qwen3VLVJepa2LProcessor, 512, 256),
            (Qwen3VLVJepa21BProcessor, 384, 384),
            (Qwen3VLVJepa21BProcessor, 384, 768),
            (Gemma4VJepa2LProcessor, 256, 256),
            (Gemma4VJEPA21BProcessor, 384, 384),
            (Gemma4VJEPA21BProcessor, 384, 768),
        ],
    )
    def test_token_count_matches_grid(self, cls, h, w):
        out = img_proc_for(cls)(images=[make_pil(h, w)], return_tensors="pt")
        _, h_total, w_total = out["image_grid_thw"][0].tolist()
        expected = (h_total // MERGE) * (w_total // MERGE)
        assert (
            out["num_soft_tokens_per_image"][0] == expected
        ), f"{cls.__name__} h={h},w={w}: tokens={out['num_soft_tokens_per_image'][0]}, formula={expected}"

    # ── tubelet 차원 = 동일 프레임 반복 ────────────────────────────────────

    @pytest.mark.parametrize(
        "cls",
        [
            Qwen3VLVJepa2LProcessor,
            Qwen3VLVJepa21BProcessor,
            Gemma4VJepa2LProcessor,
            Gemma4VJEPA21BProcessor,
        ],
    )
    def test_tubelet_frames_are_identical(self, cls):
        sz = img_proc_for(cls).image_size
        out = img_proc_for(cls)(images=[make_pil(sz, sz)], return_tensors="pt")
        pv = out["pixel_values"]  # [1, 2, 3, sz, sz]
        assert torch.allclose(
            pv[0, 0], pv[0, 1]
        ), f"{cls.__name__}: tubelet 차원의 두 프레임이 다릅니다"


# ===========================================================================
# 4. video_processor — 비디오 텐서 처리 shape / token 수 검증
# ===========================================================================


class TestVideoProcessorOutput:

    # ── Qwen3 ViT-L / Gemma4 ViT-L (256 px) ───────────────────────────────

    @pytest.mark.parametrize(
        "cls",
        [
            Qwen3VLVJepa2LProcessor,
            Qwen3VLVJepa2GProcessor,
            Gemma4VJepa2LProcessor,
            Gemma4VJepa2GProcessor,
        ],
    )
    def test_256px_4frames_square(self, cls):
        out = run_video(vid_proc_for(cls), make_video_tensor(4, 256, 256))
        assert out["pixel_values_videos"].shape == (1, 4, 3, 256, 256)
        assert out["video_grid_thw"].tolist() == [[2, 16, 16]]
        assert out["num_soft_tokens_per_video"] == [128]  # 2*8*8

    @pytest.mark.parametrize("cls", [Qwen3VLVJepa2LProcessor, Gemma4VJepa2LProcessor])
    def test_256px_wide_video(self, cls):
        out = run_video(vid_proc_for(cls), make_video_tensor(4, 256, 512))
        assert out["pixel_values_videos"].shape == (2, 4, 3, 256, 256)
        assert out["video_grid_thw"].tolist() == [[2, 16, 32]]
        assert out["num_soft_tokens_per_video"] == [256]  # 2*8*16

    # ── Qwen3 / Gemma4 VJEPA2.1 (384 px) ──────────────────────────────────

    @pytest.mark.parametrize(
        "cls",
        [
            Qwen3VLVJepa21BProcessor,
            Qwen3VLVJepa21LProcessor,
            Qwen3VLVJepa21GProcessor,
            Gemma4VJEPA21BProcessor,
            Gemma4VJEPA21LProcessor,
            Gemma4VJEPA21GProcessor,
        ],
    )
    def test_384px_4frames_square(self, cls):
        out = run_video(vid_proc_for(cls), make_video_tensor(4, 384, 384))
        assert out["pixel_values_videos"].shape == (1, 4, 3, 384, 384)
        assert out["video_grid_thw"].tolist() == [[2, 24, 24]]
        assert out["num_soft_tokens_per_video"] == [288]  # 2*12*12

    @pytest.mark.parametrize("cls", [Qwen3VLVJepa21BProcessor, Gemma4VJEPA21BProcessor])
    def test_384px_8frames_square(self, cls):
        out = run_video(vid_proc_for(cls), make_video_tensor(8, 384, 384))
        assert out["pixel_values_videos"].shape == (1, 8, 3, 384, 384)
        assert out["video_grid_thw"].tolist() == [[4, 24, 24]]
        assert out["num_soft_tokens_per_video"] == [576]  # 4*12*12

    @pytest.mark.parametrize("cls", [Qwen3VLVJepa21BProcessor, Gemma4VJEPA21BProcessor])
    def test_384px_wide_video(self, cls):
        out = run_video(vid_proc_for(cls), make_video_tensor(4, 384, 768))
        assert out["pixel_values_videos"].shape == (2, 4, 3, 384, 384)
        assert out["video_grid_thw"].tolist() == [[2, 24, 48]]
        assert out["num_soft_tokens_per_video"] == [576]  # 2*12*24

    # ── grid_t = T // tubelet_size ─────────────────────────────────────────

    @pytest.mark.parametrize(
        "cls, T, expected_grid_t",
        [
            (Qwen3VLVJepa2LProcessor, 2, 1),
            (Qwen3VLVJepa2LProcessor, 4, 2),
            (Qwen3VLVJepa2LProcessor, 8, 4),
            (Qwen3VLVJepa21BProcessor, 4, 2),
            (Qwen3VLVJepa21BProcessor, 8, 4),
            (Gemma4VJEPA21BProcessor, 4, 2),
            (Gemma4VJEPA21BProcessor, 8, 4),
        ],
    )
    def test_temporal_grid_t(self, cls, T, expected_grid_t):
        proc = vid_proc_for(cls)
        sz = proc.image_size
        out = run_video(proc, make_video_tensor(T, sz, sz))
        grid_t = out["video_grid_thw"][0, 0].item()
        assert (
            grid_t == expected_grid_t
        ), f"{cls.__name__} T={T}: grid_t={grid_t}, expected {expected_grid_t}"

    # ── token count 공식 일관성 ────────────────────────────────────────────

    @pytest.mark.parametrize(
        "cls, T, h, w",
        [
            (Qwen3VLVJepa2LProcessor, 4, 256, 256),
            (Qwen3VLVJepa2LProcessor, 4, 256, 512),
            (Qwen3VLVJepa21BProcessor, 4, 384, 384),
            (Qwen3VLVJepa21BProcessor, 8, 384, 384),
            (Qwen3VLVJepa21BProcessor, 4, 384, 768),
            (Gemma4VJepa2LProcessor, 4, 256, 256),
            (Gemma4VJEPA21BProcessor, 4, 384, 384),
            (Gemma4VJEPA21BProcessor, 8, 384, 384),
        ],
    )
    def test_token_count_matches_formula(self, cls, T, h, w):
        out = run_video(vid_proc_for(cls), make_video_tensor(T, h, w))
        grid_t, h_total, w_total = out["video_grid_thw"][0].tolist()
        formula = grid_t * (h_total // MERGE) * (w_total // MERGE)
        reported = out["num_soft_tokens_per_video"][0]
        assert (
            reported == formula
        ), f"{cls.__name__} T={T},h={h},w={w}: reported={reported}, formula={formula}"
