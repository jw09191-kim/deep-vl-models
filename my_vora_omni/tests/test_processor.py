"""
test_processor.py
=================
Qwen3VLVJEPAProcessor / Gemma4VJEPAProcessor 래퍼 클래스 검증.

검증 항목:
  1. 래퍼가 VJEPAImageProcessor / VJEPAVideoProcessor 를 올바르게 주입했는지
  2. image_processor 가 PIL 이미지를 처리해 올바른 shape 을 반환하는지
  3. video_processor 가 비디오 텐서를 처리해 올바른 shape 을 반환하는지
  4. Gemma4 래퍼의 image_seq_length 가 올바르게 계산되는지

실행:
    cd /home/jinnwoo_kim/workspace/deep-vl-models
    PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/test_processor.py -v
"""

import os
import pytest
import torch
import numpy as np
from PIL import Image
from transformers.image_utils import SizeDict

from src.processor.processor import (
    # Qwen3 래퍼 클래스
    Qwen3VLVJepa2LProcessor,
    Qwen3VLVJepa2GProcessor,
    Qwen3VLVJepa21BProcessor,
    Qwen3VLVJepa21LProcessor,
    Qwen3VLVJepa21GProcessor,
    # Gemma4 래퍼 클래스
    Gemma4VJepa2LProcessor,
    Gemma4VJepa2GProcessor,
    Gemma4VJEPA21BProcessor,
    Gemma4VJEPA21LProcessor,
    Gemma4VJEPA21GProcessor,
    # 기대 타입
    VJEPAImageProcessor,
    VJEPAVideoProcessor,
    VoRAVisionConfig,
)

MERGE = 2


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def make_pil(h: int, w: int, fill: int = 128) -> Image.Image:
    """단색 RGB PIL 이미지"""
    arr = np.full((h, w, 3), fill, dtype=np.uint8)
    return Image.fromarray(arr)


def make_video_tensor(T: int, h: int, w: int, fill: float = 128.0) -> torch.Tensor:
    """[T, C=3, H, W] float32, 픽셀값 [0, 255] 범위"""
    return torch.full((T, 3, h, w), fill, dtype=torch.float32)


def vid_preprocess(proc: VJEPAVideoProcessor, vid: torch.Tensor):
    """video_processor._preprocess 호출 공통 래퍼"""
    return proc._preprocess(
        videos=[vid],
        do_resize=True,
        size=SizeDict(
            longest_edge=proc.image_size ** 2,
            shortest_edge=proc.image_size ** 2,
        ),
        image_mean=VoRAVisionConfig.MEAN,
        image_std=VoRAVisionConfig.STD,
        do_rescale=True,
        rescale_factor=1 / 255.0,
    )


# ===========================================================================
# 1. Qwen3 래퍼 — 서브 프로세서 타입 및 설정 검증
# ===========================================================================

class TestQwen3WrapperConfig:
    """Qwen3VLVJEPAProcessor 계열이 올바른 서브 프로세서를 주입하는지 확인."""

    @pytest.mark.parametrize("cls, expected_image_size, expected_hidden", [
        (Qwen3VLVJepa2LProcessor,  256, 1024),  # ViT-L hidden=1024
        (Qwen3VLVJepa2GProcessor,  256, 1408),  # ViT-G hidden=1408
        (Qwen3VLVJepa21BProcessor, 384, 768),
        (Qwen3VLVJepa21LProcessor, 384, 1024),
        (Qwen3VLVJepa21GProcessor, 384, 1408),
    ])
    def test_sub_processor_types(self, cls, expected_image_size, expected_hidden):
        proc = cls(tokenizer=None)
        assert isinstance(proc.image_processor, VJEPAImageProcessor), \
            f"{cls.__name__}: image_processor 타입이 VJEPAImageProcessor 가 아닙니다"
        assert isinstance(proc.video_processor, VJEPAVideoProcessor), \
            f"{cls.__name__}: video_processor 타입이 VJEPAVideoProcessor 가 아닙니다"

    @pytest.mark.parametrize("cls, expected_image_size", [
        (Qwen3VLVJepa2LProcessor,  256),
        (Qwen3VLVJepa2GProcessor,  256),
        (Qwen3VLVJepa21BProcessor, 384),
        (Qwen3VLVJepa21LProcessor, 384),
        (Qwen3VLVJepa21GProcessor, 384),
    ])
    def test_image_size(self, cls, expected_image_size):
        proc = cls(tokenizer=None)
        assert proc.image_processor.image_size == expected_image_size, \
            f"{cls.__name__}: image_size={proc.image_processor.image_size}, expected {expected_image_size}"
        assert proc.video_processor.image_size == expected_image_size, \
            f"{cls.__name__}: video image_size={proc.video_processor.image_size}, expected {expected_image_size}"

    @pytest.mark.parametrize("cls", [
        Qwen3VLVJepa2LProcessor,
        Qwen3VLVJepa2GProcessor,
        Qwen3VLVJepa21BProcessor,
        Qwen3VLVJepa21LProcessor,
        Qwen3VLVJepa21GProcessor,
    ])
    def test_tubelet_and_merge_size(self, cls):
        proc = cls(tokenizer=None)
        assert proc.image_processor.tubelet_size == 2
        assert proc.image_processor.merge_size   == MERGE
        assert proc.video_processor.tubelet_size == 2
        assert proc.video_processor.merge_size   == MERGE


# ===========================================================================
# 2. Gemma4 래퍼 — 서브 프로세서 타입, 설정, image_seq_length 검증
# ===========================================================================

class TestGemma4WrapperConfig:
    """Gemma4VJEPAProcessor 계열 검증."""

    @pytest.mark.parametrize("cls, expected_image_size, expected_seq_len", [
        # image_seq_length = (image_size // patch_size // merge_size) ** 2
        # ViT-L/G (256px): (256 // 16 // 2)^2 = 8^2 = 64
        # VJEPA2.1 (384px): (384 // 16 // 2)^2 = 12^2 = 144
        (Gemma4VJepa2LProcessor,  256, 64),
        (Gemma4VJepa2GProcessor,  256, 64),
        (Gemma4VJEPA21BProcessor, 384, 144),
        (Gemma4VJEPA21LProcessor, 384, 144),
        (Gemma4VJEPA21GProcessor, 384, 144),
    ])
    def test_sub_processor_types_and_seq_len(self, cls, expected_image_size, expected_seq_len):
        proc = cls(tokenizer=None)
        assert isinstance(proc.image_processor, VJEPAImageProcessor), \
            f"{cls.__name__}: image_processor 타입 오류"
        assert isinstance(proc.video_processor, VJEPAVideoProcessor), \
            f"{cls.__name__}: video_processor 타입 오류"
        assert proc.image_processor.image_size == expected_image_size
        assert proc.image_seq_length == expected_seq_len, \
            f"{cls.__name__}: image_seq_length={proc.image_seq_length}, expected {expected_seq_len}"

    @pytest.mark.parametrize("cls", [
        Gemma4VJepa2LProcessor,
        Gemma4VJepa2GProcessor,
        Gemma4VJEPA21BProcessor,
        Gemma4VJEPA21LProcessor,
        Gemma4VJEPA21GProcessor,
    ])
    def test_tubelet_and_merge_size(self, cls):
        proc = cls(tokenizer=None)
        assert proc.image_processor.tubelet_size == 2
        assert proc.image_processor.merge_size   == MERGE
        assert proc.video_processor.tubelet_size == 2
        assert proc.video_processor.merge_size   == MERGE


# ===========================================================================
# 3. image_processor — PIL 이미지 처리 shape 검증
# ===========================================================================

class TestImageProcessorViaWrapper:
    """래퍼에서 꺼낸 image_processor 로 PIL 이미지를 처리한다."""

    # ── Qwen3 ViT-L (256px) ────────────────────────────────────────────────

    def test_qwen_vitl_square(self):
        proc = Qwen3VLVJepa2LProcessor(tokenizer=None).image_processor
        out  = proc(images=[make_pil(256, 256)], return_tensors="pt")
        assert out["pixel_values"].shape      == (1, 2, 3, 256, 256)
        assert out["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert out["num_soft_tokens_per_image"] == [64]

    def test_qwen_vitl_wide(self):
        proc = Qwen3VLVJepa2LProcessor(tokenizer=None).image_processor
        out  = proc(images=[make_pil(256, 512)], return_tensors="pt")
        # (1,2) 타일 → 2 tiles
        assert out["pixel_values"].shape      == (2, 2, 3, 256, 256)
        assert out["image_grid_thw"].tolist() == [[1, 16, 32]]
        assert out["num_soft_tokens_per_image"] == [128]

    # ── Qwen3 VJEPA2.1-B (384px) ──────────────────────────────────────────

    def test_qwen_v21b_square(self):
        proc = Qwen3VLVJepa21BProcessor(tokenizer=None).image_processor
        out  = proc(images=[make_pil(384, 384)], return_tensors="pt")
        assert out["pixel_values"].shape      == (1, 2, 3, 384, 384)
        assert out["image_grid_thw"].tolist() == [[1, 24, 24]]
        assert out["num_soft_tokens_per_image"] == [144]

    def test_qwen_v21b_wide(self):
        proc = Qwen3VLVJepa21BProcessor(tokenizer=None).image_processor
        out  = proc(images=[make_pil(384, 768)], return_tensors="pt")
        assert out["pixel_values"].shape      == (2, 2, 3, 384, 384)
        assert out["image_grid_thw"].tolist() == [[1, 24, 48]]
        assert out["num_soft_tokens_per_image"] == [288]

    # ── Gemma4 ViT-L (256px) ──────────────────────────────────────────────

    def test_gemma4_vitl_square(self):
        proc = Gemma4VJepa2LProcessor(tokenizer=None).image_processor
        out  = proc(images=[make_pil(256, 256)], return_tensors="pt")
        assert out["pixel_values"].shape      == (1, 2, 3, 256, 256)
        assert out["image_grid_thw"].tolist() == [[1, 16, 16]]
        assert out["num_soft_tokens_per_image"] == [64]

    # ── Gemma4 VJEPA2.1-B (384px) ─────────────────────────────────────────

    def test_gemma4_v21b_square(self):
        proc = Gemma4VJEPA21BProcessor(tokenizer=None).image_processor
        out  = proc(images=[make_pil(384, 384)], return_tensors="pt")
        assert out["pixel_values"].shape      == (1, 2, 3, 384, 384)
        assert out["image_grid_thw"].tolist() == [[1, 24, 24]]
        assert out["num_soft_tokens_per_image"] == [144]

    # ── token count == grid 에서 계산한 값 ────────────────────────────────

    @pytest.mark.parametrize("cls, h, w", [
        (Qwen3VLVJepa2LProcessor,  256, 256),
        (Qwen3VLVJepa2LProcessor,  256, 512),
        (Qwen3VLVJepa21BProcessor, 384, 384),
        (Qwen3VLVJepa21BProcessor, 384, 768),
        (Gemma4VJepa2LProcessor,   256, 256),
        (Gemma4VJEPA21BProcessor,  384, 384),
    ])
    def test_token_count_matches_grid(self, cls, h, w):
        proc = cls(tokenizer=None).image_processor
        out  = proc(images=[make_pil(h, w)], return_tensors="pt")
        _, h_total, w_total = out["image_grid_thw"][0].tolist()
        expected = (h_total // MERGE) * (w_total // MERGE)
        assert out["num_soft_tokens_per_image"][0] == expected, \
            f"{cls.__name__} h={h},w={w}: tokens={out['num_soft_tokens_per_image'][0]}, expected {expected}"

    # ── Gemma4 image_seq_length == tokens per single tile image ───────────

    @pytest.mark.parametrize("cls, expected_seq_len", [
        (Gemma4VJepa2LProcessor,  64),
        (Gemma4VJepa2GProcessor,  64),
        (Gemma4VJEPA21BProcessor, 144),
        (Gemma4VJEPA21LProcessor, 144),
        (Gemma4VJEPA21GProcessor, 144),
    ])
    def test_gemma4_seq_len_matches_single_tile_tokens(self, cls, expected_seq_len):
        """image_seq_length 은 타일 1개의 소프트 토큰 수와 같아야 한다."""
        wrapper = cls(tokenizer=None)
        image_size = wrapper.image_processor.image_size
        out = wrapper.image_processor(
            images=[make_pil(image_size, image_size)], return_tensors="pt"
        )
        assert out["num_soft_tokens_per_image"][0] == wrapper.image_seq_length, \
            f"{cls.__name__}: tokens={out['num_soft_tokens_per_image'][0]}, seq_len={wrapper.image_seq_length}"


# ===========================================================================
# 4. video_processor — 비디오 텐서 처리 shape 검증
# ===========================================================================

class TestVideoProcessorViaWrapper:
    """래퍼에서 꺼낸 video_processor 로 비디오를 처리한다."""

    # ── Qwen3 ViT-L (256px) ────────────────────────────────────────────────

    def test_qwen_vitl_4frames(self):
        proc = Qwen3VLVJepa2LProcessor(tokenizer=None).video_processor
        out  = vid_preprocess(proc, make_video_tensor(4, 256, 256))
        assert out["pixel_values_videos"].shape   == (1, 4, 3, 256, 256)
        assert out["video_grid_thw"].tolist()      == [[2, 16, 16]]
        assert out["num_soft_tokens_per_video"]    == [128]   # 2*8*8

    def test_qwen_vitl_wide_video(self):
        proc = Qwen3VLVJepa2LProcessor(tokenizer=None).video_processor
        out  = vid_preprocess(proc, make_video_tensor(4, 256, 512))
        assert out["pixel_values_videos"].shape   == (2, 4, 3, 256, 256)
        assert out["video_grid_thw"].tolist()      == [[2, 16, 32]]
        assert out["num_soft_tokens_per_video"]    == [256]   # 2*8*16

    # ── Qwen3 VJEPA2.1-B (384px) ──────────────────────────────────────────

    def test_qwen_v21b_4frames(self):
        proc = Qwen3VLVJepa21BProcessor(tokenizer=None).video_processor
        out  = vid_preprocess(proc, make_video_tensor(4, 384, 384))
        assert out["pixel_values_videos"].shape   == (1, 4, 3, 384, 384)
        assert out["video_grid_thw"].tolist()      == [[2, 24, 24]]
        assert out["num_soft_tokens_per_video"]    == [288]   # 2*12*12

    def test_qwen_v21b_8frames(self):
        proc = Qwen3VLVJepa21BProcessor(tokenizer=None).video_processor
        out  = vid_preprocess(proc, make_video_tensor(8, 384, 384))
        assert out["pixel_values_videos"].shape   == (1, 8, 3, 384, 384)
        assert out["video_grid_thw"].tolist()      == [[4, 24, 24]]
        assert out["num_soft_tokens_per_video"]    == [576]   # 4*12*12

    def test_qwen_v21b_wide_video(self):
        proc = Qwen3VLVJepa21BProcessor(tokenizer=None).video_processor
        out  = vid_preprocess(proc, make_video_tensor(4, 384, 768))
        assert out["pixel_values_videos"].shape   == (2, 4, 3, 384, 384)
        assert out["video_grid_thw"].tolist()      == [[2, 24, 48]]
        assert out["num_soft_tokens_per_video"]    == [576]   # 2*12*24

    # ── Gemma4 ViT-L (256px) ──────────────────────────────────────────────

    def test_gemma4_vitl_4frames(self):
        proc = Gemma4VJepa2LProcessor(tokenizer=None).video_processor
        out  = vid_preprocess(proc, make_video_tensor(4, 256, 256))
        assert out["pixel_values_videos"].shape   == (1, 4, 3, 256, 256)
        assert out["video_grid_thw"].tolist()      == [[2, 16, 16]]
        assert out["num_soft_tokens_per_video"]    == [128]

    # ── Gemma4 VJEPA2.1-B (384px) ─────────────────────────────────────────

    def test_gemma4_v21b_4frames(self):
        proc = Gemma4VJEPA21BProcessor(tokenizer=None).video_processor
        out  = vid_preprocess(proc, make_video_tensor(4, 384, 384))
        assert out["pixel_values_videos"].shape   == (1, 4, 3, 384, 384)
        assert out["video_grid_thw"].tolist()      == [[2, 24, 24]]
        assert out["num_soft_tokens_per_video"]    == [288]

    # ── token count 공식 일관성 ────────────────────────────────────────────

    @pytest.mark.parametrize("cls, T, h, w, expected_tokens", [
        (Qwen3VLVJepa2LProcessor,  4,  256, 256, 128),   # 2*8*8
        (Qwen3VLVJepa2LProcessor,  8,  256, 256, 256),   # 4*8*8
        (Qwen3VLVJepa21BProcessor, 4,  384, 384, 288),   # 2*12*12
        (Qwen3VLVJepa21BProcessor, 8,  384, 384, 576),   # 4*12*12
        (Qwen3VLVJepa21BProcessor, 4,  384, 768, 576),   # 2*12*24
        (Gemma4VJepa2LProcessor,   4,  256, 256, 128),
        (Gemma4VJEPA21BProcessor,  4,  384, 384, 288),
    ])
    def test_token_count(self, cls, T, h, w, expected_tokens):
        proc = cls(tokenizer=None).video_processor
        out  = vid_preprocess(proc, make_video_tensor(T, h, w))
        grid_t, h_total, w_total = out["video_grid_thw"][0].tolist()
        formula = grid_t * (h_total // MERGE) * (w_total // MERGE)
        assert out["num_soft_tokens_per_video"][0] == expected_tokens, \
            f"{cls.__name__} T={T},h={h},w={w}: got {out['num_soft_tokens_per_video'][0]}"
        assert out["num_soft_tokens_per_video"][0] == formula, \
            f"token count 공식 불일치: {out['num_soft_tokens_per_video'][0]} != {formula}"
