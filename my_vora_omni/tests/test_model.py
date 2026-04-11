"""
test_model.py
=============
VJEPA2VisualModule / get_image_features / get_video_features 및 관련 로직 검증.

실제 체크포인트나 GPU 없이 CPU 단위 테스트로 실행 가능.

실행:
    cd /home/jinnwoo_kim/workspace/deep-vl-models
    PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/test_model.py -v
"""

import types
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from transformers.modeling_outputs import BaseModelOutputWithPooling

import src.model.model as model_module
from src.model.model import (
    VJEPA2VisualModule,
    Qwen3_5VJEPAInnerModel,
    Qwen3_5VJEPALModel,
    Qwen3_5VJEPAGModel,
    Qwen3_5VJEPA21BModel,
    Qwen3_5VJEPA21LModel,
    Qwen3_5VJEPA21GModel,
    Gemma4VJEPAModel,
    Gemma4VJEPALModel,
    Gemma4VJEPAGModel,
    Gemma4VJEPA21BModel,
    Gemma4VJEPA21LModel,
    Gemma4VJEPA21GModel,
    _gemma4_safe_get_per_layer_inputs,
    _gemma4_orig_get_per_layer_inputs,
)


# ---------------------------------------------------------------------------
# 헬퍼: 가짜 visual 모듈 및 모델 객체 생성
# ---------------------------------------------------------------------------

class FakeVisual(nn.Module):
    """
    VJEPA2VisualModule 역할을 하는 최소 가짜 객체.
    - patches_per_side / dtype property 제공
    - merger: Linear(vjepa_dim * merge_size², llm_dim)
    - forward: set_encoder_output()으로 주입한 텐서 반환
    """

    def __init__(self, pps: int, vjepa_dim: int, merge_size: int, llm_dim: int):
        super().__init__()
        self._pps = pps
        self._merge_size = merge_size
        self._vjepa_dim = vjepa_dim
        in_dim = vjepa_dim * merge_size ** 2
        self.merger = nn.Linear(in_dim, llm_dim, bias=False)
        self._encoder_output: torch.Tensor | None = None

    @property
    def patches_per_side(self) -> int:
        return self._pps

    @property
    def dtype(self) -> torch.dtype:
        return next(self.merger.parameters()).dtype

    def set_encoder_output(self, tensor: torch.Tensor):
        """테스트별로 encoder가 반환할 텐서를 주입."""
        self._encoder_output = tensor

    def forward(self, pixel_values, **kwargs):
        return self._encoder_output


def make_fake_qwen_model(
    pps: int = 16,
    merge_size: int = 2,
    vjepa_dim: int = 64,
    llm_dim: int = 32,
) -> types.SimpleNamespace:
    """
    Qwen3_5VJEPAInnerModel.get_image_features 가 필요로 하는
    self.config / self.visual 속성만 갖는 가짜 객체 반환.
    """
    visual = FakeVisual(pps, vjepa_dim, merge_size, llm_dim)
    config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=merge_size)
    )
    return types.SimpleNamespace(config=config, visual=visual)


def make_fake_gemma4_model(
    pps: int = 16,
    merge_size: int = 2,
    vjepa_dim: int = 64,
    llm_dim: int = 32,
) -> types.SimpleNamespace:
    """
    Gemma4VJEPAModel.get_image_features 가 필요로 하는 가짜 객체.
    Gemma4는 outer model에 self.visual이 붙으므로 구조는 동일.
    """
    return make_fake_qwen_model(pps, merge_size, vjepa_dim, llm_dim)


def encoder_output(n_items: int, patches: int, vjepa_dim: int) -> torch.Tensor:
    """[n_items, patches, vjepa_dim] 형태의 더미 텐서."""
    return torch.zeros(n_items, patches, vjepa_dim)


# ---------------------------------------------------------------------------
# 1. VJEPA2VisualModule 구조 검증
# ---------------------------------------------------------------------------

class TestVJEPA2VisualModule:
    """VJEPA2VisualModule 인스턴스화 및 내부 구조 검증 (encoder 는 mock)."""

    PPS = 16
    VJEPA_DIM = 64
    MERGE = 2
    LLM_DIM = 32

    def _make_module(self, is_v21=False) -> VJEPA2VisualModule:
        mock_encoder = MagicMock()
        return VJEPA2VisualModule(
            vjepa2_model=mock_encoder,
            vjepa_dim=self.VJEPA_DIM,
            merge_size=self.MERGE,
            llm_dim=self.LLM_DIM,
            is_v21=is_v21,
            patches_per_side=self.PPS,
        )

    def test_merger_architecture(self):
        """merger Sequential의 레이어 순서: LN→Lin→GELU→Lin→GELU→LN→Lin→LN."""
        m = self._make_module()
        layer_types = [type(l).__name__ for l in m.merger]
        assert layer_types == [
            "LayerNorm", "Linear", "GELU",
            "Linear", "GELU",
            "LayerNorm", "Linear", "LayerNorm",
        ], f"예상과 다른 merger 구조: {layer_types}"

    def test_merger_input_output_dims(self):
        """첫 Linear의 in_features = vjepa_dim * merge² , 마지막 Linear의 out_features = llm_dim."""
        m = self._make_module()
        linears = [l for l in m.merger if isinstance(l, nn.Linear)]
        in_dim = self.VJEPA_DIM * self.MERGE ** 2
        assert linears[0].in_features == in_dim
        assert linears[-1].out_features == self.LLM_DIM

    def test_properties(self):
        m = self._make_module()
        assert m.patches_per_side == self.PPS
        assert m.spatial_merge_size == self.MERGE
        assert m.dtype == torch.float32  # 기본 float32

    def test_forward_v2_mode(self):
        """is_v21=False: encoder().last_hidden_state 를 반환."""
        mock_encoder = MagicMock()
        n_items, patches = 2, self.PPS ** 2
        expected = torch.zeros(n_items, patches, self.VJEPA_DIM)
        mock_encoder.return_value.last_hidden_state = expected

        m = VJEPA2VisualModule(
            mock_encoder, self.VJEPA_DIM, self.MERGE, self.LLM_DIM,
            is_v21=False, patches_per_side=self.PPS,
        )
        pixel_values = torch.zeros(n_items, 3, self.PPS, self.PPS)
        out = m(pixel_values)
        assert out.shape == expected.shape

    def test_forward_v21_mode_permutes_input(self):
        """is_v21=True: encoder는 permute된 텐서를 받아야 함."""
        received: list[torch.Tensor] = []

        class RecordEncoder(nn.Module):
            def forward(self, x):
                received.append(x.clone())
                return torch.zeros(x.shape[0], x.shape[2] * x.shape[3] * x.shape[4], 64)

        m = VJEPA2VisualModule(
            RecordEncoder(), self.VJEPA_DIM, self.MERGE, self.LLM_DIM,
            is_v21=True, patches_per_side=self.PPS,
        )
        # [n, C, T, H, W] → 내부에서 permute(0,2,1,3,4) → [n, T, C, H, W]
        pixel_values = torch.zeros(1, 3, 2, self.PPS, self.PPS)
        m(pixel_values)

        assert len(received) == 1
        perm = received[0]
        # permute 후 shape: [n, T, C, H, W]
        assert perm.shape == (1, 2, 3, self.PPS, self.PPS)

    def test_forward_v21_mode_output_shape(self):
        """is_v21=True: forward 결과 shape == [n_items, patches, vjepa_dim]."""
        n_items, T = 1, 2
        patches = T * self.PPS ** 2

        class DummyEncoder(nn.Module):
            def forward(self, x):  # x: [n, T, C, H, W]
                return torch.zeros(x.shape[0], x.shape[1] * x.shape[3] * x.shape[4], 64)

        m = VJEPA2VisualModule(
            DummyEncoder(), self.VJEPA_DIM, self.MERGE, self.LLM_DIM,
            is_v21=True, patches_per_side=self.PPS,
        )
        pv = torch.zeros(n_items, 3, T, self.PPS, self.PPS)
        out = m(pv)
        assert out.shape == (n_items, patches, self.VJEPA_DIM)


# ---------------------------------------------------------------------------
# 2. Qwen3.5 get_image_features — tiling 어셈블리 및 merge 검증
# ---------------------------------------------------------------------------

class TestGetImageFeatures:
    """
    Qwen3_5VJEPAInnerModel.get_image_features 를 unbound 호출로 검증.
    실제 Qwen3_5 가중치 로드 없이 FakeVisual로 동작.
    """

    PPS = 16
    MERGE = 2
    VJEPA_DIM = 64
    LLM_DIM = 32

    def _call(self, fake_model, n_items: int, t: int, grid_thw: list[list[int]]):
        """
        fake_model.visual에 encoder 출력을 설정한 뒤 get_image_features 호출.
        n_items: encoder가 반환하는 총 항목 수 (타일 수의 합)
        t: 각 항목의 시간 축 (이미지=1, 비디오=t)
        """
        pps = self.PPS
        enc_out = encoder_output(n_items, t * pps * pps, self.VJEPA_DIM)
        fake_model.visual.set_encoder_output(enc_out)
        grid = torch.tensor(grid_thw, dtype=torch.long)
        dummy_pv = torch.zeros(n_items, 1)  # shape은 무시됨
        result = Qwen3_5VJEPAInnerModel.get_image_features(
            fake_model, dummy_pv, grid
        )
        return result.pooler_output  # tuple of tensors

    def test_single_tile_image(self):
        """`[[1, pps, pps]]` — 1 타일, 토큰 수 = (pps/m)²."""
        fake = make_fake_qwen_model(self.PPS, self.MERGE, self.VJEPA_DIM, self.LLM_DIM)
        out = self._call(fake, n_items=1, t=1, grid_thw=[[1, self.PPS, self.PPS]])
        assert len(out) == 1
        expected_tokens = (self.PPS // self.MERGE) ** 2
        assert out[0].shape == (1, expected_tokens, self.LLM_DIM)

    def test_wide_2tile_image(self):
        """`[[1, pps, 2*pps]]` — 1×2 타일, 토큰 수 = (pps/m) * (2pps/m)."""
        fake = make_fake_qwen_model(self.PPS, self.MERGE, self.VJEPA_DIM, self.LLM_DIM)
        out = self._call(fake, n_items=2, t=1, grid_thw=[[1, self.PPS, 2 * self.PPS]])
        m = self.MERGE
        pps = self.PPS
        expected_tokens = (pps // m) * (2 * pps // m)
        assert out[0].shape == (1, expected_tokens, self.LLM_DIM)

    def test_tall_2tile_image(self):
        """`[[1, 2*pps, pps]]` — 2×1 타일, 토큰 수 = (2pps/m) * (pps/m)."""
        fake = make_fake_qwen_model(self.PPS, self.MERGE, self.VJEPA_DIM, self.LLM_DIM)
        out = self._call(fake, n_items=2, t=1, grid_thw=[[1, 2 * self.PPS, self.PPS]])
        m = self.MERGE
        pps = self.PPS
        expected_tokens = (2 * pps // m) * (pps // m)
        assert out[0].shape == (1, expected_tokens, self.LLM_DIM)

    def test_4tile_image(self):
        """`[[1, 2*pps, 2*pps]]` — 2×2 타일, 토큰 수 = (2pps/m)²."""
        fake = make_fake_qwen_model(self.PPS, self.MERGE, self.VJEPA_DIM, self.LLM_DIM)
        out = self._call(fake, n_items=4, t=1, grid_thw=[[1, 2 * self.PPS, 2 * self.PPS]])
        m = self.MERGE
        pps = self.PPS
        expected_tokens = (2 * pps // m) ** 2
        assert out[0].shape == (1, expected_tokens, self.LLM_DIM)

    def test_multiple_images_batch(self):
        """이미지 2장: 첫 번째는 1타일, 두 번째는 2타일(wide)."""
        pps, m = self.PPS, self.MERGE
        # 총 encoder 항목: 1(첫 번째 이미지) + 2(두 번째 이미지) = 3
        fake = make_fake_qwen_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        enc_out = encoder_output(3, pps * pps, self.VJEPA_DIM)
        fake.visual.set_encoder_output(enc_out)
        grid = torch.tensor([[1, pps, pps], [1, pps, 2 * pps]], dtype=torch.long)
        dummy_pv = torch.zeros(3, 1)
        results = Qwen3_5VJEPAInnerModel.get_image_features(
            fake, dummy_pv, grid
        ).pooler_output

        assert len(results) == 2
        assert results[0].shape == (1, (pps // m) ** 2, self.LLM_DIM)
        assert results[1].shape == (1, (pps // m) * (2 * pps // m), self.LLM_DIM)

    @pytest.mark.parametrize(
        "h, w, n_tiles",
        [
            (16, 16, 1),   # 1×1
            (16, 32, 2),   # 1×2
            (32, 16, 2),   # 2×1
            (32, 32, 4),   # 2×2
        ],
    )
    def test_output_token_count_matches_formula(self, h, w, n_tiles):
        """결과 토큰 수 == (h // merge) * (w // merge)."""
        pps, m = self.PPS, self.MERGE
        fake = make_fake_qwen_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        out = self._call(fake, n_items=n_tiles, t=1, grid_thw=[[1, h, w]])
        expected_tokens = (h // m) * (w // m)
        assert out[0].shape[1] == expected_tokens, (
            f"h={h},w={w}: tokens={out[0].shape[1]}, formula={expected_tokens}"
        )


# ---------------------------------------------------------------------------
# 3. Qwen3.5 get_video_features — 비디오 분기 검증
# ---------------------------------------------------------------------------

class TestGetVideoFeatures:
    """비디오 경로: t > 1 인 grid_thw 처리."""

    PPS = 16
    MERGE = 2
    VJEPA_DIM = 64
    LLM_DIM = 32

    def _call_video(self, fake_model, n_tiles: int, t: int, grid_thw: list[list[int]]):
        pps = self.PPS
        enc_out = encoder_output(n_tiles, t * pps * pps, self.VJEPA_DIM)
        fake_model.visual.set_encoder_output(enc_out)
        grid = torch.tensor(grid_thw, dtype=torch.long)
        dummy_pv = torch.zeros(n_tiles, 1)
        result = Qwen3_5VJEPAInnerModel.get_video_features(
            fake_model, dummy_pv, grid
        )
        return result.pooler_output

    def test_video_single_tile(self):
        """`[[t, pps, pps]]` — 단일 공간 타일, 시간 t."""
        pps, m = self.PPS, self.MERGE
        fake = make_fake_qwen_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        t = 2
        out = self._call_video(fake, n_tiles=1, t=t, grid_thw=[[t, pps, pps]])
        expected_tokens = (pps // m) ** 2
        assert out[0].shape == (t, expected_tokens, self.LLM_DIM)

    def test_video_wide_multitile(self):
        """`[[t, pps, 2*pps]]` — 2 공간 타일, wide."""
        pps, m = self.PPS, self.MERGE
        fake = make_fake_qwen_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        t = 2
        out = self._call_video(fake, n_tiles=2, t=t, grid_thw=[[t, pps, 2 * pps]])
        expected_tokens = (pps // m) * (2 * pps // m)
        assert out[0].shape == (t, expected_tokens, self.LLM_DIM)

    @pytest.mark.parametrize("t", [2, 4])
    def test_video_temporal_dimension(self, t):
        """결과 tensor의 첫 번째 차원 == t (시간 그리드 수)."""
        pps, m = self.PPS, self.MERGE
        fake = make_fake_qwen_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        out = self._call_video(fake, n_tiles=1, t=t, grid_thw=[[t, pps, pps]])
        assert out[0].shape[0] == t, f"t={t}: result.shape[0]={out[0].shape[0]}"

    def test_video_delegates_to_image_features(self):
        """get_video_features 가 get_image_features 와 동일한 결과를 반환."""
        pps, m = self.PPS, self.MERGE
        t = 2
        grid_thw = [[t, pps, pps]]
        enc_out = encoder_output(1, t * pps * pps, self.VJEPA_DIM)

        fake1 = make_fake_qwen_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        fake1.visual.set_encoder_output(enc_out)
        grid = torch.tensor(grid_thw, dtype=torch.long)
        dummy_pv = torch.zeros(1, 1)

        # 동일 가중치 재현을 위해 같은 visual 공유
        fake2 = types.SimpleNamespace(config=fake1.config, visual=fake1.visual)

        res_img = Qwen3_5VJEPAInnerModel.get_image_features(fake1, dummy_pv, grid)
        fake1.visual.set_encoder_output(enc_out)
        res_vid = Qwen3_5VJEPAInnerModel.get_video_features(fake2, dummy_pv, grid)

        assert res_img.pooler_output[0].shape == res_vid.pooler_output[0].shape


# ---------------------------------------------------------------------------
# 4. Gemma4 get_image_features — Qwen3.5와 동일한 tiling 로직 검증
# ---------------------------------------------------------------------------

class TestGemma4GetImageFeatures:
    """
    Gemma4VJEPAModel.get_image_features 검증.
    로직은 Qwen3.5와 동일하므로 핵심 케이스만 확인.
    """

    PPS = 16
    MERGE = 2
    VJEPA_DIM = 64
    LLM_DIM = 32

    def _call(self, fake_model, n_items: int, t: int, grid_thw: list[list[int]]):
        pps = self.PPS
        enc_out = encoder_output(n_items, t * pps * pps, self.VJEPA_DIM)
        fake_model.visual.set_encoder_output(enc_out)
        grid = torch.tensor(grid_thw, dtype=torch.long)
        dummy_pv = torch.zeros(n_items, 1)
        result = Gemma4VJEPAModel.get_image_features(fake_model, dummy_pv, grid)
        return result.pooler_output

    @pytest.mark.parametrize(
        "h, w, n_tiles",
        [
            (16, 16, 1),
            (16, 32, 2),
            (32, 16, 2),
            (32, 32, 4),
        ],
    )
    def test_output_token_count(self, h, w, n_tiles):
        """결과 토큰 수 == (h // merge) * (w // merge)."""
        pps, m = self.PPS, self.MERGE
        fake = make_fake_gemma4_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        out = self._call(fake, n_items=n_tiles, t=1, grid_thw=[[1, h, w]])
        expected = (h // m) * (w // m)
        assert out[0].shape[1] == expected

    def test_video_single_tile(self):
        pps, m, t = self.PPS, self.MERGE, 2
        fake = make_fake_gemma4_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        out = self._call(fake, n_items=1, t=t, grid_thw=[[t, pps, pps]])
        assert out[0].shape == (t, (pps // m) ** 2, self.LLM_DIM)

    def test_multiple_grids_in_batch(self):
        """이미지 2장: 각각 다른 grid_thw."""
        pps, m = self.PPS, self.MERGE
        fake = make_fake_gemma4_model(pps, m, self.VJEPA_DIM, self.LLM_DIM)
        enc_out = encoder_output(3, pps * pps, self.VJEPA_DIM)
        fake.visual.set_encoder_output(enc_out)
        grid = torch.tensor([[1, pps, pps], [1, pps, 2 * pps]], dtype=torch.long)
        dummy_pv = torch.zeros(3, 1)
        results = Gemma4VJEPAModel.get_image_features(fake, dummy_pv, grid).pooler_output
        assert len(results) == 2
        assert results[0].shape[1] == (pps // m) ** 2
        assert results[1].shape[1] == (pps // m) * (2 * pps // m)


# ---------------------------------------------------------------------------
# 5. Gemma4 OOM Fix (_gemma4_safe_get_per_layer_inputs)
# ---------------------------------------------------------------------------

class TestGemma4OOMFix:
    """monkey-patch된 get_per_layer_inputs 의 세 가지 경로 검증."""

    def _make_fake_self(self, pad_token_id: int = 0):
        return types.SimpleNamespace(
            config=types.SimpleNamespace(pad_token_id=pad_token_id)
        )

    def test_uses_vjepa_input_ids_when_set(self):
        """`_vjepa_llm_input_ids` 가 있으면 그것을 orig에 전달하고 속성 삭제."""
        preset_ids = torch.tensor([[1, 2, 3]])
        fake_self = self._make_fake_self()
        fake_self._vjepa_llm_input_ids = preset_ids
        inputs_embeds = torch.zeros(1, 3, 8)

        captured = {}

        def fake_orig(self_, input_ids, inputs_embeds_):
            captured["input_ids"] = input_ids
            return input_ids

        with patch.object(model_module, "_gemma4_orig_get_per_layer_inputs", fake_orig):
            _gemma4_safe_get_per_layer_inputs(fake_self, None, inputs_embeds)

        assert torch.equal(captured["input_ids"], preset_ids)
        assert not hasattr(fake_self, "_vjepa_llm_input_ids"), \
            "_vjepa_llm_input_ids 가 삭제되지 않았습니다"

    def test_fallback_to_pad_tokens(self):
        """`_vjepa_llm_input_ids` 없을 때 → PAD 토큰으로 채운 텐서 생성."""
        pad_id = 99
        fake_self = self._make_fake_self(pad_token_id=pad_id)
        batch, seq_len = 2, 5
        inputs_embeds = torch.zeros(batch, seq_len, 8)

        captured = {}

        def fake_orig(self_, input_ids, inputs_embeds_):
            captured["input_ids"] = input_ids
            return input_ids

        with patch.object(model_module, "_gemma4_orig_get_per_layer_inputs", fake_orig):
            _gemma4_safe_get_per_layer_inputs(fake_self, None, inputs_embeds)

        ids = captured["input_ids"]
        assert ids.shape == (batch, seq_len)
        assert ids.dtype == torch.long
        assert ids.eq(pad_id).all(), f"PAD id={pad_id} 로 채워져야 함, 실제: {ids}"

    def test_passthrough_when_input_ids_provided(self):
        """`input_ids` 가 주어지면 그대로 orig에 전달."""
        fake_self = self._make_fake_self()
        input_ids = torch.tensor([[7, 8, 9]])
        inputs_embeds = torch.zeros(1, 3, 8)

        captured = {}

        def fake_orig(self_, input_ids_, inputs_embeds_):
            captured["input_ids"] = input_ids_
            return input_ids_

        with patch.object(model_module, "_gemma4_orig_get_per_layer_inputs", fake_orig):
            _gemma4_safe_get_per_layer_inputs(fake_self, input_ids, inputs_embeds)

        assert torch.equal(captured["input_ids"], input_ids)


# ---------------------------------------------------------------------------
# 6. 모델 변형 클래스 속성 검증 (가중치 로드 없음)
# ---------------------------------------------------------------------------

class TestModelVariantAttributes:
    """VISION_MODEL_ID / TORCH_HUB_NAME 클래스 속성만 검증."""

    @pytest.mark.parametrize(
        "cls, expected_vision_model_id",
        [
            (Qwen3_5VJEPALModel, "facebook/vjepa2-vitl-fpc64-256"),
            (Qwen3_5VJEPAGModel, "facebook/vjepa2-vitg-fpc64-256"),
            (Gemma4VJEPALModel,  "facebook/vjepa2-vitl-fpc64-256"),
            (Gemma4VJEPAGModel,  "facebook/vjepa2-vitg-fpc64-256"),
        ],
    )
    def test_vision_model_id(self, cls, expected_vision_model_id):
        assert cls.VISION_MODEL_ID == expected_vision_model_id, (
            f"{cls.__name__}.VISION_MODEL_ID={cls.VISION_MODEL_ID!r}"
        )

    @pytest.mark.parametrize(
        "cls",
        [Qwen3_5VJEPA21BModel, Qwen3_5VJEPA21LModel, Qwen3_5VJEPA21GModel,
         Gemma4VJEPA21BModel, Gemma4VJEPA21LModel, Gemma4VJEPA21GModel],
    )
    def test_torch_hub_variants_have_no_hf_vision_id(self, cls):
        """torch.hub 계열은 VISION_MODEL_ID == None."""
        assert cls.VISION_MODEL_ID is None, (
            f"{cls.__name__}.VISION_MODEL_ID 가 None 이어야 함"
        )

    @pytest.mark.parametrize(
        "cls, expected_hub_name",
        [
            (Qwen3_5VJEPA21BModel, "vjepa2_1_vit_base_384"),
            (Qwen3_5VJEPA21LModel, "vjepa2_1_vit_large_384"),
            (Qwen3_5VJEPA21GModel, "vjepa2_1_vit_giant_384"),
            (Gemma4VJEPA21BModel,  "vjepa2_1_vit_base_384"),
            (Gemma4VJEPA21LModel,  "vjepa2_1_vit_large_384"),
            (Gemma4VJEPA21GModel,  "vjepa2_1_vit_giant_384"),
        ],
    )
    def test_torch_hub_name(self, cls, expected_hub_name):
        assert cls.TORCH_HUB_NAME == expected_hub_name, (
            f"{cls.__name__}.TORCH_HUB_NAME={cls.TORCH_HUB_NAME!r}"
        )
