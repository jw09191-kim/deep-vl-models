"""
test_template.py
================
Qwen3_5VJEPATemplate / Gemma4VJEPATemplate._post_encode 및
Gemma4VJEPATemplate._data_collator 검증.

_post_encode 는 Swift/Transformers 의존성 없이
- template self 를 MagicMock 으로 대체
- base_model 을 MagicMock 으로 대체
하여 unbound 호출로 테스트한다.

실행:
    cd /home/jinnwoo_kim/workspace/deep-vl-models
    PYTHONPATH=my_vora_omni python -m pytest my_vora_omni/tests/test_template.py -v
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, call
from transformers.modeling_outputs import BaseModelOutputWithPooling

from swift.template.templates.gemma import Gemma4Template

from src.template.template import Qwen3_5VJEPATemplate, Gemma4VJEPATemplate


# ---------------------------------------------------------------------------
# 공통 상수
# ---------------------------------------------------------------------------

PPS = 16        # patches_per_side
EMBED_DIM = 32  # embedding dimension
SEQ_LEN = 30    # 입력 시퀀스 길이


# ---------------------------------------------------------------------------
# 헬퍼: 가짜 self / base_model 생성
# ---------------------------------------------------------------------------

def _img_output(n_tokens: int, embed_dim: int = EMBED_DIM) -> BaseModelOutputWithPooling:
    """get_image/video_features 가 반환하는 BaseModelOutputWithPooling."""
    return BaseModelOutputWithPooling(
        pooler_output=(torch.zeros(1, n_tokens, embed_dim),)
    )


def _bool_mask_3d(n_tokens: int, seq_len: int = SEQ_LEN,
                  embed_dim: int = EMBED_DIM) -> torch.Tensor:
    """
    Qwen3.5 masked_scatter 용 3D bool 마스크.
    inputs_embeds 와 동일 shape [1, seq_len, embed_dim],
    첫 n_tokens 행이 True → True 원소 수 = n_tokens * embed_dim
    (image_embeds 의 원소 수와 일치해야 masked_scatter 성공).
    """
    mask = torch.zeros(1, seq_len, embed_dim, dtype=torch.bool)
    mask[0, :n_tokens, :] = True
    return mask


def _bool_mask_2d(n_tokens: int, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """
    Gemma4 masked_scatter 용 2D bool 마스크.
    [1, seq_len], 첫 n_tokens 위치가 True.
    unsqueeze(-1).expand_as(inputs_embeds) 후 True 원소 수 = n_tokens * embed_dim.
    """
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    mask[0, :n_tokens] = True
    return mask


def make_qwen_mocks(
    pps: int = PPS,
    embed_dim: int = EMBED_DIM,
    seq_len: int = SEQ_LEN,
    n_img_tokens: int = 4,
    n_vid_tokens: int = 4,
):
    """
    Qwen3_5VJEPATemplate._post_encode 에 필요한 (fake_self, base_model) 반환.
    fake_self.is_training = True 로 설정되어 있음.
    """
    base_model = MagicMock()
    inputs_embeds = torch.zeros(1, seq_len, embed_dim)
    base_model.model.language_model.embed_tokens.return_value = inputs_embeds.clone()
    base_model.model.visual.patches_per_side = pps

    # image / video 기본 반환값
    base_model.model.get_image_features.return_value = _img_output(n_img_tokens, embed_dim)
    base_model.model.get_video_features.return_value = _img_output(n_vid_tokens, embed_dim)

    # get_placeholder_mask — (image_mask_3d, video_mask_3d) 반환
    img_mask = _bool_mask_3d(n_img_tokens, seq_len, embed_dim)
    vid_mask = _bool_mask_3d(n_vid_tokens, seq_len, embed_dim)
    base_model.model.get_placeholder_mask.return_value = (img_mask, vid_mask)

    fake_self = MagicMock()
    fake_self.is_training = True
    fake_self.get_base_model.return_value = base_model

    return fake_self, base_model


def make_gemma4_mocks(
    pps: int = PPS,
    embed_dim: int = EMBED_DIM,
    seq_len: int = SEQ_LEN,
    n_img_tokens: int = 4,
    n_vid_tokens: int = 4,
):
    """
    Gemma4VJEPATemplate._post_encode 에 필요한 (fake_self, base_model) 반환.
    """
    base_model = MagicMock()
    inputs_embeds = torch.zeros(1, seq_len, embed_dim)
    # Gemma4: base_model.model.get_input_embeddings()(input_ids)
    base_model.model.get_input_embeddings.return_value = MagicMock(
        return_value=inputs_embeds.clone()
    )
    base_model.visual.patches_per_side = pps

    base_model.get_image_features.return_value = _img_output(n_img_tokens, embed_dim)
    base_model.get_video_features.return_value = _img_output(n_vid_tokens, embed_dim)

    # get_placeholder_mask — (image_mask_2d, video_mask_2d, audio_mask_2d) 반환
    img_mask = _bool_mask_2d(n_img_tokens, seq_len)
    vid_mask = _bool_mask_2d(n_vid_tokens, seq_len)
    audio_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    # OOM fix 분기: hidden_size_per_layer_input 없음 → 캐싱 건너뜀 (기본)
    base_model.model.config.text_config.hidden_size_per_layer_input = None
    base_model.model.get_placeholder_mask.return_value = (img_mask, vid_mask, audio_mask)

    fake_self = MagicMock()
    fake_self.is_training = True
    fake_self.get_base_model.return_value = base_model

    return fake_self, base_model


# ---------------------------------------------------------------------------
# 1. Qwen3.5 — is_training=False 조기 반환
# ---------------------------------------------------------------------------

class TestQwenPostEncodeInference:

    def test_returns_inputs_unchanged_when_not_training(self):
        """is_training=False → inputs 를 그대로 반환."""
        fake_self = MagicMock()
        fake_self.is_training = False

        sentinel = {"input_ids": torch.zeros(1, 5), "pixel_values": torch.zeros(1)}
        result = Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), sentinel)

        assert result is sentinel

    def test_base_model_not_called_when_not_training(self):
        """is_training=False → get_base_model 호출 없음."""
        fake_self = MagicMock()
        fake_self.is_training = False

        Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), {"input_ids": torch.zeros(1, 5)})

        fake_self.get_base_model.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Qwen3.5 — tile_offset 계산 (이미지)
# ---------------------------------------------------------------------------

class TestQwenPostEncodeTileOffset:
    """
    get_image_features 에 전달되는 pixel_values 슬라이스가
    tile_offset 에 따라 올바르게 분리되는지 검증.
    """

    def _run(self, fake_self, base_model, pixel_values, image_grid_thw):
        inputs = {
            "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)

    def test_single_image_single_tile(self):
        """1장, 1타일 → pixel_values[0:1] 전달."""
        fake_self, base_model = make_qwen_mocks(n_img_tokens=(PPS // 2) ** 2)
        pv = torch.zeros(1, 2, 3, PPS, PPS)
        grid = torch.tensor([[1, PPS, PPS]])

        self._run(fake_self, base_model, pv, grid)

        args = base_model.model.get_image_features.call_args_list
        assert len(args) == 1
        passed_pv = args[0][0][0]
        assert passed_pv.shape == pv[0:1].shape

    def test_single_image_2tiles_wide(self):
        """1장, 2타일(1×2) → pixel_values[0:2] 전달."""
        n_tokens = (PPS // 2) * (2 * PPS // 2)
        fake_self, base_model = make_qwen_mocks(n_img_tokens=n_tokens)
        pv = torch.zeros(2, 2, 3, PPS, PPS)
        grid = torch.tensor([[1, PPS, 2 * PPS]])

        self._run(fake_self, base_model, pv, grid)

        args = base_model.model.get_image_features.call_args_list
        assert len(args) == 1
        passed_pv = args[0][0][0]
        assert passed_pv.shape == pv[0:2].shape

    def test_two_images_with_different_tile_counts(self):
        """
        이미지 2장: 첫 번째 1타일, 두 번째 2타일.
        → 첫 번째 호출: pixel_values[0:1]
           두 번째 호출: pixel_values[1:3]
        """
        n1 = (PPS // 2) ** 2        # 1타일 토큰 수
        n2 = (PPS // 2) * (2 * PPS // 2)  # 2타일 토큰 수

        fake_self, base_model = make_qwen_mocks(n_img_tokens=n1)
        # 두 번째 이미지 호출 시 n2 토큰 반환하도록 side_effect 설정
        fake_self2, base_model2 = make_qwen_mocks(n_img_tokens=n2)
        base_model.model.get_image_features.side_effect = [
            _img_output(n1),
            _img_output(n2),
        ]

        # 두 호출의 mask 합산: n1 + n2 tokens
        total = n1 + n2
        combined_mask = _bool_mask_3d(total)
        base_model.model.get_placeholder_mask.return_value = (combined_mask, None)

        pv = torch.zeros(3, 2, 3, PPS, PPS)   # 총 3타일
        grid = torch.tensor([
            [1, PPS, PPS],        # 1타일
            [1, PPS, 2 * PPS],    # 2타일
        ])

        inputs = {
            "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
            "pixel_values": pv,
            "image_grid_thw": grid,
        }
        Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)

        calls = base_model.model.get_image_features.call_args_list
        assert len(calls) == 2

        first_pv  = calls[0][0][0]
        second_pv = calls[1][0][0]
        assert first_pv.shape[0] == 1, "첫 번째 이미지: 1타일 slice"
        assert second_pv.shape[0] == 2, "두 번째 이미지: 2타일 slice"

    def test_grid_thw_passed_per_image(self):
        """각 이미지에 대해 image_grid_thw[i:i+1] 이 전달되는지."""
        n_tokens = (PPS // 2) ** 2
        fake_self, base_model = make_qwen_mocks(n_img_tokens=n_tokens)
        pv = torch.zeros(2, 2, 3, PPS, PPS)
        grid = torch.tensor([[1, PPS, PPS], [1, PPS, PPS]])

        base_model.model.get_image_features.side_effect = [
            _img_output(n_tokens), _img_output(n_tokens)
        ]
        total_mask = _bool_mask_3d(n_tokens * 2)
        base_model.model.get_placeholder_mask.return_value = (total_mask, None)

        inputs = {
            "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
            "pixel_values": pv,
            "image_grid_thw": grid,
        }
        Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)

        for i, c in enumerate(base_model.model.get_image_features.call_args_list):
            passed_grid = c[0][1]
            assert passed_grid.shape == (1, 3), \
                f"이미지 {i}: grid_thw shape {passed_grid.shape} ≠ (1,3)"


# ---------------------------------------------------------------------------
# 3. Qwen3.5 — 비디오 tile_offset
# ---------------------------------------------------------------------------

class TestQwenPostEncodeVideo:

    def _run_video(self, fake_self, base_model, pixel_values_videos, video_grid_thw):
        inputs = {
            "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)

    def test_single_video_single_tile(self):
        """비디오 1개, 1타일 → pixel_values_videos[0:1] 전달."""
        n_tokens = 2 * (PPS // 2) ** 2   # t=2
        fake_self, base_model = make_qwen_mocks(n_vid_tokens=n_tokens)
        vid_mask = _bool_mask_3d(n_tokens)
        base_model.model.get_placeholder_mask.return_value = (None, vid_mask)

        pv = torch.zeros(1, 4, 3, PPS, PPS)
        grid = torch.tensor([[2, PPS, PPS]])

        self._run_video(fake_self, base_model, pv, grid)

        args = base_model.model.get_video_features.call_args_list
        assert len(args) == 1
        assert args[0][0][0].shape[0] == 1

    def test_two_videos_tile_offset(self):
        """비디오 2개: 첫 번째 1타일, 두 번째 2타일."""
        n1 = 2 * (PPS // 2) ** 2
        n2 = 2 * (PPS // 2) * (2 * PPS // 2)

        fake_self, base_model = make_qwen_mocks(n_vid_tokens=n1)
        base_model.model.get_video_features.side_effect = [
            _img_output(n1), _img_output(n2)
        ]
        total_mask = _bool_mask_3d(n1 + n2)
        base_model.model.get_placeholder_mask.return_value = (None, total_mask)

        pv = torch.zeros(3, 4, 3, PPS, PPS)
        grid = torch.tensor([[2, PPS, PPS], [2, PPS, 2 * PPS]])

        self._run_video(fake_self, base_model, pv, grid)

        calls = base_model.model.get_video_features.call_args_list
        assert calls[0][0][0].shape[0] == 1
        assert calls[1][0][0].shape[0] == 2


# ---------------------------------------------------------------------------
# 4. Qwen3.5 — 반환 구조
# ---------------------------------------------------------------------------

class TestQwenPostEncodeOutput:

    def test_returns_inputs_embeds_dict(self):
        """반환값이 {'inputs_embeds': tensor} 형태인지."""
        fake_self, _ = make_qwen_mocks()
        inputs = {"input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long)}
        result = Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)
        assert set(result.keys()) == {"inputs_embeds"}
        assert isinstance(result["inputs_embeds"], torch.Tensor)

    def test_inputs_embeds_shape_preserved(self):
        """시각 입력 없을 때 embed_tokens 결과 shape 유지."""
        fake_self, base_model = make_qwen_mocks()
        inputs = {"input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long)}
        result = Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)
        assert result["inputs_embeds"].shape == (1, SEQ_LEN, EMBED_DIM)

    def test_image_embeds_injected_shape(self):
        """이미지 embeds 주입 후 inputs_embeds shape 불변."""
        n_tokens = (PPS // 2) ** 2
        fake_self, base_model = make_qwen_mocks(n_img_tokens=n_tokens)
        pv = torch.zeros(1, 2, 3, PPS, PPS)
        grid = torch.tensor([[1, PPS, PPS]])
        inputs = {
            "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
            "pixel_values": pv,
            "image_grid_thw": grid,
        }
        result = Qwen3_5VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)
        assert result["inputs_embeds"].shape == (1, SEQ_LEN, EMBED_DIM)


# ---------------------------------------------------------------------------
# 5. Gemma4 — is_training=False 조기 반환
# ---------------------------------------------------------------------------

class TestGemma4PostEncodeInference:

    def test_returns_inputs_unchanged_when_not_training(self):
        fake_self = MagicMock()
        fake_self.is_training = False
        sentinel = {"input_ids": torch.zeros(1, 5)}
        result = Gemma4VJEPATemplate._post_encode(fake_self, MagicMock(), sentinel)
        assert result is sentinel


# ---------------------------------------------------------------------------
# 6. Gemma4 — tile_offset 계산 (이미지)
# ---------------------------------------------------------------------------

class TestGemma4PostEncodeTileOffset:

    def _run(self, fake_self, base_model, pixel_values, image_grid_thw):
        inputs = {
            "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        Gemma4VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)

    def test_single_image_single_tile(self):
        n_tokens = (PPS // 2) ** 2
        fake_self, base_model = make_gemma4_mocks(n_img_tokens=n_tokens)
        pv = torch.zeros(1, 2, 3, PPS, PPS)
        grid = torch.tensor([[1, PPS, PPS]])

        self._run(fake_self, base_model, pv, grid)

        args = base_model.get_image_features.call_args_list
        assert len(args) == 1
        assert args[0][0][0].shape[0] == 1

    def test_two_images_tile_offset(self):
        """2장: 첫 번째 1타일, 두 번째 2타일 → 올바른 슬라이스 전달."""
        n1 = (PPS // 2) ** 2
        n2 = (PPS // 2) * (2 * PPS // 2)

        fake_self, base_model = make_gemma4_mocks(n_img_tokens=n1)
        base_model.get_image_features.side_effect = [_img_output(n1), _img_output(n2)]

        combined_mask = _bool_mask_2d(n1 + n2)
        audio_mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
        base_model.model.get_placeholder_mask.return_value = (
            combined_mask, torch.zeros(1, SEQ_LEN, dtype=torch.bool), audio_mask
        )

        pv = torch.zeros(3, 2, 3, PPS, PPS)
        grid = torch.tensor([[1, PPS, PPS], [1, PPS, 2 * PPS]])

        self._run(fake_self, base_model, pv, grid)

        calls = base_model.get_image_features.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0].shape[0] == 1
        assert calls[1][0][0].shape[0] == 2


# ---------------------------------------------------------------------------
# 7. Gemma4 — OOM fix (_vjepa_llm_input_ids 캐싱)
# ---------------------------------------------------------------------------

class TestGemma4OOMFixCaching:
    """
    Gemma4VJEPATemplate._post_encode 내에서
    hidden_size_per_layer_input 가 설정된 경우 _vjepa_llm_input_ids 를
    language_model 에 캐싱하는지 검증.
    """

    def test_caches_llm_input_ids_when_hidden_size_per_layer_input_set(self):
        fake_self, base_model = make_gemma4_mocks()
        pad_id = 0
        base_model.model.config.text_config.hidden_size_per_layer_input = 64
        base_model.model.config.text_config.pad_token_id = pad_id

        # get_placeholder_mask (OOM fix 분기용): (image_mask_2d, video_mask_2d, audio_mask_2d)
        seq_len = SEQ_LEN
        img_mask = _bool_mask_2d(2, seq_len)   # 위치 0,1 에 이미지
        vid_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        audio_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        base_model.model.get_placeholder_mask.return_value = (img_mask, vid_mask, audio_mask)

        input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        inputs = {"input_ids": input_ids}

        Gemma4VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)

        # language_model 에 _vjepa_llm_input_ids 가 설정되어야 함
        lm = base_model.model.language_model
        assert hasattr(lm, "_vjepa_llm_input_ids"), \
            "_vjepa_llm_input_ids 가 language_model 에 캐싱되지 않았습니다"
        cached = lm._vjepa_llm_input_ids
        # 멀티모달 위치(img_mask True)는 pad_id 로 교체
        assert cached[0, 0].item() == pad_id
        assert cached[0, 1].item() == pad_id
        # 나머지는 원래 값 유지
        assert cached[0, 2].item() == input_ids[0, 2].item()

    def test_no_caching_when_hidden_size_per_layer_input_not_set(self):
        """hidden_size_per_layer_input 없으면 _vjepa_llm_input_ids 미설정."""
        fake_self, base_model = make_gemma4_mocks()
        base_model.model.config.text_config.hidden_size_per_layer_input = None

        # MagicMock 은 존재하지 않는 속성도 hasattr()=True 를 반환하므로
        # 일반 객체로 교체해서 실제로 setattr 됐는지 확인한다.
        class _FakeLM:
            pass
        lm = _FakeLM()
        base_model.model.language_model = lm

        inputs = {"input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long)}
        Gemma4VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)

        assert not hasattr(lm, "_vjepa_llm_input_ids"), \
            "_vjepa_llm_input_ids 가 예기치 않게 설정되었습니다"

    def test_exception_in_oom_fix_does_not_propagate(self):
        """OOM fix 분기에서 예외 발생 시 _post_encode 가 중단되지 않음."""
        fake_self, base_model = make_gemma4_mocks()
        base_model.model.config.text_config.hidden_size_per_layer_input = 64
        # get_placeholder_mask 에서 예외 발생 유도
        base_model.model.get_placeholder_mask.side_effect = [
            RuntimeError("의도된 오류"),       # OOM fix 분기 호출
        ]

        inputs = {"input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long)}
        # 예외가 전파되지 않고 정상 반환
        result = Gemma4VJEPATemplate._post_encode(fake_self, MagicMock(), inputs)
        assert "inputs_embeds" in result


# ---------------------------------------------------------------------------
# 8. Gemma4 — _data_collator: image_grid_thw / video_grid_thw 수집
# ---------------------------------------------------------------------------

class TestGemma4DataCollator:
    """
    _data_collator 가 image_grid_thw / video_grid_thw 를
    batch 에서 torch.cat 으로 합산하는지 검증.
    """

    def _call(self, batch):
        fake_self = MagicMock(spec=Gemma4VJEPATemplate)
        with patch.object(Gemma4Template, "_data_collator", return_value={}):
            return Gemma4VJEPATemplate._data_collator(fake_self, batch)

    def test_image_grid_thw_concatenated(self):
        """두 샘플의 image_grid_thw 가 cat 되어 result 에 포함."""
        batch = [
            {"image_grid_thw": torch.tensor([[1, 16, 16]])},
            {"image_grid_thw": torch.tensor([[1, 16, 32]])},
        ]
        result = self._call(batch)
        assert "image_grid_thw" in result
        assert result["image_grid_thw"].shape == (2, 3)
        assert result["image_grid_thw"].tolist() == [[1, 16, 16], [1, 16, 32]]

    def test_video_grid_thw_concatenated(self):
        batch = [
            {"video_grid_thw": torch.tensor([[2, 16, 16]])},
            {"video_grid_thw": torch.tensor([[4, 24, 24]])},
        ]
        result = self._call(batch)
        assert "video_grid_thw" in result
        assert result["video_grid_thw"].shape == (2, 3)

    def test_missing_grid_thw_excluded(self):
        """grid_thw 없는 샘플만 있으면 key 가 result 에 없음."""
        batch = [{"other_key": torch.zeros(1)}]
        result = self._call(batch)
        assert "image_grid_thw" not in result
        assert "video_grid_thw" not in result

    def test_partial_batch_mixed_presence(self):
        """일부 샘플에만 image_grid_thw 있어도 있는 것만 cat."""
        batch = [
            {"image_grid_thw": torch.tensor([[1, 16, 16]])},
            {},  # image_grid_thw 없음
            {"image_grid_thw": torch.tensor([[1, 24, 24]])},
        ]
        result = self._call(batch)
        assert result["image_grid_thw"].shape == (2, 3)
