import os
import math
from transformers import (
    Qwen2VLImageProcessorFast,
    Qwen3VLVideoProcessor,
    Qwen3VLProcessor,
    AutoConfig,
)
from transformers import Gemma4ImageProcessor, Gemma4VideoProcessor, Gemma4Processor
from transformers.image_utils import SizeDict
from transformers.image_processing_utils import BatchFeature

import torch


VJEPA21_CONFIGS = {
    "vjepa2_1_vit_base_384": dict(
        image_size=384, patch_size=16, tubelet_size=2, hidden_size=768
    ),
    "vjepa2_1_vit_large_384": dict(
        image_size=384, patch_size=16, tubelet_size=2, hidden_size=1024
    ),
    "vjepa2_1_vit_giant_384": dict(
        image_size=384, patch_size=16, tubelet_size=2, hidden_size=1408
    ),
}


class VoRAVisionConfig:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    MERGE_SIZE = 2

    def __init__(self, vision_model_id: str):
        if vision_model_id in VJEPA21_CONFIGS:
            cfg = VJEPA21_CONFIGS[vision_model_id]
            self.vision_model_id = vision_model_id
            self.image_size = cfg["image_size"]
            self.patch_size = cfg["patch_size"]
            self.tubelet_size = cfg["tubelet_size"]
            self.hidden_size = cfg["hidden_size"]
        else:
            cfg = AutoConfig.from_pretrained(vision_model_id)
            self.vision_model_id = vision_model_id
            self.image_size = cfg.image_size
            self.patch_size = cfg.patch_size
            self.tubelet_size = cfg.tubelet_size
            self.hidden_size = cfg.hidden_size
            self.tokens_per_image = (self.image_size // self.patch_size) ** 2


VJEPA2L_CFG = VoRAVisionConfig("facebook/vjepa2-vitl-fpc64-256")
VJEPA2G_CFG = VoRAVisionConfig("facebook/vjepa2-vitg-fpc64-256")


def _select_tile_layout(orig_h: int, orig_w: int, max_tiles: int):
    """가로세로 비율에 가장 가깝고 n_rows*n_cols <= max_tiles 인 (n_rows, n_cols) 반환."""
    aspect = orig_w / orig_h
    best, best_score = (1, 1), float("inf")
    for n_rows in range(1, max_tiles + 1):
        for n_cols in range(1, max_tiles + 1):
            if n_rows * n_cols > max_tiles:
                continue
            score = abs(math.log(aspect / (n_cols / n_rows)))
            if score < best_score:
                best_score = score
                best = (n_rows, n_cols)
    return best


# ──────────────────────────────────────────────
# Qwen3.5 VJEPA Processors
# ──────────────────────────────────────────────


class VJEPAImageMixin:
    """VJEPA 타일 기반 이미지 전처리 로직을 제공하는 Mixin."""

    def _vjepa_preprocess_images(self, images, **kwargs):
        max_tiles = int(os.environ.get("IMAGE_MAX_TILES", "4"))
        merge = getattr(self, "merge_size", 2)
        h_patch = self.image_size // self.patch_size
        rescale_factor = kwargs.get("rescale_factor", 1 / 255.0)
        do_rescale = kwargs.get("do_rescale", True)
        image_mean = kwargs.get("image_mean") or self.image_mean
        image_std = kwargs.get("image_std") or self.image_std

        all_tiles = []
        all_grid_thw = []
        all_tokens = []

        for img in images:
            # img: [C, H, W]
            _, orig_h, orig_w = img.shape
            n_rows, n_cols = _select_tile_layout(orig_h, orig_w, max_tiles)
            n_tiles = n_rows * n_cols

            # (n_rows × image_size) × (n_cols × image_size) 로 리사이즈
            img_batch = self.resize(
                img.unsqueeze(0),
                SizeDict(
                    height=n_rows * self.image_size,
                    width=n_cols * self.image_size,
                ),
            )  # [1, C, target_h, target_w]

            if do_rescale:
                img_batch = img_batch * rescale_factor

            mean = torch.tensor(
                image_mean, dtype=img_batch.dtype, device=img_batch.device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                image_std, dtype=img_batch.dtype, device=img_batch.device
            ).view(1, 3, 1, 1)
            img_batch = (img_batch - mean) / std
            img_norm = img_batch.squeeze(0)  # [C, target_h, target_w]

            # 타일 분할: [n_tiles, C, image_size, image_size]
            C = img_norm.shape[0]
            tiles = img_norm.view(C, n_rows, self.image_size, n_cols, self.image_size)
            tiles = (
                tiles.permute(1, 3, 0, 2, 4)
                .contiguous()
                .reshape(n_tiles, C, self.image_size, self.image_size)
            )
            # tubelet 차원 추가: [n_tiles, tubelet_size, C, H, W]
            tiles = tiles.unsqueeze(1).repeat(1, self.tubelet_size, 1, 1, 1)

            h_total = h_patch * n_rows
            w_total = h_patch * n_cols
            all_tiles.append(tiles)
            all_grid_thw.append([1, h_total, w_total])
            all_tokens.append((h_total // merge) * (w_total // merge))

        pixel_values = torch.cat(all_tiles, dim=0)  # [total_tiles, T, C, H, W]
        image_grid_thw = torch.tensor(all_grid_thw, dtype=torch.long)

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "num_soft_tokens_per_image": all_tokens,
            },
            tensor_type=kwargs.get("return_tensors", None),
        )


class VJEPAImageProcessor(VJEPAImageMixin, Qwen2VLImageProcessorFast):
    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs.setdefault("merge_size", cfg.MERGE_SIZE)
        kwargs.setdefault("patch_size", cfg.patch_size)
        kwargs.setdefault("image_mean", cfg.MEAN)
        kwargs.setdefault("image_std", cfg.STD)
        kwargs.setdefault(
            "size",
            {
                "longest_edge": cfg.image_size**2,
                "shortest_edge": cfg.image_size**2,
            },
        )
        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size

    def _preprocess(
        self, images, do_resize=None, size=None, disable_grouping=None, **kwargs
    ):  # noqa: ARG002
        return self._vjepa_preprocess_images(images, **kwargs)


class VJEPAVideoMixin:
    def _vjepa_preprocess_videos(self, videos, **kwargs):
        max_tiles = int(os.environ.get("VIDEO_MAX_TILES", "4"))
        merge = getattr(self, "merge_size", 2)
        h_patch = self.image_size // self.patch_size
        rescale_factor = kwargs.get("rescale_factor", 1 / 255.0)
        do_rescale = kwargs.get("do_rescale", True)
        image_mean = kwargs.get("image_mean") or self.image_mean
        image_std = kwargs.get("image_std") or self.image_std

        all_tiles = []
        all_grid_thw = []
        all_tokens = []

        for vid in videos:
            # vid: [T, C, H, W]
            T, C, orig_h, orig_w = vid.shape

            # T 가 tubelet_size 의 배수가 아니면 마지막 프레임으로 패딩
            # (grid_t = T // tubelet_size = 0 이 되는 것을 방지)
            if T % self.tubelet_size != 0:
                pad = self.tubelet_size - (T % self.tubelet_size)
                vid = torch.cat([vid, vid[-1:].expand(pad, -1, -1, -1)], dim=0)
                T = vid.shape[0]

            n_rows, n_cols = _select_tile_layout(orig_h, orig_w, max_tiles)
            n_tiles = n_rows * n_cols

            # 모든 프레임을 n_rows*384 × n_cols*384 로 리사이즈 (T가 배치 차원으로 동작)
            frames = self.resize(
                vid,
                SizeDict(
                    height=n_rows * self.image_size, width=n_cols * self.image_size
                ),
            )  # [T, C, target_h, target_w]

            if do_rescale:
                frames = frames * rescale_factor

            mean = torch.tensor(
                image_mean, dtype=frames.dtype, device=frames.device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                image_std, dtype=frames.dtype, device=frames.device
            ).view(1, 3, 1, 1)
            frames = (frames - mean) / std  # [T, C, target_h, target_w]

            # 공간 타일 분할: [n_tiles, T, C, 384, 384]
            tiles = frames.view(T, C, n_rows, self.image_size, n_cols, self.image_size)
            tiles = tiles.permute(
                2, 4, 0, 1, 3, 5
            ).contiguous()  # [n_rows, n_cols, T, C, 384, 384]
            tiles = tiles.reshape(n_tiles, T, C, self.image_size, self.image_size)

            grid_t = T // self.tubelet_size
            h_total = h_patch * n_rows
            w_total = h_patch * n_cols

            all_tiles.append(tiles)
            all_grid_thw.append([grid_t, h_total, w_total])
            all_tokens.append(grid_t * (h_total // merge) * (w_total // merge))

        pixel_values_videos = torch.cat(all_tiles, dim=0)
        video_grid_thw = torch.tensor(all_grid_thw, dtype=torch.long)

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
                "num_soft_tokens_per_video": all_tokens,
            },
            tensor_type=kwargs.get("return_tensors", None),
        )


class VJEPAVideoProcessor(VJEPAVideoMixin, Qwen3VLVideoProcessor):
    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs.setdefault("merge_size", cfg.MERGE_SIZE)
        kwargs.setdefault("patch_size", cfg.patch_size)
        kwargs.setdefault("image_mean", cfg.MEAN)
        kwargs.setdefault("image_std", cfg.STD)
        kwargs.setdefault(
            "size",
            {
                "longest_edge": cfg.image_size**2,
                "shortest_edge": cfg.image_size**2,
            },
        )
        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size

        self.max_frames = int(os.environ.get("FPS_MAX_FRAMES", "16"))
        self.max_frames = (self.max_frames // self.tubelet_size) * self.tubelet_size
        # Qwen3VLVideoProcessor 기본값 fps=2를 비활성화.
        # preprocess()가 fps와 num_frames를 동시에 바인딩해 ValueError가 발생하는 것을 방지.
        self.fps = None

    def _preprocess(self, videos, do_resize=None, size=None, **kwargs):  # noqa: ARG002
        return self._vjepa_preprocess_videos(videos, **kwargs)


class Qwen3VLVJEPAProcessor(Qwen3VLProcessor):
    VISION_MODEL_ID = None

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs
    ):
        image_processor = VJEPAImageProcessor(self.VISION_MODEL_ID)
        video_processor = VJEPAVideoProcessor(self.VISION_MODEL_ID)

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )


class Qwen3VLVJepa2LProcessor(Qwen3VLVJEPAProcessor):
    VISION_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class Qwen3VLVJepa2GProcessor(Qwen3VLVJEPAProcessor):
    VISION_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class Qwen3VLVJepa21BProcessor(Qwen3VLVJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_base_384"


class Qwen3VLVJepa21LProcessor(Qwen3VLVJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_large_384"


class Qwen3VLVJepa21GProcessor(Qwen3VLVJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_giant_384"


# ──────────────────────────────────────────────
# Gemma-4 VJEPA Processors
# ──────────────────────────────────────────────


class Gemma4VJEPAImageProcessor(VJEPAImageMixin, Gemma4ImageProcessor):
    """Gemma4 base를 사용하는 VJEPA 이미지 프로세서.
    _preprocess를 VJEPA 타일 방식으로 완전 override한다.
    출력: pixel_values [total_tiles, tubelet_size, C, H, W], image_grid_thw
    """

    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs.setdefault("image_mean", cfg.MEAN)
        kwargs.setdefault("image_std", cfg.STD)
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255.0)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("patch_size", cfg.patch_size)
        # Gemma4ImageProcessor.__init__이 max_soft_tokens 유효성 검사를 하므로
        # _SUPPORTED_SOFT_TOKENS 중 최솟값(70)을 전달 (실제 처리엔 미사용)
        kwargs.setdefault("max_soft_tokens", 70)

        super().__init__(**kwargs)

        # VJEPA 전용 속성 (super().__init__ 이후 설정)
        self.tubelet_size = cfg.tubelet_size
        self.image_size = cfg.image_size
        self.merge_size = cfg.MERGE_SIZE
        # self.patch_size는 kwargs를 통해 super().__init__에서 이미 설정됨

    def _preprocess(self, images, **kwargs):
        return self._vjepa_preprocess_images(images, **kwargs)


class Gemma4VJEPAVideoProcessor(VJEPAVideoMixin, Gemma4VideoProcessor):
    """Gemma4 base를 사용하는 VJEPA 비디오 프로세서.
    sample_frames를 override해 torchcodec metadata overcount를 방어하고,
    _preprocess를 VJEPA 타일 방식으로 완전 override한다.
    출력: pixel_values_videos [n_tiles, T, C, H, W], video_grid_thw
    """

    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs.setdefault("image_mean", cfg.MEAN)
        kwargs.setdefault("image_std", cfg.STD)
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255.0)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("patch_size", cfg.patch_size)
        kwargs.setdefault("max_soft_tokens", 70)

        super().__init__(**kwargs)

        # VJEPA 전용 속성 (super().__init__ 이후 설정)
        self.tubelet_size = cfg.tubelet_size
        self.image_size = cfg.image_size
        self.merge_size = cfg.MERGE_SIZE
        # self.patch_size는 kwargs를 통해 super().__init__에서 이미 설정됨

        # FPS_MAX_FRAMES를 tubelet_size 단위로 정렬하여 num_frames 설정
        # BaseVideoProcessor.sample_frames는 self.num_frames를 기본 프레임 수로 사용
        max_frames = int(os.environ.get("FPS_MAX_FRAMES", "16"))
        max_frames = (max_frames // self.tubelet_size) * self.tubelet_size
        self.num_frames = max(self.tubelet_size, max_frames)

    def _preprocess(self, videos, **kwargs):
        return self._vjepa_preprocess_videos(videos, **kwargs)


class Gemma4VJEPAProcessor(Gemma4Processor):
    VISION_MODEL_ID = None

    def __init__(
        self,
        feature_extractor=None,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs
    ):
        cfg = VoRAVisionConfig(self.VISION_MODEL_ID)
        image_seq_length = (cfg.image_size // cfg.patch_size // cfg.MERGE_SIZE) ** 2

        # Gemma4 전용 VJEPA processor 사용
        image_processor = Gemma4VJEPAImageProcessor(self.VISION_MODEL_ID)
        video_processor = Gemma4VJEPAVideoProcessor(self.VISION_MODEL_ID)

        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            image_seq_length=image_seq_length,
            **kwargs,
        )


class Gemma4VJepa2LProcessor(Gemma4VJEPAProcessor):
    VISION_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class Gemma4VJepa2GProcessor(Gemma4VJEPAProcessor):
    VISION_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class Gemma4VJEPA21BProcessor(Gemma4VJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_base_384"


class Gemma4VJEPA21LProcessor(Gemma4VJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_large_384"


class Gemma4VJEPA21GProcessor(Gemma4VJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_giant_384"
