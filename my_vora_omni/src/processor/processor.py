import os
import math
import torch
import numpy as np
from PIL import Image

from transformers import (
    Qwen2VLImageProcessor,
    Qwen3VLVideoProcessor,
    Qwen3VLProcessor,
    Gemma4ImageProcessor, 
    Gemma4VideoProcessor, 
    Gemma4Processor,
    AutoConfig,
)
from transformers.image_utils import SizeDict
from transformers.image_processing_utils import BatchFeature


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


def _select_tile_layout(orig_h: int, orig_w: int, max_tiles: int):
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

class VJEPAImageMixin:
    def _vjepa_preprocess_images(self, images, **kwargs):
        max_tiles = int(os.environ.get("IMAGE_MAX_TILES", "4"))
        merge = getattr(self, "merge_size", 1)

        h_patch = self.image_size // self.patch_size
        rescale_factor = kwargs.get("rescale_factor", 1 / 255.0)
        do_rescale = kwargs.get("do_rescale", True)
        image_mean = kwargs.get("image_mean") or self.image_mean
        image_std = kwargs.get("image_std") or self.image_std

        all_tiles = []
        all_grid_thw = []
        all_tokens = []

        for img in images:
            _, orig_h, orig_w = img.shape
            n_rows, n_cols = _select_tile_layout(orig_h, orig_w, max_tiles)
            n_tiles = n_rows * n_cols

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
            img_norm = img_batch.squeeze(0)

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

class Qwen3VJEPAImageProcessor(VJEPAImageMixin, Qwen2VLImageProcessor):
    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs["merge_size"] = cfg.MERGE_SIZE
        kwargs["patch_size"] = cfg.patch_size
        kwargs["image_mean"] = cfg.MEAN
        kwargs["image_std"] = cfg.STD
        kwargs["size"] = {
            "longest_edge": cfg.image_size**2,
            "shortest_edge": cfg.image_size**2,
        }
        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size

    def _preprocess(
        self, images, do_resize=None, size=None, disable_grouping=None, **kwargs
    ):
        return self._vjepa_preprocess_images(images, **kwargs)

class VJEPAVideoMixin:
    def _vjepa_preprocess_videos(self, videos, **kwargs):
        max_tiles = int(os.environ.get("IMAGE_MAX_TILES", "4"))
        merge = getattr(self, "merge_size", 1)
        
        h_patch = self.image_size // self.patch_size
        rescale_factor = kwargs.get("rescale_factor", 1 / 255.0)
        do_rescale = kwargs.get("do_rescale", True)
        image_mean = kwargs.get("image_mean") or self.image_mean
        image_std = kwargs.get("image_std") or self.image_std

        all_tiles = []
        all_grid_thw = []
        all_tokens = []
        
        for vid in videos:
            T, C, orig_h, orig_w = vid.shape

            if T % self.tubelet_size != 0:
                pad = self.tubelet_size - (T % self.tubelet_size)
                vid = torch.cat([vid, vid[-1:].expand(pad, -1, -1, -1)], dim=0)
                T = vid.shape[0]

            n_rows, n_cols = _select_tile_layout(orig_h, orig_w, max_tiles)
            n_tiles = n_rows * n_cols
            frames = self.resize(
                vid,
                SizeDict(
                    height=n_rows * self.image_size,
                    width=n_cols * self.image_size
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
            frames = (frames - mean) / std

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

class Qwen3VJEPAVideoProcessor(VJEPAVideoMixin, Qwen3VLVideoProcessor):
    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs["merge_size"] = cfg.MERGE_SIZE
        kwargs["patch_size"] = cfg.patch_size
        kwargs["image_mean"] = cfg.MEAN
        kwargs["image_std"] = cfg.STD
        kwargs["size"] = {
            "longest_edge": cfg.image_size**2,
            "shortest_edge": cfg.image_size**2,
        }
        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size

        self.max_frames = int(os.environ.get("FPS_MAX_FRAMES", "16"))
        self.max_frames = (self.max_frames // self.tubelet_size) * self.tubelet_size

    def _preprocess(self, videos, do_resize=None, size=None, **kwargs):
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
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        processor.image_processor = Qwen3VJEPAImageProcessor(vision_model_id=cls.VISION_MODEL_ID, **kwargs)
        processor.video_processor = Qwen3VJEPAVideoProcessor(vision_model_id=cls.VISION_MODEL_ID, **kwargs)
        
        if getattr(processor, 'chat_template', None) is None:
            jinja_path = os.path.join(pretrained_model_name_or_path, 'chat_template.jinja')
            if os.path.exists(jinja_path):
                with open(jinja_path) as f:
                    processor.chat_template = f.read()

        return processor



class Gemma4VJEPAImageProcessor(VJEPAImageMixin, Gemma4ImageProcessor):
    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs["patch_size"] = cfg.patch_size
        kwargs["image_mean"] = cfg.MEAN
        kwargs["image_std"] = cfg.STD
        # kwargs.setdefault("max_soft_tokens", 70)
        
        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.image_size = cfg.image_size
        self.merge_size = cfg.MERGE_SIZE

    def _preprocess(self, images, **kwargs):
        return self._vjepa_preprocess_images(images, **kwargs)

class Gemma4VJEPAVideoProcessor(VJEPAVideoMixin, Gemma4VideoProcessor):
    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs["patch_size"] = cfg.patch_size
        kwargs["image_mean"] = cfg.MEAN
        kwargs["image_std"] = cfg.STD
        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.image_size = cfg.image_size
        self.merge_size = cfg.MERGE_SIZE

        self.max_frames = int(os.environ.get("FPS_MAX_FRAMES", "16"))
        self.max_frames = (self.max_frames // self.tubelet_size) * self.tubelet_size
        self.num_frames = max(self.tubelet_size, self.max_frames)
        
    def preprocess(self, videos, **kwargs):
        result = super().preprocess(videos, **kwargs)
        if "video_metadata" in result:
            for meta in result["video_metadata"]:
                meta.fps = 1
                if meta.frames_indices is not None and len(meta.frames_indices) > 0:
                    meta.frames_indices = [meta.frames_indices[0]]
        return result

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

        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            image_seq_length=image_seq_length,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        processor.image_processor = Gemma4VJEPAImageProcessor(vision_model_id=cls.VISION_MODEL_ID, **kwargs)
        processor.video_processor = Gemma4VJEPAVideoProcessor(vision_model_id=cls.VISION_MODEL_ID, **kwargs)
        
        if getattr(processor, 'chat_template', None) is None:
            jinja_path = os.path.join(pretrained_model_name_or_path, 'chat_template.jinja')
            if os.path.exists(jinja_path):
                with open(jinja_path) as f:
                    processor.chat_template = f.read()
                    
        return processor

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