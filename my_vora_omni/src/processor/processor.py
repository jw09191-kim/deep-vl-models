import os
from transformers import (
    Qwen2VLImageProcessorFast,
    Qwen3VLVideoProcessor,
    Qwen3VLProcessor,
    AutoConfig,
)
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
from transformers.image_utils import SizeDict
from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import group_images_by_shape, reorder_images
from transformers.video_utils import group_videos_by_shape, reorder_videos

import torch


VJEPA21_CONFIGS = {
    "vjepa2_1_vit_base_384":    dict(image_size=384, patch_size=16, tubelet_size=2, hidden_size=768),
    "vjepa2_1_vit_large_384":   dict(image_size=384, patch_size=16, tubelet_size=2, hidden_size=1024),
    "vjepa2_1_vit_giant_384":   dict(image_size=384, patch_size=16, tubelet_size=2, hidden_size=1408),
}

class VoRAVisionConfig:
    MEAN       = [0.485, 0.456, 0.406]
    STD        = [0.229, 0.224, 0.225]
    MERGE_SIZE = 2

    def __init__(self, vision_model_id: str):
        if vision_model_id in VJEPA21_CONFIGS:
            cfg = VJEPA21_CONFIGS[vision_model_id]
            self.vision_model_id  = vision_model_id
            self.image_size       = cfg["image_size"]
            self.patch_size       = cfg["patch_size"]
            self.tubelet_size     = cfg["tubelet_size"]
            self.hidden_size      = cfg["hidden_size"]
        else:
            cfg = AutoConfig.from_pretrained(vision_model_id)
            self.vision_model_id  = vision_model_id
            self.image_size       = cfg.image_size
            self.patch_size       = cfg.patch_size
            self.tubelet_size     = cfg.tubelet_size
            self.hidden_size      = cfg.hidden_size
            self.tokens_per_image = (self.image_size // self.patch_size) ** 2

VJEPA2L_CFG = VoRAVisionConfig("facebook/vjepa2-vitl-fpc64-256")
VJEPA2G_CFG = VoRAVisionConfig("facebook/vjepa2-vitg-fpc64-256")


class VJEPAImageProcessor(Qwen2VLImageProcessorFast):
    def __init__(self, vision_model_id: str="facebook/vjepa2-vitl-fpc64-256", **kwargs):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs.setdefault("merge_size", cfg.MERGE_SIZE)
        kwargs.setdefault("patch_size", cfg.patch_size)
        kwargs.setdefault("image_mean", cfg.MEAN)
        kwargs.setdefault("image_std",  cfg.STD)
        kwargs.setdefault("size", {
            "longest_edge":  cfg.image_size ** 2,
            "shortest_edge": cfg.image_size ** 2,
        })
        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.patch_size   = cfg.patch_size
        self.image_size   = cfg.image_size

    def _preprocess(self, images, do_resize, size, disable_grouping=None, **kwargs):
        grouped_images, indices = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_groups = {}

        for shape, stacked_images in grouped_images.items():
            stacked_images = self.resize(stacked_images, SizeDict(
                height=self.image_size, width=self.image_size,
            ))
            if kwargs.get("do_rescale", True):
                stacked_images = stacked_images * kwargs.get("rescale_factor", 1/255.0)

            mean = torch.tensor(kwargs.get("image_mean"), dtype=stacked_images.dtype, device=stacked_images.device).view(1, 3, 1, 1)
            std  = torch.tensor(kwargs.get("image_std"),  dtype=stacked_images.dtype, device=stacked_images.device).view(1, 3, 1, 1)
            stacked_images = (stacked_images - mean) / std
            processed_groups[shape] = stacked_images

        processed_images = reorder_images(processed_groups, indices)
        pixel_values = torch.stack(processed_images).unsqueeze(1).repeat(1, self.tubelet_size, 1, 1, 1)

        B  = pixel_values.shape[0]
        h  = w = self.image_size // self.patch_size  # 16

        image_grid_thw = torch.tensor([[1, h, w]] * B, dtype=torch.long)

        merge = getattr(self, 'merge_size', 2)
        tokens_per_image = (self.image_size // self.patch_size // merge) ** 2
        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "num_soft_tokens_per_image": [tokens_per_image] * B,
            },
            tensor_type=kwargs.get("return_tensors", None),
        )


class VJEPAVideoProcessor(Qwen3VLVideoProcessor):
    def __init__(self, vision_model_id: str="facebook/vjepa2-vitl-fpc64-256", **kwargs):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs.setdefault("merge_size", cfg.MERGE_SIZE)
        kwargs.setdefault("patch_size", cfg.patch_size)
        kwargs.setdefault("image_mean", cfg.MEAN)
        kwargs.setdefault("image_std",  cfg.STD)
        kwargs.setdefault("size", {
            "longest_edge":  cfg.image_size ** 2,
            "shortest_edge": cfg.image_size ** 2,
        })
        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.patch_size   = cfg.patch_size
        self.image_size   = cfg.image_size

        self.max_frames = int(os.environ.get("FPS_MAX_FRAMES", "16"))
        self.max_frames = (self.max_frames // self.tubelet_size) * self.tubelet_size

    def _preprocess(self, videos, do_resize, size, **kwargs):
        grouped_videos, indices = group_videos_by_shape(videos)
        processed_groups = {}

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape

            stacked_videos = stacked_videos.view(B * T, C, H, W)
            stacked_videos = self.resize(stacked_videos, SizeDict(height=self.image_size, width=self.image_size))
            stacked_videos = stacked_videos.view(B, T, C, self.image_size, self.image_size)

            if kwargs.get("do_rescale", True):
                stacked_videos = stacked_videos * kwargs.get("rescale_factor", 1/255.0)

            mean = torch.tensor(kwargs.get("image_mean"), dtype=stacked_videos.dtype, device=stacked_videos.device).view(1, 1, 3, 1, 1)
            std  = torch.tensor(kwargs.get("image_std"),  dtype=stacked_videos.dtype, device=stacked_videos.device).view(1, 1, 3, 1, 1)
            stacked_videos = (stacked_videos - mean) / std
            processed_groups[shape] = stacked_videos

        processed_videos = reorder_videos(processed_groups, indices)
        pixel_values_videos = torch.stack(processed_videos)

        B, T = pixel_values_videos.shape[:2]
        h = w  = self.image_size // self.patch_size
        grid_t = T // self.tubelet_size

        video_grid_thw = torch.tensor([[grid_t, h, w]] * B, dtype=torch.long)

        merge = getattr(self, 'merge_size', 2)
        tokens_per_frame = (self.image_size // self.patch_size // merge) ** 2
        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw":      video_grid_thw,
                "num_soft_tokens_per_video": [grid_t * tokens_per_frame] * B,
            },
            tensor_type=kwargs.get("return_tensors", None),
        )


class Qwen3VLVJEPAProcessor(Qwen3VLProcessor):
    VISION_MODEL_ID = None
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
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
# Gemma-4 Processors
# ──────────────────────────────────────────────

class Gemma4VJEPAProcessor(Gemma4Processor):
    VISION_MODEL_ID = None

    def __init__(self, feature_extractor=None, image_processor=None, tokenizer=None,
                 video_processor=None, chat_template=None, **kwargs):
        cfg = VoRAVisionConfig(self.VISION_MODEL_ID)
        image_seq_length = (cfg.image_size // cfg.patch_size // cfg.MERGE_SIZE) ** 2

        # Always replace with VJEPA processors
        image_processor = VJEPAImageProcessor(self.VISION_MODEL_ID)
        video_processor = VJEPAVideoProcessor(self.VISION_MODEL_ID)

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
