import os
from transformers import (
    Qwen2VLImageProcessor,
    Qwen3VLVideoProcessor,
    Qwen3VLProcessor,
)
from .processor_base import VoRAVisionConfig, VJEPAImageMixin, VJEPAVideoMixin


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
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        processor = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        processor.image_processor = Qwen3VJEPAImageProcessor(
            vision_model_id=cls.VISION_MODEL_ID, **kwargs
        )
        processor.video_processor = Qwen3VJEPAVideoProcessor(
            vision_model_id=cls.VISION_MODEL_ID, **kwargs
        )

        if getattr(processor, "chat_template", None) is None:
            jinja_path = os.path.join(
                pretrained_model_name_or_path, "chat_template.jinja"
            )
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
