import os

from transformers import (
    Gemma4ImageProcessor,
    Gemma4VideoProcessor,
    Gemma4Processor,
)
from .processor_base import VoRAVisionConfig, VJEPAImageMixin, VJEPAVideoMixin


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
        processor = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        processor.image_processor = Gemma4VJEPAImageProcessor(
            vision_model_id=cls.VISION_MODEL_ID, **kwargs
        )
        processor.video_processor = Gemma4VJEPAVideoProcessor(
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
