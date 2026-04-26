import os

from transformers import Lfm2VlImageProcessor, Lfm2VlProcessor
from transformers.image_processing_utils import BatchFeature

from .processor_base import VoRAVisionConfig, VJEPAImageMixin, VJEPAVideoMixin


class Lfm2VJEPAImageProcessor(VJEPAImageMixin, Lfm2VlImageProcessor):
    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs["encoder_patch_size"] = cfg.patch_size
        kwargs["image_mean"] = cfg.MEAN
        kwargs["image_std"] = cfg.STD

        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size
        self.merge_size = cfg.MERGE_SIZE

    def _preprocess(self, images, **kwargs):
        return self._vjepa_preprocess_images(images, **kwargs)


class Lfm2VJEPAVideoProcessor(VJEPAVideoMixin, Lfm2VlImageProcessor):
    """
    LFM2-VL has no native video processor class. We inherit from Lfm2VlImageProcessor
    solely to get the TorchvisionBackend.resize() method used by VJEPAVideoMixin.
    """

    def __init__(
        self, vision_model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kwargs
    ):
        cfg = VoRAVisionConfig(vision_model_id)

        kwargs["encoder_patch_size"] = cfg.patch_size
        kwargs["image_mean"] = cfg.MEAN
        kwargs["image_std"] = cfg.STD

        super().__init__(**kwargs)
        self.tubelet_size = cfg.tubelet_size
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size
        self.merge_size = cfg.MERGE_SIZE

        self.max_frames = int(os.environ.get("FPS_MAX_FRAMES", "16"))
        self.max_frames = (self.max_frames // self.tubelet_size) * self.tubelet_size

    def _preprocess(self, videos, **kwargs):
        return self._vjepa_preprocess_videos(videos, **kwargs)


class Lfm2VLVJEPAProcessor(Lfm2VlProcessor):
    VISION_MODEL_ID = None

    def __call__(self, images=None, videos=None, text=None, **kwargs):
        """
        Override Lfm2VlProcessor.__call__ to bypass LFM2's native text-expansion logic
        (which expects image_rows/cols/sizes not produced by VJEPA preprocessing).
        The VoRA template handles visual token injection instead.
        """
        inputs = {}

        if images is not None:
            if not isinstance(images, list):
                images = [images]
            flat_images = (
                images
                if not isinstance(images[0], list)
                else [img for batch in images for img in batch]
            )
            vision_inputs = self.image_processor._vjepa_preprocess_images(
                flat_images, **kwargs
            )
            inputs.update(vision_inputs.data)

        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]
            vision_inputs = self.video_processor._vjepa_preprocess_videos(
                videos, **kwargs
            )
            inputs.update(vision_inputs.data)

        if text is not None:
            if isinstance(text, str):
                text = [text]
            text_inputs = self.tokenizer(text, add_special_tokens=False)
            inputs.update(text_inputs)

        return_tensors = kwargs.get("return_tensors", None)
        return BatchFeature(inputs, tensor_type=return_tensors)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        processor = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        processor.image_processor = Lfm2VJEPAImageProcessor(
            vision_model_id=cls.VISION_MODEL_ID
        )
        processor.video_processor = Lfm2VJEPAVideoProcessor(
            vision_model_id=cls.VISION_MODEL_ID
        )

        if getattr(processor, "chat_template", None) is None:
            jinja_path = os.path.join(
                pretrained_model_name_or_path, "chat_template.jinja"
            )
            if os.path.exists(jinja_path):
                with open(jinja_path) as f:
                    processor.chat_template = f.read()

        return processor


class Lfm2VLVJepa2LProcessor(Lfm2VLVJEPAProcessor):
    VISION_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class Lfm2VLVJepa2GProcessor(Lfm2VLVJEPAProcessor):
    VISION_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class Lfm2VLVJEPA21BProcessor(Lfm2VLVJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_base_384"


class Lfm2VLVJEPA21LProcessor(Lfm2VLVJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_large_384"


class Lfm2VLVJEPA21GProcessor(Lfm2VLVJEPAProcessor):
    VISION_MODEL_ID = "vjepa2_1_vit_giant_384"
