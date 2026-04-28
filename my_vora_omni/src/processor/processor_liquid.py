import os
import torch
from transformers import Lfm2VlImageProcessor, Lfm2VlProcessor
from transformers.image_processing_utils import BatchFeature

from .processor_base import VoRAVisionConfig, VJEPAImageMixin, VJEPAVideoMixin

def _decode_video_to_tensor(path: str, max_frames: int = 32) -> torch.Tensor:
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    total = len(vr)
    indices = torch.linspace(0, total - 1, min(max_frames, total)).long().tolist()
    frames = vr.get_batch(indices)  # (T, H, W, C) uint8
    return frames.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)



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
        inputs = {}
        all_soft_tokens = []

        if videos is not None:
            if not isinstance(videos, list): videos = [videos]
            
            max_frames = int(os.environ.get("FPS_MAX_FRAMES", "32"))
            
            decoded_videos = []
            for v in videos:
                actual_v = v[0] if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str) else v
                
                if isinstance(actual_v, str):
                    decoded_videos.append(_decode_video_to_tensor(actual_v, max_frames))
                else:
                    decoded_videos.append(actual_v)
            
            vision_inputs = self.video_processor._vjepa_preprocess_videos(decoded_videos, **kwargs)
            inputs.update(vision_inputs.data)
            soft = vision_inputs['num_soft_tokens_per_video']
            all_soft_tokens.extend(soft.tolist() if isinstance(soft, torch.Tensor) else soft)

        if text is not None:
            if isinstance(text, str): text = [text]
            image_token_id = 396 # LFM2-VL 표준 ID
            
            expanded_input_ids = []
            for prompt in text:
                if "<image>" not in prompt:
                    prompt = "<image>" * len(all_soft_tokens) + prompt
                
                parts = prompt.split("<image>")
                final_ids = []
                for i, part in enumerate(parts):
                    final_ids.extend(self.tokenizer.encode(part, add_special_tokens=False))
                    if i < len(parts) - 1 and i < len(all_soft_tokens):
                        final_ids.extend([image_token_id] * int(all_soft_tokens[i]))
                expanded_input_ids.append(final_ids)

            inputs["input_ids"] = torch.tensor(expanded_input_ids)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        from transformers import BatchFeature
        return BatchFeature(inputs, tensor_type=kwargs.get("return_tensors", "pt"))

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
