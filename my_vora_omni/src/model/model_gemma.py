import os
import glob
from safetensors import safe_open

import torch

from transformers import AutoModel, Gemma4ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

from .model_base import VJEPA2VisualModule


class Gemma4VJEPAModel(Gemma4ForConditionalGeneration):
    VISION_MODEL_ID = None
    TORCH_HUB_NAME = None

    def __init__(self, config: Gemma4Config):
        config.vision_config.spatial_merge_size = 2
        super().__init__(config)

        self.model.visual = None
        self.model._current_image_grid_thw = None
        self.model._current_video_grid_thw = None

        self.model.get_image_features = self.get_image_features
        self.model.get_video_features = self.get_video_features

    def _validate_model_kwargs(self, model_kwargs):
        # image_grid_thw/video_grid_thw: stored in _current_*_grid_thw, not forwarded to base
        model_kwargs.pop("image_grid_thw", None)
        model_kwargs.pop("video_grid_thw", None)
        model_kwargs.pop("mm_token_type_ids", None)
        # num_soft_tokens_*: consumed by Gemma4Processor.__call__ before reaching here,
        # but pop defensively in case of direct model calls
        model_kwargs.pop("num_soft_tokens_per_image", None)
        model_kwargs.pop("num_soft_tokens_per_video", None)
        super()._validate_model_kwargs(model_kwargs)

    def forward(self, *args, image_grid_thw=None, video_grid_thw=None, **kwargs):
        self.model._current_image_grid_thw = image_grid_thw
        self.model._current_video_grid_thw = video_grid_thw

        outputs = super().forward(*args, **kwargs)
        if (
            hasattr(outputs, "logits")
            and outputs.logits is not None
            and outputs.logits.dim() == 4
        ):
            if isinstance(outputs, dict):
                outputs["logits"] = outputs["logits"].squeeze(1)
            else:
                outputs.logits = outputs.logits.squeeze(1)

        return outputs

    def get_image_features(self, pixel_values, image_position_ids=None, **kwargs):
        image_grid_thw = self.model._current_image_grid_thw

        merge_size = self.config.vision_config.spatial_merge_size
        pps = self.model.visual.patches_per_side
        ppt = pps * pps

        image_embeds = self.model.visual(pixel_values)
        results = []
        item_offset = 0

        for grid_idx in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[grid_idx].tolist()

            if t == 1:
                n_tiles = (h * w) // ppt
                if n_tiles == 1:
                    embeds = image_embeds[item_offset]
                else:
                    tiles = image_embeds[item_offset : item_offset + n_tiles]
                    n_rows = h // pps
                    n_cols = w // pps
                    grid = tiles.view(n_rows, n_cols, pps, pps, -1)
                    assembled = grid.permute(0, 2, 1, 3, 4).contiguous()
                    embeds = assembled.reshape(h * w, -1)
                item_offset += n_tiles
            else:
                n_tiles = (h * w) // ppt
                if n_tiles == 1:
                    embeds = image_embeds[item_offset].reshape(t * h * w, -1)
                else:
                    tiles = image_embeds[item_offset : item_offset + n_tiles]
                    n_rows = h // pps
                    n_cols = w // pps
                    tiles = tiles.reshape(n_rows, n_cols, t, pps, pps, -1)
                    tiles = tiles.permute(2, 0, 3, 1, 4, 5).contiguous()
                    embeds = tiles.reshape(t * h * w, -1)
                item_offset += n_tiles

            embeds = embeds.view(
                t, h // merge_size, merge_size, w // merge_size, merge_size, -1
            )
            embeds = embeds.permute(0, 1, 3, 2, 4, 5).contiguous()
            embeds = embeds.reshape(
                t,
                (h // merge_size) * (w // merge_size),
                merge_size**2 * embeds.shape[-1],
            )

            embeds = embeds.to(dtype=self.dtype, device=self.device)
            embeds = self.model.visual.merger(embeds)

            results.append(embeds.view(-1, embeds.shape[-1]))

        self.model._current_image_grid_thw = None
        final_embeds = torch.cat(results, dim=0)

        return BaseModelOutputWithPooling(pooler_output=final_embeds)

    def get_video_features(
        self, pixel_values_videos, video_position_ids=None, **kwargs
    ):
        video_grid_thw = self.model._current_video_grid_thw
        self.model._current_image_grid_thw = video_grid_thw

        output = self.get_image_features(
            pixel_values_videos, video_position_ids, **kwargs
        )
        self.model._current_video_grid_thw = None

        return output

    @classmethod
    def _load_hf_visual(cls, merge_size, llm_dim, device, dtype):
        vjepa2 = AutoModel.from_pretrained(cls.VISION_MODEL_ID)
        encoder = vjepa2.encoder
        patches_per_side = vjepa2.config.image_size // vjepa2.config.patch_size
        return VJEPA2VisualModule(
            encoder,
            vjepa2.config.hidden_size,
            merge_size,
            llm_dim,
            patches_per_side=patches_per_side,
            is_v21=False,
        ).to(device=device, dtype=dtype)

    @classmethod
    def _load_tf_visual(cls, merge_size, llm_dim, device, dtype):
        encoder, _ = torch.hub.load("facebookresearch/vjepa2", cls.TORCH_HUB_NAME)
        patches_per_side = 384 // 16

        vjepa_dim = encoder.img_mod_embed.shape[-1]

        return VJEPA2VisualModule(
            encoder,
            vjepa_dim,
            merge_size,
            llm_dim,
            patches_per_side=patches_per_side,
            is_v21=True,
        ).to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        merge_size = model.config.vision_config.spatial_merge_size
        llm_dim = model.config.text_config.hidden_size

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if getattr(cls, "VISION_MODEL_ID", None):
            visual = cls._load_hf_visual(merge_size, llm_dim, device, dtype)
        elif getattr(cls, "TORCH_HUB_NAME", None):
            visual = cls._load_tf_visual(merge_size, llm_dim, device, dtype)
        else:
            raise ValueError("Missing VISION_MODEL_ID or TORCH_HUB_NAME")

        safetensor_files = glob.glob(
            os.path.join(pretrained_model_name_or_path, "*.safetensors")
        )
        if safetensor_files:
            visual_state = {}
            for ckpt_file in safetensor_files:
                with safe_open(ckpt_file, framework="pt") as f:
                    for k in f.keys():
                        if k.startswith("model.visual."):
                            new_k = k.replace("model.visual.", "")
                            visual_state[new_k] = f.get_tensor(k).to(
                                device=device, dtype=dtype
                            )
                        elif k.startswith("model.language_model.visual."):
                            new_k = k.replace("model.language_model.visual.", "")
                            visual_state[new_k] = f.get_tensor(k).to(
                                device=device, dtype=dtype
                            )

            if visual_state:
                missing, unexpected = visual.load_state_dict(visual_state, strict=False)
                print(
                    f"Visual & Aligner weights loaded: {len(visual_state)} keys from {len(safetensor_files)} shard files"
                )

        model.model.visual = visual
        return model


class Gemma4VJEPALModel(Gemma4VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class Gemma4VJEPAGModel(Gemma4VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class Gemma4VJEPA21BModel(Gemma4VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_base_384"


class Gemma4VJEPA21LModel(Gemma4VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_large_384"


class Gemma4VJEPA21GModel(Gemma4VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_giant_384"
