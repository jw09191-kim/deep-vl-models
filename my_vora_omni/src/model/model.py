import os
from safetensors import safe_open

import torch
from torch import nn
from transformers import AutoModel, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from transformers import Gemma4ForConditionalGeneration


class Qwen3_5VJEPAInnerModel(Qwen3_5Model):
    
    def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
        image_embeds = self.visual(pixel_values)  # [B, N, D]
        merge_size = self.config.vision_config.spatial_merge_size
        results = []

        vid_idx = 0
        offset = 0
        for grid_idx in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[grid_idx].tolist()
            n = t * h * w

            embeds = image_embeds[vid_idx, offset:offset + n]
            embeds = embeds.view(t, h // merge_size, merge_size,
                                w // merge_size, merge_size, -1)
            embeds = embeds.permute(0, 1, 3, 2, 4, 5)
            embeds = embeds.reshape(t, (h // merge_size) * (w // merge_size),
                                    merge_size ** 2 * embeds.shape[-1])
            embeds = embeds.to(self.visual.dtype)
            embeds = self.visual.merger(embeds)
            results.append(embeds)

            offset += n
            if offset >= image_embeds.shape[1]:
                vid_idx += 1
                offset = 0

        return BaseModelOutputWithPooling(pooler_output=tuple(results))

    def get_video_features(self, pixel_values_videos, video_grid_thw, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)  

    def get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw=None, video_grid_thw=None, **kwargs):
        if video_grid_thw is not None:
            expanded = []
            for thw in video_grid_thw:
                t, h, w = thw.tolist()
                expanded.extend([[1, h, w]] * t)
            video_grid_thw = torch.tensor(expanded, dtype=torch.long, device=video_grid_thw.device)

        return super().get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
    
class VJEPA2VisualModule(nn.Module):
    def __init__(self, vjepa2_model, vjepa_dim, merge_size, llm_dim, is_v21=False):
        super().__init__()
        in_dim     = vjepa_dim * merge_size ** 2
        hidden_dim = vjepa_dim * merge_size ** 2
        mid_dim    = max(llm_dim, hidden_dim // 2)
        out_dim    = llm_dim

        self.encoder = vjepa2_model
        self.merger = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self._merge_size = merge_size
        self._is_v21 = is_v21

    @property
    def dtype(self):
        return next(self.merger.parameters()).dtype
    
    @property
    def spatial_merge_size(self):
        return self._merge_size

    def forward(self, pixel_values, **kwargs):
        if self._is_v21:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
            device_type = pixel_values.device.type
            with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16):
                out = self.encoder(pixel_values)
        else:
            out = self.encoder(pixel_values)
            out = out.last_hidden_state
        return out

class Qwen3_5VJEPAModel(Qwen3_5ForConditionalGeneration):
    VISION_MODEL_ID = None

    def __init__(self, config):
        config.vision_config.spatial_merge_size = 2
        super().__init__(config)
        self.model = Qwen3_5VJEPAInnerModel(config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # model.config.use_cache = True
        # model.config.text_config.use_cache = True
        # model.model.language_model.config.use_cache = True

        vjepa2     = AutoModel.from_pretrained(cls.VISION_MODEL_ID)
        vjepa_dim  = vjepa2.config.hidden_size
        merge_size = model.config.vision_config.spatial_merge_size
        llm_dim    = model.config.text_config.hidden_size

        device = next(model.model.language_model.parameters()).device
        dtype  = next(model.model.language_model.parameters()).dtype

        visual = VJEPA2VisualModule(vjepa2, vjepa_dim, merge_size, llm_dim).to(device=device, dtype=dtype)

        ckpt_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.exists(ckpt_file):
            visual_state = {}
            with safe_open(ckpt_file, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("model.visual."):
                        new_k = k.replace("model.visual.", "")
                        visual_state[new_k] = f.get_tensor(k).to(device=device, dtype=dtype)
            if visual_state:
                # strict=False: encoder weight는 VISION_MODEL_ID에서 이미 로드됨
                missing, unexpected = visual.load_state_dict(visual_state, strict=False)
                print(f"visual weights loaded: {len(visual_state)} keys")
                print(f"missing: {missing}")

        # VJEPA2 encoder freeze
        for param in visual.encoder.parameters():
            param.requires_grad = False

        model.model.visual = visual
        return model
        

class Qwen3_5VJEPALModel(Qwen3_5VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class Qwen3_5VJEPAGModel(Qwen3_5VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class Qwen3_5VJEPA21Model(Qwen3_5VJEPAModel):
    TORCH_HUB_NAME = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs)
        old_state = model.model.state_dict()
        model.config.vision_config.spatial_merge_size = 2
        model.model = Qwen3_5VJEPAInnerModel(model.config)
        model.model.load_state_dict(old_state, strict=True)
        
        encoder, _ = torch.hub.load(
            'facebookresearch/vjepa2',
            cls.TORCH_HUB_NAME
        )

        vjepa_dim  = encoder.img_mod_embed.shape[-1]
        merge_size = model.config.vision_config.spatial_merge_size
        llm_dim    = model.config.text_config.hidden_size

        device = next(model.model.language_model.parameters()).device
        dtype  = next(model.model.language_model.parameters()).dtype

        visual = VJEPA2VisualModule(encoder, vjepa_dim, merge_size, llm_dim, is_v21=True).to(device=device, dtype=dtype)

        ckpt_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.exists(ckpt_file):
            visual_state = {}
            with safe_open(ckpt_file, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("model.visual."):
                        new_k = k.replace("model.visual.", "")
                        visual_state[new_k] = f.get_tensor(k).to(device=device, dtype=dtype)
            if visual_state:
                missing, unexpected = visual.load_state_dict(visual_state, strict=False)
                print(f"visual weights loaded: {len(visual_state)} keys")

        for param in visual.encoder.parameters():
            param.requires_grad = False

        model.model.visual = visual

        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(target_device)
        return model


class Qwen3_5VJEPA21BModel(Qwen3_5VJEPA21Model):
    TORCH_HUB_NAME = "vjepa2_1_vit_base_384"

class Qwen3_5VJEPA21LModel(Qwen3_5VJEPA21Model):
    TORCH_HUB_NAME = "vjepa2_1_vit_large_384"

class Qwen3_5VJEPA21GModel(Qwen3_5VJEPA21Model):
    TORCH_HUB_NAME = "vjepa2_1_vit_giant_384"


# ──────────────────────────────────────────────
# Gemma-4 Models
# ──────────────────────────────────────────────

class Gemma4VJEPAModel(Gemma4ForConditionalGeneration):
    VISION_MODEL_ID = None

    def get_image_features(self, pixel_values, image_grid_thw, image_position_ids=None, **kwargs):
        image_embeds = self.visual(pixel_values)  # [B, N, D]
        merge_size = self.config.vision_config.spatial_merge_size
        results = []

        vid_idx = 0
        offset = 0
        for grid_idx in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[grid_idx].tolist()
            n = t * h * w

            embeds = image_embeds[vid_idx, offset:offset + n]
            embeds = embeds.view(t, h // merge_size, merge_size,
                                 w // merge_size, merge_size, -1)
            embeds = embeds.permute(0, 1, 3, 2, 4, 5)
            embeds = embeds.reshape(t, (h // merge_size) * (w // merge_size),
                                    merge_size ** 2 * embeds.shape[-1])
            embeds = embeds.to(self.visual.dtype)
            embeds = self.visual.merger(embeds)
            results.append(embeds)

            offset += n
            if offset >= image_embeds.shape[1]:
                vid_idx += 1
                offset = 0

        return BaseModelOutputWithPooling(pooler_output=tuple(results))

    def get_video_features(self, pixel_values_videos, video_grid_thw, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        vjepa2    = AutoModel.from_pretrained(cls.VISION_MODEL_ID)
        vjepa_dim = vjepa2.config.hidden_size
        model.config.vision_config.spatial_merge_size = 2
        merge_size = model.config.vision_config.spatial_merge_size
        llm_dim    = model.config.text_config.hidden_size

        device = next(model.model.language_model.parameters()).device
        dtype  = next(model.model.language_model.parameters()).dtype

        visual = VJEPA2VisualModule(vjepa2, vjepa_dim, merge_size, llm_dim).to(device=device, dtype=dtype)

        # visual은 outer class에 직접 붙으므로 key prefix = "visual." (Qwen은 "model.visual.")
        ckpt_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.exists(ckpt_file):
            visual_state = {}
            with safe_open(ckpt_file, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("visual."):
                        new_k = k.replace("visual.", "")
                        visual_state[new_k] = f.get_tensor(k).to(device=device, dtype=dtype)
            if visual_state:
                missing, unexpected = visual.load_state_dict(visual_state, strict=False)
                print(f"visual weights loaded: {len(visual_state)} keys")
                print(f"missing: {missing}")

        for param in visual.encoder.parameters():
            param.requires_grad = False

        model.visual = visual
        return model


class Gemma4VJEPALModel(Gemma4VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class Gemma4VJEPAGModel(Gemma4VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class Gemma4VJEPA21Model(Gemma4VJEPAModel):
    TORCH_HUB_NAME = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = Gemma4ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs)
        model.config.vision_config.spatial_merge_size = 2

        encoder, _ = torch.hub.load(
            'facebookresearch/vjepa2',
            cls.TORCH_HUB_NAME
        )

        vjepa_dim  = encoder.img_mod_embed.shape[-1]
        merge_size = model.config.vision_config.spatial_merge_size
        llm_dim    = model.config.text_config.hidden_size

        device = next(model.model.language_model.parameters()).device
        dtype  = next(model.model.language_model.parameters()).dtype

        visual = VJEPA2VisualModule(encoder, vjepa_dim, merge_size, llm_dim, is_v21=True).to(device=device, dtype=dtype)

        ckpt_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.exists(ckpt_file):
            visual_state = {}
            with safe_open(ckpt_file, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("visual."):
                        new_k = k.replace("visual.", "")
                        visual_state[new_k] = f.get_tensor(k).to(device=device, dtype=dtype)
            if visual_state:
                missing, unexpected = visual.load_state_dict(visual_state, strict=False)
                print(f"visual weights loaded: {len(visual_state)} keys")

        for param in visual.encoder.parameters():
            param.requires_grad = False

        model.visual = visual

        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(target_device)
        return model


class Gemma4VJEPA21BModel(Gemma4VJEPA21Model):
    TORCH_HUB_NAME = "vjepa2_1_vit_base_384"

class Gemma4VJEPA21LModel(Gemma4VJEPA21Model):
    TORCH_HUB_NAME = "vjepa2_1_vit_large_384"

class Gemma4VJEPA21GModel(Gemma4VJEPA21Model):
    TORCH_HUB_NAME = "vjepa2_1_vit_giant_384"
