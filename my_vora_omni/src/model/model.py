import os
import glob
from safetensors import safe_open

import torch
from torch import nn
from transformers import AutoModel, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from transformers import Gemma4ForConditionalGeneration
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4Model


class VJEPA2VisualModule(nn.Module):
    def __init__(self, vjepa2_model, vjepa_dim, merge_size, llm_dim, is_v21=False, patches_per_side=16):
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
        self._patches_per_side = patches_per_side

    @property
    def dtype(self):
        return next(self.merger.parameters()).dtype

    @property
    def spatial_merge_size(self):
        return self._merge_size

    @property
    def patches_per_side(self):
        return self._patches_per_side

    def forward(self, pixel_values, **kwargs):
        if self._is_v21:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
            device_type = pixel_values.device.type
            with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16):
                out = self.encoder(pixel_values)
        else:
            with torch.no_grad():
                out = self.encoder(pixel_values)
            out = out.last_hidden_state
        return out
    
    
class Qwen3_5VJEPAModel(Qwen3_5ForConditionalGeneration):
    VISION_MODEL_ID = None
    TORCH_HUB_NAME = None

    def __init__(self, config):
        config.vision_config.spatial_merge_size = 2
        super().__init__(config)
        
        self.model.visual = None
        self.model.get_image_features = self.get_image_features
        self.model.get_video_features = self.get_video_features

    def _validate_model_kwargs(self, model_kwargs):
        model_kwargs.pop("num_soft_tokens_per_image", None)
        model_kwargs.pop("num_soft_tokens_per_video", None)
        super()._validate_model_kwargs(model_kwargs)

    def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
        merge_size = self.config.vision_config.spatial_merge_size
        pps        = self.model.visual.patches_per_side   # patches-per-side per tile
        ppt        = pps * pps                      # patches per tile
        image_embeds = self.model.visual(pixel_values)    # [total_items, patches, D]

        results     = []
        item_offset = 0

        for grid_idx in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[grid_idx].tolist()

            if t == 1:
                n_tiles = (h * w) // ppt
                if n_tiles == 1:
                    embeds = image_embeds[item_offset]  # [h*w, D]
                else:
                    tiles  = image_embeds[item_offset:item_offset + n_tiles]
                    n_rows = h // pps
                    n_cols = w // pps
                    grid      = tiles.view(n_rows, n_cols, pps, pps, -1)
                    assembled = grid.permute(0, 2, 1, 3, 4).contiguous()
                    assembled = assembled.reshape(h, w, -1)
                    embeds    = assembled.reshape(h * w, -1)
                item_offset += n_tiles
            else:
                n_tiles = (h * w) // ppt
                if n_tiles == 1:
                    embeds = image_embeds[item_offset].reshape(t * h * w, -1)
                else:
                    tiles  = image_embeds[item_offset:item_offset + n_tiles]  
                    n_rows = h // pps
                    n_cols = w // pps
                    tiles  = tiles.reshape(n_rows, n_cols, t, pps, pps, -1)
                    tiles  = tiles.permute(2, 0, 3, 1, 4, 5).contiguous()   
                    embeds = tiles.reshape(t * h * w, -1)
                item_offset += n_tiles

            embeds = embeds.view(t, h // merge_size, merge_size,
                                 w // merge_size, merge_size, -1)
            embeds = embeds.permute(0, 1, 3, 2, 4, 5)
            embeds = embeds.reshape(t, (h // merge_size) * (w // merge_size),
                                    merge_size ** 2 * embeds.shape[-1])
            
            embeds = embeds.to(self.dtype)
            embeds = self.model.visual.merger(embeds)
    
            results.append(embeds.view(-1, embeds.shape[-1]))
        return BaseModelOutputWithPooling(pooler_output=tuple(results))

    def get_video_features(self, pixel_values_videos, video_grid_thw, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs
        )
        
        if hasattr(outputs, 'logits') and outputs.logits is not None and outputs.logits.dim() == 4:
            if isinstance(outputs, dict):
                outputs['logits'] = outputs['logits'].squeeze(1)
            else:
                outputs.logits = outputs.logits.squeeze(1)
                
        return outputs

    @classmethod
    def _load_hf_visual(cls, merge_size, llm_dim, device, dtype):
        vjepa2 = AutoModel.from_pretrained(cls.VISION_MODEL_ID)
        encoder = vjepa2.encoder
        patches_per_side = vjepa2.config.image_size // vjepa2.config.patch_size
        return VJEPA2VisualModule(
            encoder, vjepa2.config.hidden_size, merge_size, llm_dim, 
            patches_per_side=patches_per_side, is_v21=False
        ).to(device=device, dtype=dtype)

    @classmethod
    def _load_tf_visual(cls, merge_size, llm_dim, device, dtype):
        encoder, _ = torch.hub.load('facebookresearch/vjepa2', cls.TORCH_HUB_NAME)
        patches_per_side = 384 // 16
        
        vjepa_dim = encoder.img_mod_embed.shape[-1] 
        
        return VJEPA2VisualModule(
            encoder, vjepa_dim, merge_size, llm_dim, 
            patches_per_side=patches_per_side, is_v21=True
        ).to(device=device, dtype=dtype)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        merge_size = model.config.vision_config.spatial_merge_size
        llm_dim = model.config.text_config.hidden_size
        
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype

        if getattr(cls, "VISION_MODEL_ID", None):
            visual = cls._load_hf_visual(merge_size, llm_dim, device, dtype)
        elif getattr(cls, "TORCH_HUB_NAME", None):
            visual = cls._load_tf_visual(merge_size, llm_dim, device, dtype)
        else:
            raise ValueError("Missing VISION_MODEL_ID or TORCH_HUB_NAME")
        
        safetensor_files = glob.glob(os.path.join(pretrained_model_name_or_path, "*.safetensors"))
        if safetensor_files:
            visual_state = {}
            for ckpt_file in safetensor_files:
                with safe_open(ckpt_file, framework="pt") as f:
                    for k in f.keys():
                        if k.startswith("model.visual."):
                            new_k = k.replace("model.visual.", "")
                            visual_state[new_k] = f.get_tensor(k).to(device=device, dtype=dtype)
            
            if visual_state:
                missing, unexpected = visual.load_state_dict(visual_state, strict=False)
                print(f"Visual & Aligner weights loaded: {len(visual_state)} keys")

        model.model.visual = visual
        return model


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
        # Pop custom kwargs unknown to base Gemma4, but save grid_thw so forward() can use them.
        # generate() calls _validate_model_kwargs once before the loop, so without saving here
        # image_grid_thw would be permanently gone by the time forward() is called.
        self._queued_image_grid_thw = model_kwargs.pop("image_grid_thw", None)
        self._queued_video_grid_thw = model_kwargs.pop("video_grid_thw", None)
        model_kwargs.pop("mm_token_type_ids", None)
        model_kwargs.pop("num_soft_tokens_per_video", None)
        super()._validate_model_kwargs(model_kwargs)

    def forward(self, *args, image_grid_thw=None, video_grid_thw=None, **kwargs):
        if image_grid_thw is None:
            image_grid_thw = getattr(self, '_queued_image_grid_thw', None)
            self._queued_image_grid_thw = None
        if video_grid_thw is None:
            video_grid_thw = getattr(self, '_queued_video_grid_thw', None)
            self._queued_video_grid_thw = None
        self.model._current_image_grid_thw = image_grid_thw
        self.model._current_video_grid_thw = video_grid_thw

        outputs = super().forward(*args, **kwargs)
        if hasattr(outputs, 'logits') and outputs.logits is not None and outputs.logits.dim() == 4:
            if isinstance(outputs, dict):
                outputs['logits'] = outputs['logits'].squeeze(1)
            else:
                outputs.logits = outputs.logits.squeeze(1)
                
        return outputs

    def get_image_features(self, pixel_values, image_position_ids=None, **kwargs):
        image_grid_thw = self.model._current_image_grid_thw
        
        merge_size = self.config.vision_config.spatial_merge_size
        pps        = self.model.visual.patches_per_side
        ppt        = pps * pps 

        image_embeds = self.model.visual(pixel_values) 
        results     = []
        item_offset = 0

        for grid_idx in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[grid_idx].tolist()

            if t == 1:
                n_tiles = (h * w) // ppt
                if n_tiles == 1:
                    embeds = image_embeds[item_offset]
                else:
                    tiles  = image_embeds[item_offset:item_offset + n_tiles]
                    n_rows = h // pps
                    n_cols = w // pps
                    grid      = tiles.view(n_rows, n_cols, pps, pps, -1)
                    assembled = grid.permute(0, 2, 1, 3, 4).contiguous()
                    embeds    = assembled.reshape(h * w, -1)
                item_offset += n_tiles
            else:
                n_tiles = (h * w) // ppt
                if n_tiles == 1:
                    embeds = image_embeds[item_offset].reshape(t * h * w, -1)
                else:
                    tiles  = image_embeds[item_offset:item_offset + n_tiles]
                    n_rows = h // pps
                    n_cols = w // pps
                    tiles  = tiles.reshape(n_rows, n_cols, t, pps, pps, -1)
                    tiles  = tiles.permute(2, 0, 3, 1, 4, 5).contiguous()
                    embeds = tiles.reshape(t * h * w, -1)
                item_offset += n_tiles

            embeds = embeds.view(t, h // merge_size, merge_size, 
                                 w // merge_size, merge_size, -1)
            embeds = embeds.permute(0, 1, 3, 2, 4, 5).contiguous()
            embeds = embeds.reshape(t, (h // merge_size) * (w // merge_size), -1)

            embeds = embeds.to(dtype=self.dtype, device=self.device)
            embeds = self.model.visual.merger(embeds) 
            
            results.append(embeds.view(-1, embeds.shape[-1]))

        self.model._current_image_grid_thw = None
        final_embeds = torch.cat(results, dim=0)
        
        return BaseModelOutputWithPooling(pooler_output=final_embeds)

    def get_video_features(self, pixel_values_videos, video_position_ids=None, **kwargs):
        video_grid_thw = self.model._current_video_grid_thw
        self.model._current_image_grid_thw = video_grid_thw
        
        output = self.get_image_features(pixel_values_videos, video_position_ids, **kwargs)
        self.model._current_video_grid_thw = None
        
        return output

    @classmethod
    def _load_hf_visual(cls, merge_size, llm_dim, device, dtype):
        vjepa2 = AutoModel.from_pretrained(cls.VISION_MODEL_ID)
        encoder = vjepa2.encoder
        patches_per_side = vjepa2.config.image_size // vjepa2.config.patch_size
        return VJEPA2VisualModule(
            encoder, vjepa2.config.hidden_size, merge_size, llm_dim, 
            patches_per_side=patches_per_side, is_v21=False
        ).to(device=device, dtype=dtype)

    @classmethod
    def _load_tf_visual(cls, merge_size, llm_dim, device, dtype):
        encoder, _ = torch.hub.load('facebookresearch/vjepa2', cls.TORCH_HUB_NAME)
        patches_per_side = 384 // 16
        
        vjepa_dim = encoder.img_mod_embed.shape[-1] 
        
        return VJEPA2VisualModule(
            encoder, vjepa_dim, merge_size, llm_dim, 
            patches_per_side=patches_per_side, is_v21=True
        ).to(device=device, dtype=dtype)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):        
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        merge_size = model.config.vision_config.spatial_merge_size
        llm_dim = model.config.text_config.hidden_size
        
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype

        if getattr(cls, "VISION_MODEL_ID", None):
            visual = cls._load_hf_visual(merge_size, llm_dim, device, dtype)
        elif getattr(cls, "TORCH_HUB_NAME", None):
            visual = cls._load_tf_visual(merge_size, llm_dim, device, dtype)
        else:
            raise ValueError("Missing VISION_MODEL_ID or TORCH_HUB_NAME")
        
        safetensor_files = glob.glob(os.path.join(pretrained_model_name_or_path, "*.safetensors"))
        if safetensor_files:
            visual_state = {}
            for ckpt_file in safetensor_files:
                with safe_open(ckpt_file, framework="pt") as f:
                    for k in f.keys():
                        if k.startswith("model.visual."):
                            new_k = k.replace("model.visual.", "")
                            visual_state[new_k] = f.get_tensor(k).to(device=device, dtype=dtype)
            
            if visual_state:
                missing, unexpected = visual.load_state_dict(visual_state, strict=False)
                print(f"Visual & Aligner weights loaded: {len(visual_state)} keys from {len(safetensor_files)} shard files")

        model.model.visual = visual
        return model

class Qwen3_5VJEPALModel(Qwen3_5VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

class Qwen3_5VJEPAGModel(Qwen3_5VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"

class Qwen3_5VJEPA21BModel(Qwen3_5VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_base_384"

class Qwen3_5VJEPA21LModel(Qwen3_5VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_large_384"

class Qwen3_5VJEPA21GModel(Qwen3_5VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_giant_384"

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
