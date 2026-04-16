import os
from safetensors import safe_open

import torch
from torch import nn
from transformers import AutoModel, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from transformers import Gemma4ForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel


# ──────────────────────────────────────────────
# Gemma4 per_layer_inputs OOM fix
#
# When Swift's pre_forward_hook returns {'inputs_embeds': ...}, it removes input_ids.
# Gemma4's get_per_layer_inputs(input_ids=None, inputs_embeds) then tries to recover
# input_ids via a [batch, seq_len, vocab_size, hidden_dim] boolean comparison → OOM.
#
# Fix: if _vjepa_llm_input_ids was pre-computed and stored on the language model by
# Gemma4VJEPATemplate._post_encode, use it directly. Otherwise fall back to PAD tokens.
# ──────────────────────────────────────────────
_gemma4_orig_get_per_layer_inputs = Gemma4TextModel.get_per_layer_inputs

def _gemma4_safe_get_per_layer_inputs(self, input_ids, inputs_embeds):
    if input_ids is None:
        if hasattr(self, '_vjepa_llm_input_ids'):
            input_ids = self._vjepa_llm_input_ids
            del self._vjepa_llm_input_ids
        elif inputs_embeds is not None:
            pad_id = getattr(self.config, 'pad_token_id', 0)
            batch, seq_len = inputs_embeds.shape[:2]
            input_ids = inputs_embeds.new_full((batch, seq_len), pad_id, dtype=torch.long)
    return _gemma4_orig_get_per_layer_inputs(self, input_ids, inputs_embeds)

Gemma4TextModel.get_per_layer_inputs = _gemma4_safe_get_per_layer_inputs


class Qwen3_5VJEPAInnerModel(Qwen3_5Model):
    
    def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
        merge_size = self.config.vision_config.spatial_merge_size
        pps        = self.visual.patches_per_side   # patches-per-side per tile
        ppt        = pps * pps                      # patches per tile

        # 모든 항목(타일 또는 비디오 클립)을 한 번에 인코딩
        image_embeds = self.visual(pixel_values)    # [total_items, patches, D]

        results     = []
        item_offset = 0   # image_embeds 첫 번째 차원 인덱스

        for grid_idx in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[grid_idx].tolist()

            if t == 1:   # 이미지 — 타일링 지원
                n_tiles = (h * w) // ppt

                if n_tiles == 1:
                    embeds = image_embeds[item_offset]  # [h*w, D]
                else:
                    tiles  = image_embeds[item_offset:item_offset + n_tiles]
                    # tiles: [n_tiles, pps*pps, D]
                    n_rows = h // pps
                    n_cols = w // pps
                    # [n_rows, n_cols, pps, pps, D] → [h, w, D]
                    grid      = tiles.view(n_rows, n_cols, pps, pps, -1)
                    assembled = grid.permute(0, 2, 1, 3, 4).contiguous()
                    assembled = assembled.reshape(h, w, -1)
                    embeds    = assembled.reshape(h * w, -1)

                item_offset += n_tiles

            else:   # 비디오 — dynamic spatial tiling 지원
                n_tiles = (h * w) // ppt

                if n_tiles == 1:
                    # 단일 타일: [t*pps², D] → [t*h*w, D]
                    embeds = image_embeds[item_offset].reshape(t * h * w, -1)
                    item_offset += 1
                else:
                    tiles  = image_embeds[item_offset:item_offset + n_tiles]  # [n_tiles, t*pps², D]
                    n_rows = h // pps
                    n_cols = w // pps
                    tiles  = tiles.reshape(n_rows, n_cols, t, pps, pps, -1)
                    tiles  = tiles.permute(2, 0, 3, 1, 4, 5).contiguous()   # [t, n_rows, pps, n_cols, pps, D]
                    embeds = tiles.reshape(t * h * w, -1)
                    item_offset += n_tiles

            # 공간 merge (이미지/비디오 공통)
            embeds = embeds.view(t, h // merge_size, merge_size,
                                 w // merge_size, merge_size, -1)
            embeds = embeds.permute(0, 1, 3, 2, 4, 5)
            embeds = embeds.reshape(t, (h // merge_size) * (w // merge_size),
                                    merge_size ** 2 * embeds.shape[-1])
            embeds = embeds.to(self.visual.dtype)
            embeds = self.visual.merger(embeds)
            results.append(embeds)

        return BaseModelOutputWithPooling(pooler_output=tuple(results))

    def get_video_features(self, pixel_values_videos, video_grid_thw, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

class VJEPA2VisualModule(nn.Module):
    def __init__(self, vjepa2_model, vjepa_dim, merge_size, llm_dim,
                 is_v21=False, patches_per_side=16):
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
            # encoder는 기본적으로 frozen — no_grad로 불필요한 gradient 계산 방지
            # VJEPA를 실제로 학습할 때는 이 with 블록을 제거할 것
            with torch.no_grad():
                out = self.encoder(pixel_values)
            out = out.last_hidden_state
        return out

class Qwen3_5VJEPAModel(Qwen3_5ForConditionalGeneration):
    VISION_MODEL_ID = None

    def __init__(self, config):
        config.vision_config.spatial_merge_size = 2
        super().__init__(config)
        self.model = Qwen3_5VJEPAInnerModel(config)

    def _validate_model_kwargs(self, model_kwargs):
        model_kwargs.pop("num_soft_tokens_per_image", None)
        model_kwargs.pop("num_soft_tokens_per_video", None)
        model_kwargs.pop("mm_token_type_ids", None)
        super()._validate_model_kwargs(model_kwargs)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        has_visual = (
            (pixel_values is not None and image_grid_thw is not None)
            or (pixel_values_videos is not None and video_grid_thw is not None)
        )

        if has_visual and inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)

            if pixel_values is not None and image_grid_thw is not None:
                n_images = image_grid_thw.shape[0]
                pps = self.model.visual.patches_per_side
                image_embeds_list, tile_offset = [], 0
                for i in range(n_images):
                    t, h, w = image_grid_thw[i].tolist()
                    n_tiles = (h * w) // (pps * pps) if t == 1 else 1
                    pv = pixel_values[tile_offset:tile_offset + n_tiles]
                    output = self.model.get_image_features(pv, image_grid_thw[i:i + 1])
                    embeds = output.pooler_output[0]
                    image_embeds_list.append(embeds.view(-1, embeds.shape[-1]))
                    tile_offset += n_tiles
                image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None and video_grid_thw is not None:
                n_videos = video_grid_thw.shape[0]
                pps = self.model.visual.patches_per_side
                video_embeds_list, tile_offset = [], 0
                for i in range(n_videos):
                    t, h, w = video_grid_thw[i].tolist()
                    n_tiles = (h * w) // (pps * pps)
                    pv = pixel_values_videos[tile_offset:tile_offset + n_tiles]
                    output = self.model.get_video_features(pv, video_grid_thw[i:i + 1])
                    embeds = output.pooler_output[0]
                    video_embeds_list.append(embeds.view(-1, embeds.shape[-1]))
                    tile_offset += n_tiles
                video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # input_ids를 None으로 전환하기 전에 mrope position_ids 계산.
            # parent forward()는 input_ids=None이면 get_rope_index()를 올바르게
            # 호출할 수 없어 모든 토큰에 1D 순차 position_ids가 부여된다.
            if 'position_ids' not in kwargs:
                position_ids, _ = self.get_rope_index(
                    input_ids,
                    image_grid_thw if pixel_values is not None else None,
                    video_grid_thw if pixel_values_videos is not None else None,
                    attention_mask,
                )
                kwargs['position_ids'] = position_ids

            input_ids = None  # inputs_embeds로 전환

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # pixel_values / pixel_values_videos 전달 안 함 (이미 주입 완료)
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # model.config.use_cache = True
        # model.config.text_config.use_cache = True
        # model.model.language_model.config.use_cache = True

        vjepa2           = AutoModel.from_pretrained(cls.VISION_MODEL_ID)
        vjepa_dim        = vjepa2.config.hidden_size
        merge_size       = model.config.vision_config.spatial_merge_size
        llm_dim          = model.config.text_config.hidden_size
        patches_per_side = vjepa2.config.image_size // vjepa2.config.patch_size

        device = next(model.model.language_model.parameters()).device
        dtype  = next(model.model.language_model.parameters()).dtype

        visual = VJEPA2VisualModule(
            vjepa2, vjepa_dim, merge_size, llm_dim,
            patches_per_side=patches_per_side,
        ).to(device=device, dtype=dtype)

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

        vjepa_dim        = encoder.img_mod_embed.shape[-1]
        merge_size       = model.config.vision_config.spatial_merge_size
        llm_dim          = model.config.text_config.hidden_size
        patches_per_side = 384 // 16  # VJEPA2.1 전 변형 공통 (image_size=384, patch_size=16)

        device = next(model.model.language_model.parameters()).device
        dtype  = next(model.model.language_model.parameters()).dtype

        visual = VJEPA2VisualModule(
            encoder, vjepa_dim, merge_size, llm_dim, is_v21=True,
            patches_per_side=patches_per_side,
        ).to(device=device, dtype=dtype)

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
        merge_size = self.config.vision_config.spatial_merge_size
        pps        = self.visual.patches_per_side
        ppt        = pps * pps

        image_embeds = self.visual(pixel_values)    # [total_items, patches, D]

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
                    assembled = assembled.reshape(h, w, -1)
                    embeds    = assembled.reshape(h * w, -1)

                item_offset += n_tiles

            else:   # 비디오 — dynamic spatial tiling 지원
                n_tiles = (h * w) // ppt

                if n_tiles == 1:
                    embeds = image_embeds[item_offset].reshape(t * h * w, -1)
                    item_offset += 1
                else:
                    tiles  = image_embeds[item_offset:item_offset + n_tiles]  # [n_tiles, t*pps², D]
                    n_rows = h // pps
                    n_cols = w // pps
                    tiles  = tiles.reshape(n_rows, n_cols, t, pps, pps, -1)
                    tiles  = tiles.permute(2, 0, 3, 1, 4, 5).contiguous()   # [t, n_rows, pps, n_cols, pps, D]
                    embeds = tiles.reshape(t * h * w, -1)
                    item_offset += n_tiles

            embeds = embeds.view(t, h // merge_size, merge_size,
                                 w // merge_size, merge_size, -1)
            embeds = embeds.permute(0, 1, 3, 2, 4, 5)
            embeds = embeds.reshape(t, (h // merge_size) * (w // merge_size),
                                    merge_size ** 2 * embeds.shape[-1])
            embeds = embeds.to(self.visual.dtype)
            embeds = self.visual.merger(embeds)
            results.append(embeds)

        return BaseModelOutputWithPooling(pooler_output=tuple(results))

    def get_video_features(self, pixel_values_videos, video_grid_thw, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        has_visual = (
            (pixel_values is not None and image_grid_thw is not None)
            or (pixel_values_videos is not None and video_grid_thw is not None)
        )

        if has_visual and inputs_embeds is None and input_ids is not None:
            # 1. 토큰 임베딩 계산
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

            # 2. sanitized input_ids 생성 (visual 위치 → pad_token_id)
            #    monkey-patched get_per_layer_inputs가 inputs_embeds만으로 역추적하는
            #    O(batch × seq × vocab × hidden) OOM을 방지하기 위해 캐싱
            image_mask_2d, video_mask_2d, audio_mask_2d = self.model.get_placeholder_mask(input_ids)
            visual_mask = image_mask_2d | video_mask_2d | audio_mask_2d
            sanitized_ids = input_ids.clone()
            sanitized_ids[visual_mask] = self.config.text_config.pad_token_id
            self.model.language_model._vjepa_llm_input_ids = sanitized_ids

            # 3. 이미지 feature 주입
            if pixel_values is not None and image_grid_thw is not None:
                n_images = image_grid_thw.shape[0]
                pps = self.visual.patches_per_side
                image_embeds_list = []
                tile_offset = 0
                for i in range(n_images):
                    t, h, w = image_grid_thw[i].tolist()
                    n_tiles = (h * w) // (pps * pps) if t == 1 else 1
                    pv = pixel_values[tile_offset:tile_offset + n_tiles]
                    output = self.get_image_features(pv, image_grid_thw[i:i + 1])
                    embeds = output.pooler_output[0]
                    image_embeds_list.append(embeds.view(-1, embeds.shape[-1]))
                    tile_offset += n_tiles
                image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                img_mask, _, _ = self.model.get_placeholder_mask(input_ids)
                inputs_embeds = inputs_embeds.masked_scatter(
                    img_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
                )

            # 4. 비디오 feature 주입
            if pixel_values_videos is not None and video_grid_thw is not None:
                n_videos = video_grid_thw.shape[0]
                pps = self.visual.patches_per_side
                video_embeds_list = []
                tile_offset = 0
                for i in range(n_videos):
                    t, h, w = video_grid_thw[i].tolist()
                    n_tiles = (h * w) // (pps * pps)
                    pv = pixel_values_videos[tile_offset:tile_offset + n_tiles]
                    output = self.get_video_features(pv, video_grid_thw[i:i + 1])
                    embeds = output.pooler_output[0]
                    video_embeds_list.append(embeds.view(-1, embeds.shape[-1]))
                    tile_offset += n_tiles
                video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, vid_mask, _ = self.model.get_placeholder_mask(input_ids)
                inputs_embeds = inputs_embeds.masked_scatter(
                    vid_mask.unsqueeze(-1).expand_as(inputs_embeds), video_embeds
                )

            input_ids = None  # inputs_embeds로 전환, pixel_values는 parent에 전달하지 않음

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # pixel_values / pixel_values_videos: 이미 주입 완료 → parent에 전달 안 함
            **kwargs,
        )

    def _validate_model_kwargs(self, model_kwargs):
        model_kwargs.pop("num_soft_tokens_per_image", None)
        model_kwargs.pop("num_soft_tokens_per_video", None)
        model_kwargs.pop("mm_token_type_ids", None)
        # image_grid_thw / video_grid_thw는 forward()가 직접 소비하므로 제거하지 않음
        super()._validate_model_kwargs(model_kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        vjepa2           = AutoModel.from_pretrained(cls.VISION_MODEL_ID)
        vjepa_dim        = vjepa2.config.hidden_size
        model.config.vision_config.spatial_merge_size = 2
        merge_size       = model.config.vision_config.spatial_merge_size
        llm_dim          = model.config.text_config.hidden_size
        patches_per_side = vjepa2.config.image_size // vjepa2.config.patch_size

        device = next(model.model.language_model.parameters()).device
        dtype  = next(model.model.language_model.parameters()).dtype

        visual = VJEPA2VisualModule(
            vjepa2, vjepa_dim, merge_size, llm_dim,
            patches_per_side=patches_per_side,
        ).to(device=device, dtype=dtype)

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

        vjepa_dim        = encoder.img_mod_embed.shape[-1]
        merge_size       = model.config.vision_config.spatial_merge_size
        llm_dim          = model.config.text_config.hidden_size
        patches_per_side = 384 // 16  # VJEPA2.1 전 변형 공통 (image_size=384, patch_size=16)

        device = next(model.model.language_model.parameters()).device
        dtype  = next(model.model.language_model.parameters()).dtype

        visual = VJEPA2VisualModule(
            encoder, vjepa_dim, merge_size, llm_dim, is_v21=True,
            patches_per_side=patches_per_side,
        ).to(device=device, dtype=dtype)

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
