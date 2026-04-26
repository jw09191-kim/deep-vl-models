import os
import glob
from safetensors import safe_open

import torch
from transformers import AutoModel
from transformers.models.lfm2_vl.modeling_lfm2_vl import (
    Lfm2VlForConditionalGeneration,
    Lfm2VlCausalLMOutputWithPast,
)
from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig

from .model_base import VJEPA2VisualModule


class Lfm2VJEPAModel(Lfm2VlForConditionalGeneration):
    VISION_MODEL_ID = None
    TORCH_HUB_NAME = None

    def __init__(self, config: Lfm2VlConfig):
        super().__init__(config)
        # Native vision_tower and multi_modal_projector are kept in the module tree
        # so that HF weight loading doesn't fail on unexpected keys; they are not
        # called at runtime — VJEPA visual module is used instead.
        self.model.visual = None

    def _validate_model_kwargs(self, model_kwargs):
        model_kwargs.pop("image_grid_thw", None)
        model_kwargs.pop("video_grid_thw", None)
        model_kwargs.pop("mm_token_type_ids", None)
        model_kwargs.pop("num_soft_tokens_per_image", None)
        model_kwargs.pop("num_soft_tokens_per_video", None)
        super()._validate_model_kwargs(model_kwargs)

    def _get_vjepa_features(self, pixel_values, grid_thw):
        """Run VJEPA encoder + tile assembly + merger for image or video inputs.

        Args:
            pixel_values: [total_tiles, T, C, H, W] — processor output
            grid_thw: [N, 3] long tensor — (t, h, w) patch grid per item

        Returns:
            list of [n_tokens, llm_dim] tensors, one per item in grid_thw
        """
        merge_size = self.model.visual.spatial_merge_size
        pps = self.model.visual.patches_per_side  # patches per side per tile
        ppt = pps * pps                            # patches per tile

        image_embeds = self.model.visual(pixel_values)  # [total_tiles, patches, D]
        results = []
        item_offset = 0

        for grid_idx in range(grid_thw.shape[0]):
            t, h, w = grid_thw[grid_idx].tolist()
            n_tiles = (h * w) // ppt

            if t == 1:
                # Image path
                if n_tiles == 1:
                    embeds = image_embeds[item_offset]  # [h*w, D]
                else:
                    tiles = image_embeds[item_offset : item_offset + n_tiles]
                    n_rows = h // pps
                    n_cols = w // pps
                    grid = tiles.view(n_rows, n_cols, pps, pps, -1)
                    assembled = grid.permute(0, 2, 1, 3, 4).contiguous()
                    embeds = assembled.reshape(h * w, -1)
            else:
                # Video path
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

            # Spatial merge
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

        return results

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
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Compute VJEPA visual features and inject into image token placeholders
        all_features = []
        if pixel_values is not None and image_grid_thw is not None:
            all_features.extend(self._get_vjepa_features(pixel_values, image_grid_thw))
        if pixel_values_videos is not None and video_grid_thw is not None:
            all_features.extend(self._get_vjepa_features(pixel_values_videos, video_grid_thw))

        if all_features:
            image_features = torch.cat(all_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            # Reuse Lfm2VlModel's placeholder mask (checks image_token_id count)
            special_image_mask = self.model.get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Bypass Lfm2VlModel.forward's native vision path; call language model directly
        lm_outputs = self.model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = lm_outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return Lfm2VlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Pass VJEPA-specific grid tensors and video pixel values through,
        # matching the pixel_values first-iteration-only logic in the parent.
        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["image_grid_thw"] = kwargs.get("image_grid_thw")
            model_inputs["video_grid_thw"] = kwargs.get("video_grid_thw")
            model_inputs["pixel_values_videos"] = kwargs.get("pixel_values_videos")

        return model_inputs

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

        merge_size = 2  # VoRAVisionConfig.MERGE_SIZE
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


class Lfm2VJEPALModel(Lfm2VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class Lfm2VJEPAGModel(Lfm2VJEPAModel):
    VISION_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"


class Lfm2VJEPA21BModel(Lfm2VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_base_384"


class Lfm2VJEPA21LModel(Lfm2VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_large_384"


class Lfm2VJEPA21GModel(Lfm2VJEPAModel):
    TORCH_HUB_NAME = "vjepa2_1_vit_giant_384"
