import torch
from typing import Any, Dict, List, Optional
from swift.template.templates.qwen import Qwen3_5Template
from swift.template.templates.gemma import Gemma4Template

class Qwen3_5VJEPATemplate(Qwen3_5Template):

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs

        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)

        pixel_values        = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw      = inputs.get('image_grid_thw')
        video_grid_thw      = inputs.get('video_grid_thw')

        if pixel_values is not None:
            n_images = image_grid_thw.shape[0]
            pps      = base_model.model.visual.patches_per_side
            image_embeds_list = []
            tile_offset = 0

            for img_idx in range(n_images):
                t, h, w = image_grid_thw[img_idx].tolist()
                n_tiles  = (h * w) // (pps * pps) if t == 1 else 1

                pv = pixel_values[tile_offset:tile_offset + n_tiles]
                ig = image_grid_thw[img_idx:img_idx + 1]
                output = base_model.model.get_image_features(pv, ig)

                embeds = output.pooler_output[0]
                image_embeds_list.append(embeds.view(-1, embeds.shape[-1]))
                tile_offset += n_tiles

            image_embeds = torch.cat(image_embeds_list, dim=0)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = base_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            n_videos = video_grid_thw.shape[0]
            pps      = base_model.model.visual.patches_per_side
            video_embeds_list = []
            tile_offset = 0

            for vid_idx in range(n_videos):
                t, h, w = video_grid_thw[vid_idx].tolist()
                n_tiles  = (h * w) // (pps * pps)

                pv = pixel_values_videos[tile_offset:tile_offset + n_tiles]
                vg = video_grid_thw[vid_idx:vid_idx + 1]
                output = base_model.model.get_video_features(pv, vg)

                embeds = output.pooler_output[0]
                video_embeds_list.append(embeds.view(-1, embeds.shape[-1]))
                tile_offset += n_tiles

            video_embeds = torch.cat(video_embeds_list, dim=0)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = base_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return {'inputs_embeds': inputs_embeds}


# ──────────────────────────────────────────────
# Gemma-4 Template
# ──────────────────────────────────────────────

class Gemma4VJEPATemplate(Gemma4Template):

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs

        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        # Gemma4: inner model의 get_input_embeddings() 사용
        inputs_embeds = base_model.model.get_input_embeddings()(input_ids)

        pixel_values        = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw      = inputs.get('image_grid_thw')
        video_grid_thw      = inputs.get('video_grid_thw')

        if pixel_values is not None:
            n_images = image_grid_thw.shape[0]
            pps      = base_model.visual.patches_per_side
            image_embeds_list = []
            tile_offset = 0

            for img_idx in range(n_images):
                t, h, w = image_grid_thw[img_idx].tolist()
                n_tiles  = (h * w) // (pps * pps) if t == 1 else 1

                pv = pixel_values[tile_offset:tile_offset + n_tiles]
                ig = image_grid_thw[img_idx:img_idx + 1]
                output = base_model.get_image_features(pv, ig)
                embeds = output.pooler_output[0]
                image_embeds_list.append(embeds.view(-1, embeds.shape[-1]))
                tile_offset += n_tiles

            image_embeds = torch.cat(image_embeds_list, dim=0)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            # Gemma4 get_placeholder_mask은 (image_mask, video_mask, audio_mask) 반환
            image_mask, _, _ = base_model.model.get_placeholder_mask(input_ids)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
            )

        if pixel_values_videos is not None:
            n_videos = video_grid_thw.shape[0]
            pps      = base_model.visual.patches_per_side
            video_embeds_list = []
            tile_offset = 0

            for vid_idx in range(n_videos):
                t, h, w = video_grid_thw[vid_idx].tolist()
                n_tiles  = (h * w) // (pps * pps)

                pv = pixel_values_videos[tile_offset:tile_offset + n_tiles]
                vg = video_grid_thw[vid_idx:vid_idx + 1]
                output = base_model.get_video_features(pv, vg)
                embeds = output.pooler_output[0]
                video_embeds_list.append(embeds.view(-1, embeds.shape[-1]))
                tile_offset += n_tiles

            video_embeds = torch.cat(video_embeds_list, dim=0)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = base_model.model.get_placeholder_mask(input_ids)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.unsqueeze(-1).expand_as(inputs_embeds), video_embeds
            )

        return {'inputs_embeds': inputs_embeds}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # Gemma4는 mrope 없음 — 부모가 mm_token_type_ids까지 처리하므로 그대로 위임
        return super()._data_collator(batch, padding_to=padding_to)
