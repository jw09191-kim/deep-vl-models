import numpy as np
import torch
from PIL import Image
from typing import Any, Dict, List, Literal
from swift.template import Template
from swift.template.templates.qwen import Qwen3_5Template
from swift.template.templates.gemma import Gemma4Template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.utils import findall

_IMAGE_EXTS = frozenset(('jpg', 'jpeg', 'png', 'bmp', 'webp', 'gif', 'tiff', 'tif'))


def _is_frame_list(video) -> bool:
    return (
        isinstance(video, (list, tuple))
        and len(video) > 0
        and all(isinstance(p, str) and p.rsplit('.', 1)[-1].lower() in _IMAGE_EXTS
                for p in video)
    )


def _load_frames_as_tensor(frame_paths: List[str]) -> torch.Tensor:
    frames = [
        torch.from_numpy(np.array(Image.open(p).convert('RGB'))).permute(2, 0, 1)
        for p in frame_paths
    ]
    return torch.stack(frames)  # (T, C, H, W) uint8


class Qwen3_5VJEPATemplate(Qwen3_5Template):

    def _post_encode(self, _model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # visual feature 주입은 Qwen3_5VJEPAModel.forward()가 담당한다.
        return inputs


# ──────────────────────────────────────────────
# Gemma-4 Template
# ──────────────────────────────────────────────

class Gemma4VJEPATemplate(Gemma4Template):
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super(Gemma4Template, self)._encode(inputs)

        videos_for_processor = None
        if inputs.videos:
            videos_for_processor = [
                _load_frames_as_tensor(v) if _is_frame_list(v) else v
                for v in inputs.videos
            ]

        split_token = self._tokenize('\n')
        media_inputs = self.processor(
            text='\n'.join(
                ['<|image|>'] * len(inputs.images)
                + ['<|video|>'] * len(inputs.videos)
                + ['<|audio|>'] * len(inputs.audios)
            ),
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=videos_for_processor,
            return_tensors='pt',
            add_special_tokens=False,
        )
        splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')

        input_ids  = encoded['input_ids']
        labels     = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        idx_list = []
        for key in ['image', 'video', 'audio']:
            idx_list += findall(input_ids, getattr(self.config, f'{key}_token_id'))
        sorted_order  = sorted(range(len(idx_list)), key=lambda i: idx_list[i])
        idx_list      = [idx_list[i] for i in sorted_order]
        splited_tokens = [splited_tokens[i] for i in sorted_order]

        if idx_list:
            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, lambda i: splited_tokens[i]
            )

        COPY_KEYS = [
            'pixel_values', 'image_position_ids',
            'pixel_values_videos', 'video_position_ids',
            'input_features', 'input_features_mask',
            'image_grid_thw', 'video_grid_thw',
            # num_soft_tokens_*: consumed by Gemma4Processor.__call__ (used to expand
            # <image> → N soft tokens in text), so they are absent from media_inputs here
        ]
        for key in COPY_KEYS:
            if key in media_inputs:
                encoded[key] = media_inputs[key]

        encoded['input_ids']  = input_ids
        encoded['labels']     = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        for key in ['image_grid_thw', 'video_grid_thw']:
            value = [b[key] for b in batch if b.get(key) is not None]
            if value:
                res[key] = torch.concat(value)
        return res


# ──────────────────────────────────────────────
# LFM2-VL Template
# ──────────────────────────────────────────────

class Lfm2VJEPATemplate(Template):
    """VoRA template for LFM2-VL (LiquidAI).

    LFM2 has no native video support and uses a single <image> token (id=396)
    for both images and videos. This template:
      - Inserts <image> for each image/video placeholder via replace_tag
      - Calls the VJEPA processor to get pixel_values + grid_thw + soft-token counts
      - Expands each <image> placeholder → N repetitions of image_token_id
      - Passes pixel_values / image_grid_thw / pixel_values_videos / video_grid_thw
        to the model (visual injection happens in Lfm2VJEPAModel.forward)
    """

    placeholder_tokens = ['<image>']

    def replace_tag(
        self,
        media_type: Literal['image', 'video', 'audio'],
        index: int,
        inputs: StdTemplateInputs,
    ) -> List:
        if media_type == 'video' and _is_frame_list(inputs.videos[index]):
            inputs.videos[index] = _load_frames_as_tensor(inputs.videos[index])
        if media_type in ('image', 'video'):
            return ['<image>']
        return []

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        # image_token_id == 396 for LFM2-VL (Lfm2VlConfig default)
        image_token_id = self.config.image_token_id

        # soft token count per media item (images first, then videos — matches
        # the order replace_tag inserts <image> placeholders in the prompt)
        all_soft_tokens: List[int] = []

        if inputs.images:
            # TorchvisionBackend.preprocess handles PIL→tensor conversion before
            # calling our overridden _preprocess → _vjepa_preprocess_images
            img_out = self.processor.image_processor(images=inputs.images)
            encoded['pixel_values'] = img_out['pixel_values']
            encoded['image_grid_thw'] = img_out['image_grid_thw']
            soft = img_out['num_soft_tokens_per_image']
            all_soft_tokens.extend(soft.tolist() if isinstance(soft, torch.Tensor) else soft)

        if inputs.videos:
            videos_tensors = [
                _load_frames_as_tensor(v) if _is_frame_list(v) else v
                for v in inputs.videos
            ]
            # Video processor inherits resize() from Lfm2VlImageProcessor;
            # call _vjepa_preprocess_videos directly (no parent preprocess path
            # for video since Lfm2 has no native video processor)
            vid_out = self.processor.video_processor._vjepa_preprocess_videos(videos_tensors)
            encoded['pixel_values_videos'] = vid_out['pixel_values_videos']
            encoded['video_grid_thw'] = vid_out['video_grid_thw']
            soft = vid_out['num_soft_tokens_per_video']
            all_soft_tokens.extend(soft.tolist() if isinstance(soft, torch.Tensor) else soft)

        # Expand each <image> placeholder → N image_token_id tokens
        if all_soft_tokens:
            idx_list = findall(input_ids, image_token_id)

            def _get_new_tokens(i: int) -> List[int]:
                return [image_token_id] * all_soft_tokens[i]

            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens
            )

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Visual feature injection is handled by Lfm2VJEPAModel.forward
        return inputs

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        for key in ['image_grid_thw', 'video_grid_thw']:
            value = [b[key] for b in batch if b.get(key) is not None]
            if value:
                res[key] = torch.concat(value)
        return res
