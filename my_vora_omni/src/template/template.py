import numpy as np
import torch
from PIL import Image
from typing import Any, Dict, List, Optional
from swift.template.templates.qwen import Qwen3_5Template
from swift.template.templates.gemma import Gemma4Template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.utils import findall

_IMAGE_EXTS = frozenset(('jpg', 'jpeg', 'png', 'bmp', 'webp', 'gif', 'tiff', 'tif'))


def _is_frame_list(video) -> bool:
    """video가 이미지 파일 경로 목록 또는 PIL Image 목록(프레임 기반 비디오)이면 True."""
    if not isinstance(video, (list, tuple)) or len(video) == 0:
        return False
    first = video[0]
    if isinstance(first, str):
        return all(isinstance(p, str) and p.rsplit('.', 1)[-1].lower() in _IMAGE_EXTS
                   for p in video)
    # PIL Image 목록 지원: Swift가 프레임을 PIL로 전달하는 경우
    return isinstance(first, Image.Image)


def _load_frames_as_tensor(frames) -> torch.Tensor:
    """이미지 파일 경로 목록 또는 PIL Image 목록 → (T, C, H, W) uint8 [0, 255] 텐서.

    torchcodec 디코딩 결과와 동일한 포맷으로 반환한다.
    비디오 프로세서는 do_rescale=True (×1/255)를 적용하므로
    float [0, 1]로 반환하면 값이 0에 수렴하는 버그가 발생한다.
    """
    result = []
    for f in frames:
        if isinstance(f, str):
            f = Image.open(f).convert('RGB')
        elif isinstance(f, Image.Image):
            f = f.convert('RGB')
        result.append(torch.from_numpy(np.array(f)).permute(2, 0, 1))
    return torch.stack(result)  # (T, C, H, W) uint8


class Qwen3_5VJEPATemplate(Qwen3_5Template):

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # visual feature 주입은 Qwen3_5VJEPAModel.forward()가 담당한다.
        return inputs


# ──────────────────────────────────────────────
# Gemma-4 Template
# ──────────────────────────────────────────────

class Gemma4VJEPATemplate(Gemma4Template):

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # Gemma4Template._encode()의 whitelist에 image_grid_thw / video_grid_thw가 없어
        # _post_encode (또는 model.forward)에 전달되지 않는 문제를 해결한다.
        # 부모의 super(Gemma4Template, self)._encode()로 텍스트 인코딩을 얻고,
        # processor를 1회 호출해 모든 미디어 키(grid_thw 포함)를 encoded에 복사한다.
        encoded = super(Gemma4Template, self)._encode(inputs)

        # Gemma4VideoProcessor는 HF torchcodec 파이프라인을 사용하므로
        # Qwen3VL 호환 포맷(이미지 파일 목록 = 프레임 기반 비디오)을 지원하지 않는다.
        # inputs.videos 원소가 프레임 경로 리스트이면 (T, C, H, W) 텐서로 변환한다.
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

        # 기존 Gemma4Template 키 + VJEPA에 필요한 grid_thw 키 포함
        COPY_KEYS = [
            'pixel_values', 'image_position_ids',
            'pixel_values_videos', 'video_position_ids',
            'input_features', 'input_features_mask',
            'image_grid_thw', 'video_grid_thw',
        ]
        for key in COPY_KEYS:
            if key in media_inputs:
                encoded[key] = media_inputs[key]

        encoded['input_ids']  = input_ids
        encoded['labels']     = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # visual feature 주입은 Gemma4VJEPAModel.forward()가 담당한다.
        return inputs

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        result = super()._data_collator(batch, padding_to=padding_to)
        # Gemma4의 부모 collator는 image_grid_thw / video_grid_thw를 수집하지 않으므로 직접 처리
        for key in ('image_grid_thw', 'video_grid_thw'):
            tensors = [item[key] for item in batch if item.get(key) is not None]
            if tensors:
                result[key] = torch.cat(tensors, dim=0)
        return result
