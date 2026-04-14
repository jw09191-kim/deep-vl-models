import torch
from typing import Any, Dict, List, Optional
from swift.template.templates.qwen import Qwen3_5Template
from swift.template.templates.gemma import Gemma4Template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.utils import findall


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

        split_token = self._tokenize('\n')
        media_inputs = self.processor(
            text='\n'.join(
                ['<|image|>'] * len(inputs.images)
                + ['<|video|>'] * len(inputs.videos)
                + ['<|audio|>'] * len(inputs.audios)
            ),
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
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
