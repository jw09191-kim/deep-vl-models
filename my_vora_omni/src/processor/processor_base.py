import os
import math
import torch

from transformers import AutoConfig
from transformers.image_utils import SizeDict
from transformers.image_processing_utils import BatchFeature


VJEPA21_CONFIGS = {
    "vjepa2_1_vit_base_384": dict(
        image_size=384, patch_size=16, tubelet_size=2, hidden_size=768
    ),
    "vjepa2_1_vit_large_384": dict(
        image_size=384, patch_size=16, tubelet_size=2, hidden_size=1024
    ),
    "vjepa2_1_vit_giant_384": dict(
        image_size=384, patch_size=16, tubelet_size=2, hidden_size=1408
    ),
}


class VoRAVisionConfig:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    MERGE_SIZE = 2

    def __init__(self, vision_model_id: str):
        if vision_model_id in VJEPA21_CONFIGS:
            cfg = VJEPA21_CONFIGS[vision_model_id]
            self.vision_model_id = vision_model_id
            self.image_size = cfg["image_size"]
            self.patch_size = cfg["patch_size"]
            self.tubelet_size = cfg["tubelet_size"]
            self.hidden_size = cfg["hidden_size"]
        else:
            cfg = AutoConfig.from_pretrained(vision_model_id)
            self.vision_model_id = vision_model_id
            self.image_size = cfg.image_size
            self.patch_size = cfg.patch_size
            self.tubelet_size = cfg.tubelet_size
            self.hidden_size = cfg.hidden_size


def _select_tile_layout(orig_h: int, orig_w: int, max_tiles: int):
    aspect = orig_w / orig_h
    best, best_score = (1, 1), float("inf")
    for n_rows in range(1, max_tiles + 1):
        for n_cols in range(1, max_tiles + 1):
            if n_rows * n_cols > max_tiles:
                continue
            score = abs(math.log(aspect / (n_cols / n_rows)))
            if score < best_score:
                best_score = score
                best = (n_rows, n_cols)
    return best


class VJEPAImageMixin:
    def _vjepa_preprocess_images(self, images, **kwargs):
        max_tiles = int(os.environ.get("IMAGE_MAX_TILES", "4"))
        merge = getattr(self, "merge_size", 1)

        h_patch = self.image_size // self.patch_size
        rescale_factor = kwargs.get("rescale_factor", 1 / 255.0)
        do_rescale = kwargs.get("do_rescale", True)
        image_mean = kwargs.get("image_mean") or self.image_mean
        image_std = kwargs.get("image_std") or self.image_std

        all_tiles = []
        all_grid_thw = []
        all_tokens = []

        for img in images:
            _, orig_h, orig_w = img.shape
            n_rows, n_cols = _select_tile_layout(orig_h, orig_w, max_tiles)
            n_tiles = n_rows * n_cols

            img_batch = self.resize(
                img.unsqueeze(0),
                SizeDict(
                    height=n_rows * self.image_size,
                    width=n_cols * self.image_size,
                ),
            )  # [1, C, target_h, target_w]

            if do_rescale:
                img_batch = img_batch * rescale_factor

            mean = torch.tensor(
                image_mean, dtype=img_batch.dtype, device=img_batch.device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                image_std, dtype=img_batch.dtype, device=img_batch.device
            ).view(1, 3, 1, 1)
            img_batch = (img_batch - mean) / std
            img_norm = img_batch.squeeze(0)

            # 타일 분할: [n_tiles, C, image_size, image_size]
            C = img_norm.shape[0]
            tiles = img_norm.view(C, n_rows, self.image_size, n_cols, self.image_size)
            tiles = (
                tiles.permute(1, 3, 0, 2, 4)
                .contiguous()
                .reshape(n_tiles, C, self.image_size, self.image_size)
            )
            # tubelet 차원 추가: [n_tiles, tubelet_size, C, H, W]
            tiles = tiles.unsqueeze(1).repeat(1, self.tubelet_size, 1, 1, 1)

            h_total = h_patch * n_rows
            w_total = h_patch * n_cols
            all_tiles.append(tiles)
            all_grid_thw.append([1, h_total, w_total])
            all_tokens.append((h_total // merge) * (w_total // merge))

        pixel_values = torch.cat(all_tiles, dim=0)  # [total_tiles, T, C, H, W]
        image_grid_thw = torch.tensor(all_grid_thw, dtype=torch.long)

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "num_soft_tokens_per_image": all_tokens,
            },
            tensor_type=kwargs.get("return_tensors", None),
        )


class VJEPAVideoMixin:
    def _vjepa_preprocess_videos(self, videos, **kwargs):
        max_tiles = int(os.environ.get("IMAGE_MAX_TILES", "4"))
        merge = getattr(self, "merge_size", 1)

        h_patch = self.image_size // self.patch_size
        rescale_factor = kwargs.get("rescale_factor", 1 / 255.0)
        do_rescale = kwargs.get("do_rescale", True)
        image_mean = kwargs.get("image_mean") or self.image_mean
        image_std = kwargs.get("image_std") or self.image_std

        all_tiles = []
        all_grid_thw = []
        all_tokens = []

        for vid in videos:
            T, C, orig_h, orig_w = vid.shape

            if T % self.tubelet_size != 0:
                pad = self.tubelet_size - (T % self.tubelet_size)
                vid = torch.cat([vid, vid[-1:].expand(pad, -1, -1, -1)], dim=0)
                T = vid.shape[0]

            n_rows, n_cols = _select_tile_layout(orig_h, orig_w, max_tiles)
            n_tiles = n_rows * n_cols
            frames = self.resize(
                vid,
                SizeDict(
                    height=n_rows * self.image_size, width=n_cols * self.image_size
                ),
            )  # [T, C, target_h, target_w]

            if do_rescale:
                frames = frames * rescale_factor

            mean = torch.tensor(
                image_mean, dtype=frames.dtype, device=frames.device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                image_std, dtype=frames.dtype, device=frames.device
            ).view(1, 3, 1, 1)
            frames = (frames - mean) / std

            # 공간 타일 분할: [n_tiles, T, C, 384, 384]
            tiles = frames.view(T, C, n_rows, self.image_size, n_cols, self.image_size)
            tiles = tiles.permute(
                2, 4, 0, 1, 3, 5
            ).contiguous()  # [n_rows, n_cols, T, C, 384, 384]
            tiles = tiles.reshape(n_tiles, T, C, self.image_size, self.image_size)

            grid_t = T // self.tubelet_size
            h_total = h_patch * n_rows
            w_total = h_patch * n_cols

            all_tiles.append(tiles)
            all_grid_thw.append([grid_t, h_total, w_total])
            all_tokens.append(grid_t * (h_total // merge) * (w_total // merge))

        pixel_values_videos = torch.cat(all_tiles, dim=0)
        video_grid_thw = torch.tensor(all_grid_thw, dtype=torch.long)

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
                "num_soft_tokens_per_video": all_tokens,
            },
            tensor_type=kwargs.get("return_tensors", None),
        )
