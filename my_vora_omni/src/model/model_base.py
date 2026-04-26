import torch
from torch import nn


class VJEPA2VisualModule(nn.Module):
    def __init__(
        self,
        vjepa2_model,
        vjepa_dim,
        merge_size,
        llm_dim,
        is_v21=False,
        patches_per_side=16,
    ):
        super().__init__()
        in_dim = vjepa_dim * merge_size**2
        hidden_dim = vjepa_dim * merge_size**2
        mid_dim = max(llm_dim, hidden_dim // 2)
        out_dim = llm_dim

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
