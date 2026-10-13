"""Unpaired Zero-DCE losses (spatial, exposure, color constancy, smoothness)."""

import torch
from torch import nn
import torch.nn.functional as F


class ZeroDCELoss(nn.Module):
    def __init__(self, exposure_level: float = 0.6, patch_size: int = 16) -> None:
        super().__init__()
        self.exposure_level = exposure_level
        self.patch_size = patch_size

    @staticmethod
    def _gray(image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=1, keepdim=True)

    def forward(self, source: torch.Tensor, enhanced: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
        gray = self._gray(enhanced)
        pooled = F.avg_pool2d(gray, self.patch_size)
        exposure = (pooled - self.exposure_level).square().mean()

        means = enhanced.mean(dim=(2, 3))
        red_green = (means[:, 0] - means[:, 1]).square()
        red_blue = (means[:, 0] - means[:, 2]).square()
        green_blue = (means[:, 1] - means[:, 2]).square()
        color = (red_green + red_blue + green_blue).mean()

        source_gray = self._gray(source)
        left = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        right = source_gray[:, :, :, 1:] - source_gray[:, :, :, :-1]
        up = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        down = source_gray[:, :, 1:, :] - source_gray[:, :, :-1, :]
        spatial = (left - right).abs().mean() + (up - down).abs().mean()

        tv = (curves[:, :, :, 1:] - curves[:, :, :, :-1]).abs().mean()
        tv += (curves[:, :, 1:, :] - curves[:, :, :-1, :]).abs().mean()
        return 200 * spatial + 10 * exposure + 5 * color + 1600 * tv
