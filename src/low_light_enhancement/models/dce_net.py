"""Zero-DCE curve-estimation network and image adjustment."""

import torch
from torch import nn


def apply_curve(image: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
    """Apply the bounded quadratic curve adjustment eight times."""
    enhanced = image
    for curve in curves.chunk(8, dim=1):
        enhanced = enhanced + curve * (enhanced.square() - enhanced)
    return enhanced


class ZeroDCE(nn.Module):
    """Predict 24 pixel-wise adjustment maps for RGB images in [0, 1]."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.conv3 = nn.Conv2d(width, width, 3, padding=1)
        self.conv4 = nn.Conv2d(width, width, 3, padding=1)
        self.conv5 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.conv6 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.conv7 = nn.Conv2d(width * 2, 24, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.conv1(image))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat((x3, x4), dim=1)))
        x6 = self.relu(self.conv6(torch.cat((x2, x5), dim=1)))
        curves = torch.tanh(self.conv7(torch.cat((x1, x6), dim=1)))
        return apply_curve(image, curves), curves
