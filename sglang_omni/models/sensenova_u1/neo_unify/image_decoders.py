"""Pixel-shuffle image decoder."""

import torch
import torch.nn as nn

# Modified for SGLang; see this directory's README.md for upstream source.


class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.ps1: nn.PixelShuffle = nn.PixelShuffle(2)
        self.conv1: nn.Conv2d = nn.Conv2d(
            input_dim // 4, hidden_dim, kernel_size=3, padding=1
        )
        self.act1: nn.GELU = nn.GELU()

        self.ps2: nn.PixelShuffle = nn.PixelShuffle(2)
        self.conv2: nn.Conv2d = nn.Conv2d(
            hidden_dim // 4, 192, kernel_size=3, padding=1
        )

        self.ps3: nn.PixelShuffle = nn.PixelShuffle(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(self.ps1((x))))
        x = self.ps3(self.conv2(self.ps2((x))))
        return x
