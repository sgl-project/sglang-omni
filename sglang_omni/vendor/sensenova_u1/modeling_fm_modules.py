# Modified for SGLang; see this directory's README.md for upstream source.

import math
from functools import lru_cache

import torch
import torch.nn as nn


@lru_cache(maxsize=8)
def _timestep_freqs(half: int, max_period: float, device: torch.device):
    return torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=device)


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        """Create sinusoidal embeddings for N timesteps, which may be fractional."""
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = _timestep_freqs(half, max_period, t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ConvDecoder(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024):
        super().__init__()
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(hidden_dim // 4, 192, kernel_size=3, padding=1)

        self.ps3 = nn.PixelShuffle(8)

    def forward(self, x):
        x = self.act1(self.conv1(self.ps1((x))))
        x = self.ps3(self.conv2(self.ps2((x))))
        return x
