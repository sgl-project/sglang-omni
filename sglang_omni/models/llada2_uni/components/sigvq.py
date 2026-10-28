"""Project discrete semantic tokens into decoder conditioning features."""

import torch
from torch import nn


class LinearWrapper(nn.Module):
    """Wraps nn.Linear inside a .proj attribute to match diffusers checkpoint key format."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeedForward(nn.Module):
    """SiLU feed-forward matching diffusers key layout: net.0.proj / net.1 / net.2"""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            LinearWrapper(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SigVQ(nn.Module):
    """Map VQ token IDs to continuous features via embedding and projection."""

    def __init__(self, vocab_size: int = 16384, inner_dim: int = 4096) -> None:
        super().__init__()
        self.prior_token_embedding = nn.Embedding(vocab_size, inner_dim)
        self.prior_projector = FeedForward(dim=inner_dim, hidden_dim=inner_dim)
        self.requires_grad_(False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Project [batch, sequence] IDs to [batch, sequence, inner_dim] features."""
        return self.prior_projector(self.prior_token_embedding(token_ids))
