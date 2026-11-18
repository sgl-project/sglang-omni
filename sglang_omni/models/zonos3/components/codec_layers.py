# SPDX-License-Identifier: Apache-2.0
"""Building blocks of the Zonos3 audio codec: weight loading, patching, resampling, causal convolutions and quantizers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.zonos3.components.codec_kernels import nearest_codes

CONVNEXT_LAYER_NORM_EPSILON = 1e-6
CONVNEXT_KERNEL_SIZE = 7
CODEBOOK_DIMENSION = 8


def fold_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace weight-norm (gain, direction) pairs with plain weight tensors."""
    folded: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.endswith(".parametrizations.weight.original1"):
            prefix = name.removesuffix(".parametrizations.weight.original1")
            gain = state[f"{prefix}.parametrizations.weight.original0"]
        elif name.endswith(".weight_v"):
            prefix = name.removesuffix(".weight_v")
            gain = state[f"{prefix}.weight_g"]
        elif name.endswith((".parametrizations.weight.original0", ".weight_g")):
            continue
        else:
            folded[name] = tensor
            continue
        # note (zhaochenyang20): the fused op the parametrized modules evaluate in training, so folded weights match the codec's numerics.
        folded[f"{prefix}.weight"] = (
            torch._weight_norm(  # noqa: leading-underscore  # upstream spelling, no public fused op
                tensor, gain, 0
            )
        )
    return folded


def rename_state_dict(
    state: dict[str, torch.Tensor],
    prefix_renames: list[tuple[str, str]],
    infix_renames: list[tuple[str, str]],
) -> dict[str, torch.Tensor]:
    """Keep keys under the listed checkpoint prefixes and rename them to module keys."""
    renamed: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        for checkpoint_prefix, module_prefix in prefix_renames:
            if name.startswith(checkpoint_prefix):
                module_name = module_prefix + name.removeprefix(checkpoint_prefix)
                for checkpoint_infix, module_infix in infix_renames:
                    module_name = module_name.replace(checkpoint_infix, module_infix)
                renamed[module_name] = tensor
                break
            else:
                pass
    return renamed


def patch_down(hidden_states: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Fold patch_size consecutive frames into channels: (B, C, T) -> (B, C * p, T / p)."""
    batch, channels, frames = hidden_states.shape
    patched = hidden_states.reshape(batch, channels, frames // patch_size, patch_size)
    return patched.permute(0, 1, 3, 2).reshape(
        batch, channels * patch_size, frames // patch_size
    )


def patch_up(hidden_states: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Unfold channels into patch_size frames: (B, C, T) -> (B, C / p, T * p)."""
    batch, channels, frames = hidden_states.shape
    patched = hidden_states.reshape(batch, channels // patch_size, patch_size, frames)
    return patched.permute(0, 1, 3, 2).reshape(
        batch, channels // patch_size, frames * patch_size
    )


def sinc_resample(
    waveform: torch.Tensor,
    kernel: torch.Tensor,
    input_ratio: int,
    output_ratio: int,
) -> torch.Tensor:
    """Polyphase resample (B, samples) by output_ratio / input_ratio with a fixed sinc kernel."""
    batch, sample_count = waveform.shape
    kernel_half_width = (kernel.shape[-1] - input_ratio) // 2
    padded = F.pad(
        waveform[:, None], (kernel_half_width, kernel_half_width + input_ratio)
    )
    phases = F.conv1d(padded, kernel, stride=input_ratio)
    resampled = phases.transpose(1, 2).reshape(batch, -1)
    return resampled[:, : math.ceil(output_ratio * sample_count / input_ratio)]


class CausalConv1d(nn.Conv1d):
    """Conv1d padded on the left so each output depends only on past samples."""

    @property
    def left_padding(self) -> int:
        """Zero samples prepended to the input."""
        return (self.kernel_size[0] - 1) * self.dilation[0] - (self.stride[0] - 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            F.pad(hidden_states, (self.left_padding, 0)),
            self.weight,
            self.bias,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvNeXtBlock(nn.Module):
    """Causal ConvNeXt block on (B, C, T) without layer scale."""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(
            channels, channels, CONVNEXT_KERNEL_SIZE, groups=channels
        )
        self.norm = nn.LayerNorm(channels, eps=CONVNEXT_LAYER_NORM_EPSILON)
        self.pwconv1 = nn.Linear(channels, hidden_channels)
        self.pwconv2 = nn.Linear(hidden_channels, channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.add_pointwise(hidden_states, self.dwconv(hidden_states))

    def add_pointwise(
        self, hidden_states: torch.Tensor, convolved_states: torch.Tensor
    ) -> torch.Tensor:
        """Add the normalized pointwise MLP of the depthwise-convolved states to the (B, C, T) states."""
        residual = self.norm(convolved_states.transpose(1, 2))
        residual = self.pwconv2(F.gelu(self.pwconv1(residual)))
        return hidden_states + residual.transpose(1, 2)


class FactorizedVectorQuantizer(nn.Module):
    """Low-dimensional cosine-similarity vector quantizer with in/out projections."""

    def __init__(self, input_dimension: int, codebook_size: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(input_dimension, CODEBOOK_DIMENSION, 1)
        self.out_proj = nn.Conv1d(CODEBOOK_DIMENSION, input_dimension, 1)
        self.codebook = nn.Embedding(codebook_size, CODEBOOK_DIMENSION)
        # Plain attributes the encoder sets after loading: the (8, codebook size) transpose of the L2-normalized codebook and its squared entry norms.
        self.codebook_columns: torch.Tensor | None = None
        self.codebook_squares: torch.Tensor | None = None

    @torch.no_grad()
    def cache_normalized_codebook(self) -> None:
        """Normalize the codebook once on its device; encode reads the same values the eager chain rebuilt every call."""
        normalized_entries = F.normalize(self.codebook.weight, dim=-1)
        self.codebook_columns = normalized_entries.t().contiguous()
        self.codebook_squares = normalized_entries.pow(2).sum(1)

    def encode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map (B, C, T) latents to (B, T) code indices and the (B, C, T) quantized latents.

        The quantized latents keep the training straight-through arithmetic, projected + (selected - projected).
        """
        assert self.codebook_columns is not None and self.codebook_squares is not None
        codes, straight_through = nearest_codes(
            self.in_proj(latent),
            self.codebook_columns,
            self.codebook_squares,
            self.codebook.weight,
        )
        return codes, self.out_proj(straight_through)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Map (B, T) code indices to (B, C, T) latents."""
        return self.out_proj(self.codebook(codes).transpose(1, 2))
