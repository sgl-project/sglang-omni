# SPDX-License-Identifier: Apache-2.0
"""Small Qwen3-TTS codec decoder with the real module layout, for incremental codec and vocoder tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.padding = (kernel_size - 1) * dilation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(hidden_states, (self.padding, 0))).contiguous()


class CausalTransConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.right_pad = kernel_size - stride

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.conv(hidden_states)
        return output[..., : -self.right_pad] if self.right_pad else output


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states
        frequencies = position_ids.to(torch.float32).unsqueeze(-1)
        frequencies = frequencies / torch.arange(
            1, self.head_dim // 2 + 1, device=position_ids.device
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos(), embeddings.sin()


class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, window_size: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = 2
        self.scaling = head_dim**-0.5
        self.sliding_window = window_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, window_size: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = Attention(hidden_size, head_dim, window_size)
        self.self_attn_layer_scale = nn.Identity()
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.mlp_layer_scale = nn.Identity()


class Transformer(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden_size: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_size)
        self.layers = nn.ModuleList([TransformerLayer(hidden_size, 2, 4)])
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, latent_dim)
        self.rotary_emb = RotaryEmbedding(2)
        self.window_size = 4


class Quantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1 / 16))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes[:, :2].to(self.scale.dtype) * self.scale


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(channels) * 0.1)
        self.beta = nn.Parameter(torch.randn(channels) * 0.1)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha).view(1, -1, 1)
        beta = torch.exp(self.beta).view(1, -1, 1)
        return hidden_states + torch.sin(hidden_states * alpha).pow(2) / (
            beta + self.no_div_by_zero
        )


class ConvNeXt(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = CausalConv(channels, channels, 7, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).permute(0, 2, 1)
        hidden_states = self.pwconv2(self.act(self.pwconv1(self.norm(hidden_states))))
        hidden_states = self.gamma * hidden_states
        return residual + hidden_states.permute(0, 2, 1)


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.act1 = SnakeBeta(channels)
        self.conv1 = CausalConv(channels, channels, 7, dilation=dilation)
        self.act2 = SnakeBeta(channels)
        self.conv2 = CausalConv(channels, channels, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(self.act1(hidden_states))
        hidden_states = self.conv2(self.act2(hidden_states))
        return hidden_states + residual


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                SnakeBeta(in_channels),
                CausalTransConv(in_channels, out_channels, 4, 2),
                ResidualUnit(out_channels, 1),
                ResidualUnit(out_channels, 3),
                ResidualUnit(out_channels, 9),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.block:
            hidden_states = module(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    total_upsample = 4

    def __init__(self, block_channels: tuple[int, int] = (8, 4)) -> None:
        super().__init__()
        in_channels, out_channels = block_channels
        self.quantizer = Quantizer()
        self.pre_conv = CausalConv(2, 2, 3)
        self.pre_transformer = Transformer()
        self.upsample = nn.ModuleList(
            [nn.ModuleList([CausalTransConv(2, 2, 2, 2), ConvNeXt(2)])]
        )
        self.decoder = nn.ModuleList(
            [
                CausalConv(2, in_channels, 7),
                DecoderBlock(in_channels, out_channels),
                SnakeBeta(out_channels),
                CausalConv(out_channels, 1, 7),
            ]
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        hidden_states = self.quantizer.decode(codes)
        hidden_states = self.pre_conv(hidden_states).transpose(1, 2)
        hidden_states = full_transformer(self.pre_transformer, hidden_states).permute(
            0, 2, 1
        )
        return self.conv_stack(hidden_states)

    def conv_stack(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for modules in self.upsample:
            for module in modules:
                hidden_states = module(hidden_states)
        for module in self.decoder:
            hidden_states = module(hidden_states)
        return hidden_states.clamp(min=-1, max=1)


def full_transformer(
    transformer: Transformer, hidden_states: torch.Tensor
) -> torch.Tensor:
    hidden_states = transformer.input_proj(hidden_states)
    length = int(hidden_states.shape[1])
    positions = torch.arange(length, device=hidden_states.device)
    cos, sin = transformer.rotary_emb(hidden_states, positions.unsqueeze(0))
    for layer in transformer.layers:
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attention = layer.self_attn
        shape = normalized.shape[:-1]
        query = attention.q_proj(normalized).view(*shape, -1, 2).transpose(1, 2)
        key = attention.k_proj(normalized).view(*shape, -1, 2).transpose(1, 2)
        value = attention.v_proj(normalized).view(*shape, -1, 2).transpose(1, 2)

        def rotate_half(item: torch.Tensor) -> torch.Tensor:
            first, second = item.chunk(2, dim=-1)
            return torch.cat((-second, first), dim=-1)

        rope_cos = cos.unsqueeze(1)
        rope_sin = sin.unsqueeze(1)
        query = query * rope_cos + rotate_half(query) * rope_sin
        key = key * rope_cos + rotate_half(key) * rope_sin
        key = key.repeat_interleave(attention.num_key_value_groups, dim=1)
        value = value.repeat_interleave(attention.num_key_value_groups, dim=1)
        scores = torch.matmul(query, key.transpose(2, 3)) * attention.scaling
        key_positions = positions.unsqueeze(0)
        query_positions = positions.unsqueeze(1)
        allowed = key_positions <= query_positions
        allowed &= key_positions > query_positions - attention.sliding_window
        scores = scores.masked_fill(
            ~allowed.view(1, 1, length, length),
            torch.finfo(scores.dtype).min,
        )
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(*shape, -1)
        hidden_states = residual + layer.self_attn_layer_scale(
            attention.o_proj(attended)
        )
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + layer.mlp_layer_scale(hidden_states)
    return transformer.output_proj(transformer.norm(hidden_states))
