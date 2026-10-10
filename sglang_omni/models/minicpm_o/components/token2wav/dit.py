# SPDX-License-Identifier: Apache-2.0
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Dit for MiniCPM-o."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, repeat
from torch.nn.attention.varlen import varlen_attn

from sglang_omni.models.minicpm_o.components.token2wav.causal_conv import (
    CausalConv1d,
    ConvState,
)
from sglang_omni.models.minicpm_o.components.token2wav.conformer_state import (
    AttentionState,
)
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_weight,
)

TIMESTEP_MAX_PERIOD = 10000
MIN_PACKED_BATCH_SIZE = 3


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(self.inner_dim, dim)

    def to_heads(self, ts: torch.Tensor) -> torch.Tensor:
        b, t, c = ts.shape
        ts = ts.reshape(b, t, self.num_heads, c // self.num_heads)
        ts = ts.transpose(1, 2)
        return ts

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state: AttentionState | None = None,
        window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AttentionState | None]:
        """A window holds keys and values over [new frames, history] with the history already written; the new keys and values are written into its leading frames in place."""
        b, t, c = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.to_heads(q)
        k = self.to_heads(k)
        v = self.to_heads(v)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if window is not None:
            window[:, :, :t, : self.head_dim].copy_(k)
            window[:, :, :t, self.head_dim :].copy_(v)
            k, v = window.chunk(2, dim=-1)
            next_state = None
        elif state is not None:
            if state.history is not None:
                previous_key, previous_value = state.history.chunk(2, dim=-1)
                k = torch.cat((k, previous_key), dim=2)
                v = torch.cat((v, previous_value), dim=2)
            else:
                pass
            next_state = AttentionState(history=torch.cat((k, v), dim=-1))
        else:
            next_state = None
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        else:
            pass
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, next_state

    def forward_packed(
        self,
        x: torch.Tensor,
        cumulative_sequence_lengths: torch.Tensor,
        maximum_sequence_length: int,
    ) -> torch.Tensor:
        q = self.to_q(x).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x).view(-1, self.num_heads, self.head_dim)
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        x = varlen_attn(
            q,
            k,
            v,
            cumulative_sequence_lengths,
            cumulative_sequence_lengths,
            maximum_sequence_length,
            maximum_sequence_length,
        )
        x = self.proj(x.reshape(-1, self.inner_dim))
        return self.proj_drop(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.scale = 1000
        half = frequency_embedding_size // 2
        # note (MayDomine): autocast timesteps can remain FP32 with FP16 weights.
        self.frequencies = torch.exp(
            -math.log(TIMESTEP_MAX_PERIOD) * torch.arange(half) / half
        )
        self.frequency_cache: torch.Tensor | None = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if (
            self.frequency_cache is None
            or self.frequency_cache.device != t.device
            or self.frequency_cache.dtype != t.dtype
        ):
            self.frequency_cache = self.frequencies.to(t)
        else:
            pass
        angles = (t * self.scale)[:, None] * self.frequency_cache[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        else:
            pass
        return self.mlp(embedding)


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


@dataclass(frozen=True, kw_only=True)
class ConvBlockState:
    first: ConvState = field(default_factory=ConvState)
    second: ConvState = field(default_factory=ConvState)


@dataclass(frozen=True, kw_only=True)
class DiTState:
    """Per-block states stacked along the first axis; empty fields start a stream."""

    convolution: torch.Tensor | None = None
    attention: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    valid_frame_counts: torch.Tensor | None = None


class CausalConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block = torch.nn.Sequential(
            Transpose(1, 2),
            CausalConv1d(in_channels, out_channels, kernel_size),
            Transpose(1, 2),
            nn.LayerNorm(out_channels),
            nn.Mish(),
            Transpose(1, 2),
            CausalConv1d(out_channels, out_channels, kernel_size),
            Transpose(1, 2),
        )
        self.is_channels_last = False

    def use_channels_last(self) -> None:
        """Store both conv weights channels last; every (B, T, C) path then convolves without transposes."""
        for convolution in (self.block[1], self.block[6]):
            if not isinstance(convolution, CausalConv1d):
                raise TypeError(
                    "the channels-last path expects block[1] and block[6] to be CausalConv1d"
                )
            else:
                pass
            convolution.weight.data = channels_last_weight(convolution)
        self.is_channels_last = True

    def channels_last_convolution(
        self,
        hidden_states: torch.Tensor,
        convolution: CausalConv1d,
        state: ConvState | None = None,
    ) -> tuple[torch.Tensor, ConvState | None]:
        """Causal conv of (B, T, C) frames without a channel-first transpose.

        The streaming history keeps its (B, C, frames) layout; only its two frames
        are transposed into and out of the channels-last context.
        """
        history_length = (convolution.kernel_size[0] - 1) * convolution.dilation[0]
        if state is not None and state.history is not None:
            context = torch.cat((state.history.transpose(1, 2), hidden_states), dim=1)
        else:
            context = F.pad(hidden_states, (0, 0, history_length, 0))
        next_state = (
            ConvState(
                history=context[:, -history_length:]
                .transpose(1, 2)
                .clone(memory_format=torch.contiguous_format)
            )
            if state is not None
            else None
        )
        output = channels_last_conv1d(
            context, convolution, convolution.weight, hidden_states.shape[1]
        )
        return output, next_state

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        state: ConvBlockState | None = None,
    ) -> tuple[torch.Tensor, ConvBlockState | None]:
        if mask is not None:
            x = x * mask
        else:
            pass
        if self.is_channels_last:
            first, second = (
                (state.first, state.second) if state is not None else (None, None)
            )
            x, first = self.channels_last_convolution(x, self.block[1], first)
            x = self.block[4](self.block[3](x))
            x, second = self.channels_last_convolution(x, self.block[6], second)
            next_state = (
                ConvBlockState(first=first, second=second)
                if state is not None
                else None
            )
        else:
            previous = iter(
                (state.first, state.second) if state is not None else (None, None)
            )
            histories: list[ConvState] = []
            for module in self.block:
                if isinstance(module, CausalConv1d):
                    x, history = module(x, next(previous))
                    if history is not None:
                        histories.append(history)
                    else:
                        pass
                else:
                    x = module(x)
            next_state = (
                ConvBlockState(first=histories[0], second=histories[1])
                if state is not None
                else None
            )
        if mask is not None:
            x = x * mask
        else:
            pass
        return x, next_state

    def forward_packed(
        self,
        hidden_states: torch.Tensor,
        real_frame_positions: torch.Tensor,
        real_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        def apply_causal_convolution(
            frames: torch.Tensor, convolution: CausalConv1d
        ) -> torch.Tensor:
            if self.is_channels_last:
                convolved, _ = self.channels_last_convolution(
                    frames.unsqueeze(0), convolution
                )
                return convolved.squeeze(0)
            else:
                channel_first = frames.transpose(0, 1).unsqueeze(0)
                convolved, _ = convolution(channel_first)
                return convolved.squeeze(0).transpose(0, 1)

        first_convolution = self.block[1]
        layer_norm = self.block[3]
        activation = self.block[4]
        second_convolution = self.block[6]
        channels = hidden_states.shape[1]
        expanded_hidden_states = hidden_states.new_zeros(
            real_frame_mask.shape[0], channels
        )
        expanded_hidden_states[real_frame_positions] = hidden_states
        expanded_hidden_states = apply_causal_convolution(
            expanded_hidden_states, first_convolution
        )
        normalized_hidden_states = activation(layer_norm(expanded_hidden_states))
        isolated_hidden_states = normalized_hidden_states * real_frame_mask.unsqueeze(1)
        convolved_hidden_states = apply_causal_convolution(
            isolated_hidden_states, second_convolution
        )
        return convolved_hidden_states[real_frame_positions]


class DiTBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=True,
            qk_norm=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.conv = CausalConvBlock(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=3
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        attn_mask: torch.Tensor | None,
        convolution_state: ConvBlockState | None = None,
        attention_state: AttentionState | None = None,
        attention_window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ConvBlockState | None, AttentionState | None]:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(timestep_embedding).chunk(9, dim=-1)
        attention, next_attention_state = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            attn_mask,
            attention_state,
            attention_window,
        )
        x = x + gate_msa * attention
        convolution, next_convolution_state = self.conv(
            modulate(self.norm3(x), shift_conv, scale_conv), state=convolution_state
        )
        x = x + gate_conv * convolution
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, next_convolution_state, next_attention_state

    def forward_packed(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        sequence_ids: torch.Tensor,
        cumulative_sequence_lengths: torch.Tensor,
        maximum_sequence_length: int,
        real_frame_positions: torch.Tensor,
        real_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(timestep_embedding)[sequence_ids].chunk(9, dim=-1)
        x = x + gate_msa * self.attn.forward_packed(
            modulate(self.norm1(x), shift_msa, scale_msa),
            cumulative_sequence_lengths,
            maximum_sequence_length,
        )
        x = x + gate_conv * self.conv.forward_packed(
            modulate(self.norm3(x), shift_conv, scale_conv),
            real_frame_positions,
            real_frame_mask,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(timestep_embedding).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def forward_packed(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        sequence_ids: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(timestep_embedding)[sequence_ids].chunk(
            2, dim=-1
        )
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        head_dim: int = 64,
        hidden_size: int = 256,
        enable_variable_length: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enable_variable_length = enable_variable_length
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, head_dim, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:

        def initialize_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                else:
                    pass
            else:
                pass

        self.apply(initialize_linear)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        speaker_embeddings: torch.Tensor | None = None,
        mel_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.t_embedder(t).unsqueeze(1)
        x = self.pack_inputs(x, mu, speaker_embeddings, mel_conditioning)
        attn_mask = mask.bool()
        if (
            self.enable_variable_length
            and x.shape[0] >= MIN_PACKED_BATCH_SIZE
            and x.is_cuda
        ):
            sequence_lengths = attn_mask.squeeze(1).sum(dim=1, dtype=torch.int32)
            with torch.autocast(x.device.type, dtype=torch.bfloat16):
                x = self.forward_packed(
                    self.in_proj(x), t.to(torch.bfloat16), sequence_lengths
                )
        else:
            x = self.in_proj(x)
            for block in self.blocks:
                x, _, _ = block(x, t, attn_mask)
            x = self.final_layer(x, t).transpose(1, 2)
        return x

    def pack_inputs(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor | None,
        mel_conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        x = pack([x, mu], "b * t")[0]
        if speaker_embeddings is not None:
            speaker_embeddings = repeat(
                speaker_embeddings, "b c -> b c t", t=x.shape[-1]
            )
            x = pack([x, speaker_embeddings], "b * t")[0]
        else:
            pass
        if mel_conditioning is not None:
            x = pack([x, mel_conditioning], "b * t")[0]
        else:
            pass
        return x.transpose(1, 2)

    def forward_chunk(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        state: DiTState,
    ) -> tuple[torch.Tensor, DiTState]:
        """Run one chunk against the streams' histories, masked only for ragged rows."""
        timestep_embedding = self.t_embedder(t).unsqueeze(1)
        x = self.in_proj(self.pack_inputs(x, mu, speaker_embeddings, mel_conditioning))
        next_convolution: list[torch.Tensor] = []
        next_attention: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            if state.attention is not None:
                assert state.convolution is not None
                first, second = state.convolution[index].split(
                    (block.conv.in_channels, block.conv.out_channels), dim=1
                )
                convolution_state = ConvBlockState(
                    first=ConvState(
                        history=first, valid_frame_counts=state.valid_frame_counts
                    ),
                    second=ConvState(
                        history=second, valid_frame_counts=state.valid_frame_counts
                    ),
                )
                x, convolution_state, _ = block(
                    x,
                    timestep_embedding,
                    state.attention_mask,
                    convolution_state,
                    None,
                    state.attention[index],
                )
            else:
                x, convolution_state, attention_state = block(
                    x,
                    timestep_embedding,
                    state.attention_mask,
                    ConvBlockState(),
                    AttentionState(),
                )
                assert attention_state is not None
                assert attention_state.history is not None
                next_attention.append(attention_state.history)
            assert convolution_state is not None
            assert (
                convolution_state.first.history is not None
                and convolution_state.second.history is not None
            )
            next_convolution.append(
                torch.cat(
                    (convolution_state.first.history, convolution_state.second.history),
                    dim=1,
                )
            )
        x = self.final_layer(x, timestep_embedding).transpose(1, 2)
        if state.attention is not None:
            next_attention_window = state.attention
        else:
            next_attention_window = torch.stack(next_attention)
        return x, DiTState(
            convolution=torch.stack(next_convolution),
            attention=next_attention_window,
        )

    def forward_packed(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, padded_length, _ = x.shape
        frame_indices = torch.arange(padded_length, device=x.device)
        valid_frames = frame_indices.unsqueeze(0) < sequence_lengths.unsqueeze(1)
        x = x[valid_frames]
        timestep_embedding = timestep_embedding.squeeze(1)
        cumulative_sequence_lengths = torch.nn.functional.pad(
            sequence_lengths.cumsum(0, dtype=torch.int32), (1, 0)
        )
        causal_padding_frames = self.blocks[0].conv.kernel_size - 1
        sequence_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=x.device), sequence_lengths
        )
        real_frame_positions = (
            torch.arange(x.shape[0], device=x.device)
            + (sequence_ids + 1) * causal_padding_frames
        )
        real_frame_mask = torch.zeros(
            x.shape[0] + batch_size * causal_padding_frames,
            device=x.device,
            dtype=torch.bool,
        )
        real_frame_mask[real_frame_positions] = True
        for block in self.blocks:
            x = block.forward_packed(
                x,
                timestep_embedding,
                sequence_ids,
                cumulative_sequence_lengths,
                padded_length,
                real_frame_positions,
                real_frame_mask,
            )
        x = self.final_layer.forward_packed(x, timestep_embedding, sequence_ids)
        dense = x.new_zeros(batch_size, padded_length, self.out_channels)
        dense[valid_frames] = x
        return dense.transpose(1, 2)
