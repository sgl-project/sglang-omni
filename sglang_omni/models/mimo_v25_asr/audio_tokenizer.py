# SPDX-License-Identifier: Apache-2.0
"""Input-only MiMo audio tokenizer encoder.

This module intentionally contains no audio decoder, vocoder, ISTFT, or
waveform-generation code.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sgl_kernel.flash_attn import flash_attn_varlen_func
from torch import nn


def _sequence_mask(
    inputs: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, target_len = inputs.shape[:2]
    positions = torch.arange(target_len, device=inputs.device)
    mask = positions < lengths.reshape(batch, 1)
    mask = mask.unsqueeze(-1)
    unpacking_index = torch.cumsum(mask.to(torch.int64).reshape(-1), dim=0) - 1
    return mask, unpacking_index


def _unpack_packed(packed: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch = lengths.shape[0]
    max_len = int(torch.max(lengths).item())
    mask = (
        torch.arange(max_len, device=packed.device)[None, :]
        < lengths[:, None].to(packed.device)
    ).unsqueeze(-1)
    unpacking_index = torch.cumsum(mask.to(torch.int64).reshape(-1), dim=0) - 1
    unpacked = torch.index_select(packed, 0, unpacking_index).view(
        batch, max_len, packed.shape[-1]
    )
    return torch.where(mask, unpacked, 0)


def _packed_position_ids(lengths: torch.Tensor) -> torch.Tensor:
    offsets = torch.cat(
        [torch.zeros(1, device=lengths.device, dtype=lengths.dtype), lengths[:-1]]
    ).cumsum(0)
    return torch.arange(lengths.sum(), device=lengths.device) - torch.repeat_interleave(
        offsets, lengths
    )


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


class _RotaryEmbedding(nn.Module):
    def __init__(self, theta: float, head_dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.outer(
            position_ids.float(), self.inv_freq.float().to(position_ids.device)
        )
        embedding = torch.cat([frequencies, frequencies], dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


class _Codebook(nn.Module):
    """Inference-only codebook with checkpoint-compatible buffer names."""

    def __init__(self, size: int, dimension: int) -> None:
        super().__init__()
        self.register_buffer("inited", torch.ones(1))
        self.register_buffer("cluster_size", torch.zeros(size))
        self.register_buffer("embed", torch.empty(size, dimension))
        self.register_buffer("embed_avg", torch.empty(size, dimension))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        codebook = self.embed.to(values)
        distance = -(
            values.pow(2).sum(1, keepdim=True)
            - 2 * values @ codebook.T
            + codebook.pow(2).sum(1, keepdim=False)[None, :]
        )
        return distance.argmax(dim=-1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)


class _VectorQuantization(nn.Module):
    def __init__(self, size: int, dimension: int) -> None:
        super().__init__()
        self.project_in = nn.Identity()
        self.project_out = nn.Identity()
        self._codebook = _Codebook(size, dimension)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(values)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self._codebook.decode(indices)


class _ResidualVectorQuantization(nn.Module):
    def __init__(self, sizes: Sequence[int], dimension: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_VectorQuantization(size, dimension) for size in sizes]
        )

    def encode(self, values: torch.Tensor, n_q: int | None = None) -> torch.Tensor:
        residual = values.float()
        codes: list[torch.Tensor] = []
        for layer in self.layers[: n_q or len(self.layers)]:
            indices = layer.encode(residual)
            residual = residual - layer.decode(indices)
            codes.append(indices)
        return torch.stack(codes)


class _ResidualVectorQuantizer(nn.Module):
    def __init__(self, sizes: Sequence[int], dimension: int) -> None:
        super().__init__()
        self.vq = _ResidualVectorQuantization(sizes, dimension)

    def encode(self, values: torch.Tensor, n_q: int | None = None) -> torch.Tensor:
        return self.vq.encode(values, n_q=n_q)


class _Attention(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.head_dim = dimension // heads
        self.window_size = window_size
        self.causal = causal
        self.k_proj = nn.Linear(dimension, dimension, bias=False)
        self.v_proj = nn.Linear(dimension, dimension, bias=True)
        self.q_proj = nn.Linear(dimension, dimension, bias=True)
        self.out_proj = nn.Linear(dimension, dimension, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        query = self.q_proj(hidden_states).view(-1, self.heads, self.head_dim)
        key = self.k_proj(hidden_states).view(-1, self.heads, self.head_dim)
        value = self.v_proj(hidden_states).view(-1, self.heads, self.head_dim)
        cos, sin = rotary
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        cumulative = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
        max_length = int(lengths.max().item())
        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cumulative,
            cumulative,
            max_length,
            max_length,
            causal=self.causal,
            window_size=self.window_size,
        )
        return self.out_proj(output.reshape(-1, self.dimension))


class _TransformerLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        feed_forward_dim: int,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.self_attn = _Attention(
            dimension,
            heads,
            causal=causal,
            window_size=window_size,
        )
        self.self_attn_layer_norm = nn.LayerNorm(dimension)
        self.fc1 = nn.Linear(dimension, feed_forward_dim)
        self.fc2 = nn.Linear(feed_forward_dim, dimension)
        self.final_layer_norm = nn.LayerNorm(dimension)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.self_attn_layer_norm(hidden_states),
            lengths,
            rotary,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden_states))))
        return residual + hidden_states


class MiMoAudioTokenizerEncoder(nn.Module):
    """Native input encoder matching `MiMo-Audio-Tokenizer` weight names."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        dimension = int(config["d_model"])
        heads = int(config["encoder_attention_heads"])
        self.skip_layer_idx = int(config["encoder_skip_layer_id"])
        self.stride_size = int(config["stride_size"])
        self.avg_pooler = int(config["avg_pooler"])
        self.conv1 = nn.Conv1d(
            int(config["n_mels"]),
            dimension,
            int(config["kernel_size"]),
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            dimension,
            dimension,
            int(config["kernel_size"]),
            stride=self.stride_size,
            padding=1,
        )
        self.position_embedding = _RotaryEmbedding(
            float(config["rope_theta"]),
            dimension // heads,
        )
        window = tuple(int(value) for value in config["encoder_attn_window_size"])
        self.layers = nn.ModuleList(
            [
                _TransformerLayer(
                    dimension,
                    heads,
                    int(config["encoder_ffn_dim"]),
                    causal=bool(config["encoder_causal"]),
                    window_size=window,
                )
                for _ in range(int(config["encoder_layers"]))
            ]
        )
        self.layer_norm = nn.LayerNorm(dimension)
        self.down_sample_layer = nn.Sequential(
            nn.Conv1d(
                dimension,
                dimension,
                self.avg_pooler,
                self.avg_pooler,
                bias=False,
            ),
            nn.GELU(),
        )
        self.down_sample_norm = nn.LayerNorm(dimension)
        self.quantizer = _ResidualVectorQuantizer(
            [int(size) for size in config["codebook_size"]],
            dimension,
        )

    def _output_lengths(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        kernel = int(self.config["kernel_size"])
        target = mel_lengths + 3 - kernel
        return (target + 2 - kernel) // self.stride_size + 1

    @torch.no_grad()
    def encode(
        self,
        *,
        input_features: torch.Tensor,
        input_lens: torch.Tensor,
        return_codes_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if return_codes_only is not True:
            raise ValueError("Milestone 1 tokenizer supports codes-only encoding")
        input_lens = input_lens.to(input_features.device, dtype=torch.int64)
        hidden = _unpack_packed(input_features, input_lens).transpose(1, 2)
        hidden = F.gelu(self.conv1(hidden.to(self.conv1.weight)))
        hidden = F.gelu(self.conv2(hidden)).transpose(1, 2)
        output_lengths = self._output_lengths(input_lens)
        mask, unpacking_index = _sequence_mask(hidden, output_lengths)
        hidden = torch.masked_select(hidden, mask).view(
            int(output_lengths.sum().item()), int(self.config["d_model"])
        )
        positions = _packed_position_ids(output_lengths).long()
        rotary = self.position_embedding(positions, hidden.dtype)

        skip: torch.Tensor | float = 0.0
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, output_lengths, rotary)
            if index == self.skip_layer_idx - 1:
                skip = hidden.clone()
        hidden = self.layer_norm(hidden + skip)

        batch, target_len = mask.shape[:2]
        hidden = torch.index_select(hidden, 0, unpacking_index).view(
            batch, target_len, int(self.config["d_model"])
        )
        if target_len % self.avg_pooler:
            padding = self.avg_pooler - target_len % self.avg_pooler
            hidden = F.pad(hidden, (0, 0, 0, padding))
        hidden = self.down_sample_layer(hidden.transpose(1, 2)).transpose(1, 2)
        output_lengths = output_lengths.div(self.avg_pooler, rounding_mode="floor") + (
            output_lengths % self.avg_pooler != 0
        ).to(output_lengths.dtype)
        mask, _ = _sequence_mask(hidden, output_lengths)
        hidden = torch.masked_select(hidden, mask).view(
            int(output_lengths.sum().item()), int(self.config["d_model"])
        )
        hidden = self.down_sample_norm(hidden)
        return self.quantizer.encode(hidden.float()), output_lengths

    @classmethod
    def from_checkpoint(
        cls,
        snapshot_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> MiMoAudioTokenizerEncoder:
        snapshot = Path(snapshot_path)
        config = json.loads((snapshot / "config.json").read_text())
        model = cls(config).to(device=device, dtype=dtype).eval()
        state = model.state_dict()
        expected = set(state)
        loaded: set[str] = set()
        checkpoint = snapshot / "model.safetensors"
        with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
            # `safe_open` is not iterable in the pinned safetensors build.
            for checkpoint_name in tensors.keys():  # noqa: SIM118
                if not checkpoint_name.startswith("encoder."):
                    continue
                name = checkpoint_name.removeprefix("encoder.")
                if name not in state:
                    raise ValueError(
                        f"Unexpected MiMo tokenizer encoder weight {checkpoint_name}"
                    )
                destination = state[name]
                destination.copy_(
                    tensors.get_tensor(checkpoint_name).to(
                        device=destination.device,
                        dtype=destination.dtype,
                    )
                )
                loaded.add(name)
        missing = expected - loaded
        if missing:
            raise ValueError(
                "Missing MiMo tokenizer encoder weights: "
                + ", ".join(sorted(missing)[:10])
            )
        return model


__all__ = ["MiMoAudioTokenizerEncoder"]
