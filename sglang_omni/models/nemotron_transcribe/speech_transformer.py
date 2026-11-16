# SPDX-License-Identifier: Apache-2.0
"""Speech transformer with feature stacking, rotary positions and encoder compilation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo import config as dynamo_config

from sglang_omni.models.nemotron_transcribe.config import EncoderCompileMode
from sglang_omni.models.nemotron_transcribe.encoder_cuda_graph import (
    EncoderCudaGraphs,
    EncoderLayersForward,
)

ENCODER_RECOMPILE_LIMIT = 64


class FeatureStacking(nn.Module):
    """Stack subsampling_factor frames and project. proj has no bias."""

    def __init__(self, subsampling_factor: int, feat_in: int, feat_out: int) -> None:
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj = nn.Linear(subsampling_factor * feat_in, feat_out, bias=False)

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features is (batch, channels, time). Returns (batch, time', width)."""
        sequence = features.transpose(1, 2)
        batch, frames, channels = sequence.shape
        pad_frames = (
            self.subsampling_factor - frames % self.subsampling_factor
        ) % self.subsampling_factor
        if pad_frames:
            sequence = F.pad(sequence, (0, 0, 0, pad_frames))
        else:
            pass
        grouped = sequence.reshape(batch, -1, channels * self.subsampling_factor)
        output_lengths = (
            lengths + self.subsampling_factor - 1
        ) // self.subsampling_factor
        return self.proj(grouped), output_lengths


class RotaryEmbedding(nn.Module):
    """Rotary positions with CPU-computed frequencies and non-checkpoint buffers."""

    def __init__(
        self, head_dim: int, rotary_fraction: float, rope_base: float, max_length: int
    ) -> None:
        super().__init__()
        rotary_dim = int(head_dim * rotary_fraction)
        if rotary_dim < 2 or rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary dimension must be a positive even number, got {rotary_dim}"
            )
        else:
            pass
        self.rotary_dim = rotary_dim
        # note (yhayaran): SGLang builds this under a CUDA default device, so pin factories to CPU.
        inv_freq = 1.0 / (
            rope_base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")
                / rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos", torch.empty(0, device="cpu"), persistent=False)
        self.register_buffer("sin", torch.empty(0, device="cpu"), persistent=False)
        self.extend(max_length)

    def extend(self, length: int) -> None:
        if self.cos.numel() and self.cos.shape[0] >= length:
            return
        else:
            pass
        device = self.inv_freq.device
        positions = torch.arange(length, dtype=torch.float32, device="cpu")
        frequencies = torch.outer(
            positions, self.inv_freq.detach().to(dtype=torch.float32, device="cpu")
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.cos = embeddings.cos().to(device)
        self.sin = embeddings.sin().to(device)

    def apply_rotary(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate equal-length (batch, heads, time, dim) queries and keys."""
        frames = keys.shape[2]
        self.extend(frames)
        if self.cos.device != queries.device:
            self.cos = self.cos.to(device=queries.device)
            self.sin = self.sin.to(device=queries.device)
        else:
            pass
        cos = self.cos[:frames]
        sin = self.sin[:frames]
        return self.rotate(queries, cos, sin), self.rotate(keys, cos, sin)

    def rotate(
        self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        cos = cos.view(1, 1, cos.shape[0], self.rotary_dim).to(tokens.dtype)
        sin = sin.view(1, 1, sin.shape[0], self.rotary_dim).to(tokens.dtype)
        if self.rotary_dim == tokens.shape[-1]:
            rotated = tokens
            rest = None
        else:
            rotated = tokens[..., : self.rotary_dim]
            rest = tokens[..., self.rotary_dim :]
        half = rotated.shape[-1] // 2
        twisted = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
        rotated = rotated * cos + twisted * sin
        if rest is None:
            return rotated
        else:
            return torch.cat((rotated, rest), dim=-1)


class FeedForward(nn.Module):
    """GELU feed-forward. Sequential indexes 0 and 3 are the linears."""

    def __init__(self, model_dim: int, expansion: float, drop_rate: float) -> None:
        super().__init__()
        hidden = int(expansion * model_dim)
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden, model_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class SpeechAttention(nn.Module):
    """Fused QKV attention. q_norm and k_norm exist only when qk_norm is set."""

    def __init__(
        self,
        model_dim: int,
        head_count: int,
        qkv_bias: bool,
        qk_norm: bool,
        rope: RotaryEmbedding | None,
    ) -> None:
        super().__init__()
        if model_dim % head_count != 0:
            raise ValueError(
                f"model dim {model_dim} is not divisible by {head_count} heads"
            )
        else:
            pass
        self.head_count = head_count
        self.head_dim = model_dim // head_count
        self.model_dim = model_dim
        self.rope = rope
        self.w_qkv = nn.Linear(model_dim, 3 * model_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(model_dim, model_dim)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self, hidden: torch.Tensor, key_keep: torch.Tensor | None
    ) -> torch.Tensor:
        """key_keep is (batch, time) True where the key is valid. None means every frame is valid."""
        batch, frames, _ = hidden.shape
        mixed = self.w_qkv(hidden).view(
            batch, frames, 3, self.head_count, self.head_dim
        )
        queries, keys, values = mixed.permute(2, 0, 3, 1, 4).unbind(0)
        if self.q_norm is not None and self.k_norm is not None:
            queries = self.q_norm(queries).to(values.dtype)
            keys = self.k_norm(keys).to(values.dtype)
        else:
            pass
        if self.rope is not None:
            queries, keys = self.rope.apply_rotary(queries, keys)
        else:
            pass
        if key_keep is None:
            attended = F.scaled_dot_product_attention(queries, keys, values)
        else:
            attended = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=key_keep.view(batch, 1, 1, frames)
            )
        attended = (
            attended.transpose(1, 2).contiguous().view(batch, frames, self.model_dim)
        )
        return self.out_proj(attended)


class SpeechTransformerBlock(nn.Module):
    """Pre-norm residual block."""

    def __init__(
        self,
        model_dim: int,
        head_count: int,
        expansion: float,
        drop_rate: float,
        qkv_bias: bool,
        qk_norm: bool,
        rope: RotaryEmbedding | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = SpeechAttention(model_dim, head_count, qkv_bias, qk_norm, rope)
        self.dropout = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, expansion, drop_rate)

    def forward(
        self, hidden: torch.Tensor, key_keep: torch.Tensor | None
    ) -> torch.Tensor:
        hidden = hidden + self.dropout(self.attn(self.norm1(hidden), key_keep))
        hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
        return hidden


class SpeechTransformerEncoder(nn.Module):
    """Feature stacking, rotary pre-norm stack, channels-first output."""

    def __init__(
        self,
        feat_in: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        subsampling_factor: int = 8,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        ff_expansion: float = 4.0,
        pre_block_norm: bool = True,
        rope_base: float = 10000.0,
        rotary_fraction: float = 1.0,
        pos_emb_max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.subsampling_factor = subsampling_factor
        self.pre_encode = FeatureStacking(subsampling_factor, feat_in, d_model)
        self.dropout_pre_encoder = nn.Dropout(drop_rate)
        head_dim = d_model // n_heads
        self.pos_enc = RotaryEmbedding(
            head_dim, rotary_fraction, rope_base, pos_emb_max_len
        )
        if pre_block_norm:
            self.embed_norm = nn.LayerNorm(d_model)
        else:
            self.embed_norm = nn.Identity()
        self.layers = nn.ModuleList(
            [
                SpeechTransformerBlock(
                    d_model,
                    n_heads,
                    ff_expansion,
                    drop_rate,
                    qkv_bias,
                    qk_norm,
                    self.pos_enc,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.compiled_layers: EncoderLayersForward | None = None

    def forward_layers(
        self, hidden: torch.Tensor, key_keep: torch.Tensor | None
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, key_keep)
        return self.final_norm(hidden)

    def compile_layers(
        self,
        mode: EncoderCompileMode = "default",
        cuda_graphs: bool = False,
        dynamic: bool = False,
    ) -> None:
        # note (yhayaran): Encoder-owned graphs avoid Inductor/decoder pool conflicts.
        compiled = torch.compile(
            self.forward_layers, mode=mode, fullgraph=False, dynamic=dynamic
        )
        self.compiled_layers = EncoderCudaGraphs(compiled) if cuda_graphs else compiled

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        bypass_pre_encode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map (batch, bins, time), or (batch, time, width) with bypass, to (batch, width, time)."""
        if bypass_pre_encode:
            hidden = audio_signal
            lengths = length.to(torch.int64)
        else:
            hidden, lengths = self.pre_encode(audio_signal, length)
            lengths = lengths.to(torch.int64)
        hidden = self.dropout_pre_encoder(hidden)
        hidden = self.embed_norm(hidden)
        frames = hidden.shape[1]
        # note (yhayaran): A bool mask disables flash attention, so full windows skip it.
        if bool(torch.all(lengths >= frames).item()):
            key_keep = None
        else:
            positions = torch.arange(frames, device=hidden.device)
            key_keep = positions.unsqueeze(0) < lengths.unsqueeze(1).clamp(max=frames)
        if self.compiled_layers is None:
            hidden = self.forward_layers(hidden, key_keep)
        else:
            # note (yhayaran): The patch is thread-local and covers concurrent batch and padding shapes.
            with dynamo_config.patch(recompile_limit=ENCODER_RECOMPILE_LIMIT):
                hidden = self.compiled_layers(hidden, key_keep)
        return hidden.transpose(1, 2).contiguous(), lengths
