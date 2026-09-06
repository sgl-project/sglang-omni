# SPDX-License-Identifier: Apache-2.0
# Structure follows the Qwen3-ASR MLX path in
# sglang_omni/models/qwen3_asr/mlx/model.py. Module attribute names mirror the
# official ``openai/whisper-*`` checkpoint keys so weights load without a
# rename table.

from __future__ import annotations

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class WhisperAttention(nn.Module):
    """Multi-headed attention shared by the Whisper encoder and decoder.

    Whisper omits the key projection bias, so ``k_proj`` is built without one.
    Doing that here (rather than zero-filling a fused QKV shard, as the CUDA
    path does) keeps the module names identical to the checkpoint keys.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * num_heads) != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={embed_dim}"
                f" and num_heads={num_heads})."
            )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _shape(self, states: mx.array) -> mx.array:
        bsz, seq_len, _ = states.shape
        return states.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        bsz, seq_len, _ = hidden_states.shape
        # Cross-attention reads keys and values from the encoder output; self
        # attention reads them from its own input.
        kv_source = hidden_states if key_value_states is None else key_value_states

        query_states = self._shape(self.q_proj(hidden_states) * self.scaling)
        key_states = self._shape(self.k_proj(kv_source))
        value_states = self._shape(self.v_proj(kv_source))

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=1.0, mask=mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class WhisperEncoderLayer(nn.Module):
    """A single Whisper encoder block (pre-norm residual, GELU feed-forward)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            config.d_model, config.encoder_attention_heads
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    """Whisper audio encoder: two strided convolutions then transformer layers.

    Unlike Qwen3-ASR's encoder this one has no windowing or ragged block mask:
    Whisper always consumes a fixed 30 s mel window, so the convolution stack
    emits exactly ``max_source_positions`` frames and every position attends to
    every other one.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # MLX convolutions are NLC; the PyTorch checkpoint stores NCL kernels,
        # which `WhisperMlxModel.sanitize` transposes on load.
        self.conv1 = nn.Conv1d(
            config.num_mel_bins, config.d_model, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.d_model, config.d_model, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, input_features: mx.array) -> mx.array:
        """Encode a mel spectrogram.

        Args:
            input_features: ``(batch, num_mel_bins, frames)`` to match the
                PyTorch feature extractor's output. Transposed to MLX's NLC
                layout internally.
        """
        hidden_states = input_features.transpose(0, 2, 1)
        hidden_states = nn.gelu(self.conv1(hidden_states))
        hidden_states = nn.gelu(self.conv2(hidden_states))

        hidden_states = (
            hidden_states + self.embed_positions.weight[: hidden_states.shape[1]]
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


class WhisperInnerModel(nn.Module):
    """Container matching the checkpoint's ``model.*`` key prefix."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = WhisperEncoder(config)


class WhisperMlxModel(nn.Module):
    """Native MLX Whisper model.

    Only the audio encoder is implemented so far; the decoder (self-attention
    plus ``encoder_attn`` cross-attention) lands in a follow-up change.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = WhisperInnerModel(config)

    @property
    def encoder(self) -> WhisperEncoder:
        return self.model.encoder

    def encode(self, input_features: mx.array) -> mx.array:
        return self.model.encoder(input_features)

    _CONV_WEIGHTS = ("model.encoder.conv1.weight", "model.encoder.conv2.weight")

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert PyTorch checkpoint layout to MLX layout.

        PyTorch stores Conv1d kernels as ``(out, in, kernel)`` while MLX expects
        ``(out, kernel, in)``. Re-running this on already-converted weights must
        be a no-op, so the decision is made from the module's own expected shape
        rather than from a guess about which axis holds the kernel. Decoder
        tensors are dropped until the decoder is implemented.
        """
        sanitized = {}
        for k, v in weights.items():
            if not k.startswith("model.encoder."):
                continue
            if k in self._CONV_WEIGHTS:
                v = self._to_mlx_conv1d(k, v)
            sanitized[k] = v
        return sanitized

    def _to_mlx_conv1d(self, key: str, value: mx.array) -> mx.array:
        """Transpose a Conv1d kernel into MLX's ``(out, kernel, in)`` layout."""
        conv = self.model.encoder.conv1 if "conv1" in key else self.model.encoder.conv2
        expected = conv.weight.shape
        if value.ndim != 3:
            raise ValueError(f"{key}: expected a 3D Conv1d kernel, got {value.shape}")
        if value.shape == expected:
            return value
        transposed = value.transpose(0, 2, 1)
        if transposed.shape != expected:
            raise ValueError(
                f"{key}: cannot map {value.shape} onto MLX Conv1d weight {expected}"
            )
        return transposed


Model = WhisperMlxModel
ModelArgs = ModelConfig
