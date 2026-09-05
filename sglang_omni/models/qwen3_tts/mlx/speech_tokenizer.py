# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-TTS (Copyright 2025 Prince Canuma and contributors).
"""MLX Qwen3-TTS speech tokenizer: codec frames <-> 24 kHz waveform.

The decoder ("code2wav") is a split residual vector quantiser, a small
sliding-window transformer, two ConvNeXt upsampling stages and a stack of
Snake/ConvTranspose blocks. Everything downstream of the quantiser runs in NLC
(``[batch, time, channels]``) because that is MLX's convolution layout; only
the quantiser and the final waveform are NCL.

One deliberate divergence from mlx-audio: the decoder transformer applies
sliding-window attention. The official configuration declares
``layer_types == ["sliding_attention"] * num_hidden_layers`` with
``sliding_window == 72`` as a computed property, so it never appears in
``config.json``; mlx-audio stores the window on the attention module but never
masks with it. That is invisible while decoding in short chunks and diverges
once a pass exceeds 72 frames (5.76 s at 12.5 Hz) -- which is the norm on the
voice-cloning path, where reference and generated codes are decoded together in
a single pass.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import TokenizerConfig, TokenizerDecoderConfig, TokenizerEncoderConfig


def sliding_causal_mask(
    query_len: int,
    key_len: int,
    window: int | None,
    dtype: mx.Dtype,
) -> Optional[mx.array]:
    """Additive mask for causal attention restricted to a trailing window.

    Queries occupy the last ``query_len`` absolute positions of ``key_len``.
    A key at ``j`` is visible to a query at ``i`` when ``j <= i`` and
    ``j > i - window``, matching the reference
    ``kv_idx > q_idx - sliding_window`` overlay on a causal mask.
    """
    if query_len == 1 and (window is None or key_len <= window):
        return None

    offset = key_len - query_len
    queries = mx.arange(offset, offset + query_len)[:, None]
    keys = mx.arange(key_len)[None, :]
    allowed = keys <= queries
    if window is not None:
        allowed = allowed & (keys > queries - window)
    return mx.where(allowed, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


class SnakeBeta(nn.Module):
    """``x + sin^2(alpha * x) / beta``, with log-parameterised alpha/beta."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))
        self.eps = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + (1.0 / (beta + self.eps)) * mx.square(mx.sin(x * alpha))


class CausalConv1d(nn.Module):
    """Left-padded convolution with an optional streaming tail buffer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = (kernel_size - 1) * dilation + 1 - stride
        self._buffer: mx.array | None = None
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        return self.conv(x)

    def step(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            if self._buffer is None:
                x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
            else:
                x = mx.concatenate([self._buffer, x], axis=1)
            self._buffer = x[:, -self.padding :, :]
        return self.conv(x)

    def reset_state(self) -> None:
        self._buffer = None


class CausalTransposeConv1d(nn.Module):
    """Transposed convolution trimmed on the right to stay causal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0
        )
        self.trim_right = kernel_size - stride

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.trim_right > 0:
            x = x[:, : -self.trim_right, :]
        return x


class ConvNeXtBlock(nn.Module):
    """Depthwise causal conv, LayerNorm, pointwise MLP, scaled residual."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.ones((dim,)) * 1e-6

    def _body(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        return self.gamma * x

    def __call__(self, x: mx.array) -> mx.array:
        return x + self._body(self.dwconv(x))

    def step(self, x: mx.array) -> mx.array:
        return x + self._body(self.dwconv.step(x))


class DecoderRMSNorm(nn.Module):
    """RMS norm accumulated in float32."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        promoted = x.astype(mx.float32)
        variance = mx.mean(promoted**2, axis=-1, keepdims=True)
        normed = promoted * mx.rsqrt(variance + self.eps)
        return (self.weight * normed).astype(x.dtype)


class LayerScale(nn.Module):
    """Per-channel residual scale."""

    def __init__(self, channels: int, initial_scale: float = 0.01) -> None:
        super().__init__()
        self.scale = mx.ones((channels,)) * initial_scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x


class DecoderAttention(nn.Module):
    """Sliding-window causal self-attention."""

    def __init__(self, config: TokenizerDecoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, length, _ = x.shape

        queries = (
            self.q_proj(x)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        offset = 0 if cache is None else cache.offset
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class DecoderMLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, config: TokenizerDecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderTransformerLayer(nn.Module):
    """Pre-norm layer with per-branch learned residual scales."""

    def __init__(self, config: TokenizerDecoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = DecoderAttention(config, layer_idx)
        self.mlp = DecoderMLP(config)
        self.input_layernorm = DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DecoderRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn_layer_scale = LayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )
        self.mlp_layer_scale = LayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache)
        x = residual + self.self_attn_layer_scale(x)

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + self.mlp_layer_scale(x)


class DecoderTransformer(nn.Module):
    """Latent-space transformer between the quantiser and the upsampler."""

    def __init__(self, config: TokenizerDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = [
            DecoderTransformerLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

    def make_cache(self) -> List[Any]:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        x = self.input_proj(inputs_embeds)
        offset = 0 if cache is None else cache[0].offset
        mask = sliding_causal_mask(
            x.shape[1],
            offset + x.shape[1],
            self.config.sliding_window,
            x.dtype,
        )

        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, mask=mask, cache=layer_cache)

        return self.output_proj(self.norm(x))


class EuclideanCodebook(nn.Module):
    """Codebook lookup. The checkpoint's ``cluster_usage``/``embedding_sum``
    pair is folded into a single embedding table during sanitisation."""

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.embed = nn.Embedding(codebook_size, dim)

    def decode(self, codes: mx.array) -> mx.array:
        return self.embed(codes)


class VectorQuantization(nn.Module):
    """One codebook plus its optional output projection."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        codebook_dim = codebook_dim or dim
        self.project_out = nn.Linear(codebook_dim, dim) if codebook_dim != dim else None
        self.codebook = EuclideanCodebook(codebook_dim, codebook_size)
        self.codebook_size = codebook_size

    def decode(self, codes: mx.array) -> mx.array:
        quantized = self.codebook.decode(codes)
        if self.project_out is not None:
            quantized = self.project_out(quantized)
        return mx.transpose(quantized, (0, 2, 1))


class ResidualVectorQuantization(nn.Module):
    """Sum of a stack of codebook lookups."""

    def __init__(
        self,
        num_quantizers: int,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.layers = [
            VectorQuantization(dim, codebook_size, codebook_dim)
            for _ in range(num_quantizers)
        ]

    def decode(self, codes: mx.array) -> mx.array:
        quantized = None
        for index, layer_codes in enumerate(codes):
            decoded = self.layers[index].decode(layer_codes)
            quantized = decoded if quantized is None else quantized + decoded
        return quantized


class ResidualVectorQuantizer(nn.Module):
    """Residual VQ with 1x1 input/output projections."""

    def __init__(
        self,
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        n_q: int = 8,
        bins: int = 1024,
        force_projection: bool = False,
    ) -> None:
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins

        needs_input = self.input_dimension != dimension or force_projection
        needs_output = self.output_dimension != dimension or force_projection
        self.input_proj = (
            nn.Conv1d(self.input_dimension, dimension, 1, bias=False)
            if needs_input
            else None
        )
        self.output_proj = (
            nn.Conv1d(dimension, self.output_dimension, 1, bias=False)
            if needs_output
            else None
        )
        self.vq = ResidualVectorQuantization(
            num_quantizers=n_q, dim=dimension, codebook_size=bins
        )

    def decode(self, codes: mx.array) -> mx.array:
        quantized = self.vq.decode(mx.transpose(codes, (1, 0, 2)))
        if self.output_proj is not None:
            quantized = mx.transpose(quantized, (0, 2, 1))
            quantized = self.output_proj(quantized)
            quantized = mx.transpose(quantized, (0, 2, 1))
        return quantized


class SplitResidualVectorQuantizer(nn.Module):
    """Semantic codebook(s) plus the acoustic remainder."""

    def __init__(
        self,
        n_q: int = 8,
        n_q_semantic: int = 1,
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        bins: int = 1024,
    ) -> None:
        super().__init__()
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        self.rvq_first = ResidualVectorQuantizer(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q_semantic,
            bins=bins,
            force_projection=True,
        )
        self.rvq_rest = ResidualVectorQuantizer(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q - n_q_semantic,
            bins=bins,
            force_projection=True,
        )

    def decode(self, codes: mx.array) -> mx.array:
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


class DecoderResidualUnit(nn.Module):
    """Snake -> dilated conv -> Snake -> 1x1 conv, with a residual."""

    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))

    def step(self, x: mx.array) -> mx.array:
        return x + self.conv2.step(self.act2(self.conv1.step(self.act1(x))))


class DecoderBlockUpsample(nn.Module):
    """Causal transposed convolution with a streaming overlap-add tail."""

    def __init__(self, in_dim: int, out_dim: int, upsample_rate: int) -> None:
        super().__init__()
        kernel_size = 2 * upsample_rate
        self.conv = nn.ConvTranspose1d(
            in_dim, out_dim, kernel_size, stride=upsample_rate, padding=0
        )
        self.trim_right = kernel_size - upsample_rate
        self._overflow: mx.array | None = None

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.trim_right > 0:
            x = x[:, : -self.trim_right, :]
        return x

    def step(self, x: mx.array) -> mx.array:
        y = self.conv(x)
        if self._overflow is not None:
            width = self._overflow.shape[1]
            y = mx.concatenate(
                [y[:, :width, :] + self._overflow, y[:, width:, :]], axis=1
            )
        if self.trim_right > 0:
            self._overflow = y[:, -self.trim_right :, :]
            y = y[:, : -self.trim_right, :]
        return y

    def reset_state(self) -> None:
        self._overflow = None


class DecoderBlock(nn.Module):
    """One waveform upsampling stage.

    ``block`` is a list to match the checkpoint's ``ModuleList`` keys:
    ``0`` Snake, ``1`` upsample, ``2..4`` residual units.
    """

    def __init__(self, config: TokenizerDecoderConfig, layer_idx: int) -> None:
        super().__init__()
        in_dim = config.decoder_dim // (2**layer_idx)
        out_dim = config.decoder_dim // (2 ** (layer_idx + 1))
        self.block = [
            SnakeBeta(in_dim),
            DecoderBlockUpsample(in_dim, out_dim, config.upsample_rates[layer_idx]),
            DecoderResidualUnit(out_dim, dilation=1),
            DecoderResidualUnit(out_dim, dilation=3),
            DecoderResidualUnit(out_dim, dilation=9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x

    def step(self, x: mx.array) -> mx.array:
        x = self.block[0](x)
        x = self.block[1].step(x)
        for unit in self.block[2:]:
            x = unit.step(x)
        return x


class _PaddedConv(nn.Module):
    """Left-padded convolution wrapper matching a ``.conv.*`` weight prefix."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
        self.kernel_size = kernel_size
        self._buffer: mx.array | None = None

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)]))

    def step(self, x: mx.array) -> mx.array:
        padding = self.kernel_size - 1
        if padding > 0:
            if self._buffer is None:
                x = mx.pad(x, [(0, 0), (padding, 0), (0, 0)])
            else:
                x = mx.concatenate([self._buffer, x], axis=1)
            self._buffer = x[:, -padding:, :]
        return self.conv(x)

    def reset_state(self) -> None:
        self._buffer = None


class Qwen3TTSSpeechTokenizerDecoder(nn.Module):
    """Codec frames -> 24 kHz waveform."""

    def __init__(self, config: TokenizerDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.total_upsample = math.prod(
            list(config.upsample_rates) + list(config.upsampling_ratios)
        )
        self._transformer_cache: List[Any] | None = None

        self.pre_transformer = DecoderTransformer(config)
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=config.num_semantic_quantizers,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )
        self.pre_conv = CausalConv1d(
            config.codebook_dim, config.latent_dim, kernel_size=3
        )
        self.upsample = [
            [
                CausalTransposeConv1d(
                    config.latent_dim, config.latent_dim, factor, factor
                ),
                ConvNeXtBlock(config.latent_dim),
            ]
            for factor in config.upsampling_ratios
        ]
        output_dim = config.decoder_dim // (2 ** len(config.upsample_rates))
        self.decoder = [
            _PaddedConv(config.latent_dim, config.decoder_dim, 7),
            *[DecoderBlock(config, i) for i in range(len(config.upsample_rates))],
            SnakeBeta(output_dim),
            _PaddedConv(output_dim, 1, 7),
        ]

    def _check_codes(self, codes: mx.array) -> None:
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} codebooks, "
                f"got {codes.shape[1]}"
            )

    def __call__(self, codes: mx.array) -> mx.array:
        """``codes`` is ``[batch, num_quantizers, time]``; returns ``[batch, 1, samples]``."""
        self._check_codes(codes)

        hidden = mx.transpose(self.quantizer.decode(codes), (0, 2, 1))
        hidden = self.pre_conv(hidden)
        hidden = self.pre_transformer(hidden)

        for transpose_conv, convnext in self.upsample:
            hidden = convnext(transpose_conv(hidden))

        for layer in self.decoder:
            hidden = layer(hidden)

        return mx.clip(mx.transpose(hidden, (0, 2, 1)), -1.0, 1.0)

    def reset_streaming_state(self) -> None:
        """Drop every convolution buffer, overlap tail, and the KV cache."""
        self._transformer_cache = None
        for _, module in self.named_modules():
            reset = getattr(module, "reset_state", None)
            if callable(reset):
                reset()

    def streaming_step(self, codes: mx.array) -> mx.array:
        """Decode only the new ``codes``, carrying convolution and KV state."""
        self._check_codes(codes)
        if self._transformer_cache is None:
            self._transformer_cache = self.pre_transformer.make_cache()

        hidden = mx.transpose(self.quantizer.decode(codes), (0, 2, 1))
        hidden = self.pre_conv.step(hidden)
        hidden = self.pre_transformer(hidden, cache=self._transformer_cache)

        for transpose_conv, convnext in self.upsample:
            hidden = convnext.step(transpose_conv(hidden))

        hidden = self.decoder[0].step(hidden)
        for block in self.decoder[1:-2]:
            hidden = block.step(hidden)
        hidden = self.decoder[-2](hidden)
        hidden = self.decoder[-1].step(hidden)

        return mx.clip(mx.transpose(hidden, (0, 2, 1)), -1.0, 1.0)

    # Names of the mutable streaming buffers scattered through the decoder:
    # convolution tails and the transposed-convolution overlap-add remainder.
    _STREAM_BUFFER_ATTRS = ("_buffer", "_overflow")

    def new_streaming_session(self) -> dict:
        """An empty per-request streaming state.

        The decoder's convolution tails, overlap-add remainders and transformer
        KV live on the module, so concurrent requests would otherwise overwrite
        each other. A session holds one request's values; install it with
        :meth:`streaming_session` around that request's steps.
        """
        return {"buffers": {}, "cache": None}

    @contextmanager
    def streaming_session(self, session: dict) -> Iterator[None]:
        """Install ``session``'s streaming state for the duration of the block.

        Only references are swapped, never tensor data, so entering and leaving
        a session costs nothing measurable.
        """
        saved_buffers: dict[tuple[str, str], Any] = {}
        saved_cache = self._transformer_cache

        for path, module in self.named_modules():
            for attr in self._STREAM_BUFFER_ATTRS:
                if hasattr(module, attr):
                    saved_buffers[(path, attr)] = getattr(module, attr)
                    setattr(module, attr, session["buffers"].get((path, attr)))
        self._transformer_cache = session["cache"]

        try:
            yield
        finally:
            for (path, attr), previous in saved_buffers.items():
                module = self._module_at(path)
                session["buffers"][(path, attr)] = getattr(module, attr)
                setattr(module, attr, previous)
            session["cache"] = self._transformer_cache
            self._transformer_cache = saved_cache

    def _module_at(self, path: str):
        """Resolve a dotted ``named_modules`` path, list indices included."""
        target = self
        for part in path.split("."):
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)
        return target

    def chunked_decode(
        self,
        codes: mx.array,
        chunk_size: int = 300,
        left_context_size: int | None = None,
    ) -> mx.array:
        """Decode long sequences in bounded-memory chunks.

        Each chunk is re-decoded with ``left_context_size`` frames of discarded
        history. Because attention is limited to ``sliding_window`` frames, a
        context of ``sliding_window - 1`` makes this *exact* rather than an
        approximation -- every query in the chunk sees its full window, and the
        convolution stack's receptive field is far shorter than that. Passing a
        smaller context trades accuracy for memory.
        """
        if left_context_size is None:
            window = self.config.sliding_window
            left_context_size = 0 if window is None else window - 1

        waveforms = []
        start = 0
        while start < codes.shape[-1]:
            end = min(start + chunk_size, codes.shape[-1])
            context = min(left_context_size, start)
            chunk = self(codes[..., start - context : end])
            waveforms.append(chunk[..., context * self.total_upsample :])
            start = end
        return mx.concatenate(waveforms, axis=-1)


# --------------------------------------------------------------------------
# Encoder: waveform -> codec frames (Base voice cloning only)
#
# Mirrors the checkpoint's own module layout -- which is that of
# ``transformers.MimiModel``, since the official encoder subclasses it -- so
# weight keys need no remapping beyond convolution layout and codebook folding.
# --------------------------------------------------------------------------


def _causal_pad_width(
    length: int, kernel_size: int, stride: int, dilation: int
) -> tuple[int, int]:
    """Left/right padding for a causal strided convolution.

    Left padding is the whole ``effective_kernel - stride`` budget; the right
    side gets only the extra needed to round the output up to a whole frame.
    """
    effective = (kernel_size - 1) * dilation + 1
    total = effective - stride
    frames = max(length + total - effective, 0) / stride + 1.0
    ideal = (math.ceil(frames) - 1) * stride + effective - total
    return total, max(0, ideal - length)


class StreamableConv1d(nn.Module):
    """Causal convolution with reference-compatible length rounding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        left, right = _causal_pad_width(
            x.shape[1], self.kernel_size, self.stride, self.dilation
        )
        if left or right:
            x = mx.pad(x, [(0, 0), (left, right), (0, 0)], mode=self.pad_mode)
        return self.conv(x)


class _Elu(nn.Module):
    """Parameter-free ELU, kept as a module so SEANet layer indices line up
    with the checkpoint's flat ``ModuleList``."""

    def __call__(self, x: mx.array) -> mx.array:
        return nn.elu(x, alpha=1.0)


class SeanetResnetBlock(nn.Module):
    """Pre-activation residual block; ``block`` indices 1 and 3 hold the convs."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        compress: int,
        pad_mode: str,
    ) -> None:
        super().__init__()
        hidden = dim // compress
        self.block = [
            _Elu(),
            StreamableConv1d(
                dim, hidden, kernel_size, dilation=dilation, pad_mode=pad_mode
            ),
            _Elu(),
            StreamableConv1d(hidden, dim, 1, pad_mode=pad_mode),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for layer in self.block:
            x = layer(x)
        return x + residual


class SeanetEncoder(nn.Module):
    """Strided convolutional encoder, one flat layer list.

    Downsampling ratios are applied in reverse of ``upsampling_ratios``, each
    with kernel ``2 * ratio``, doubling the channel count.
    """

    def __init__(self, config: TokenizerEncoderConfig) -> None:
        super().__init__()
        pad_mode = config.pad_mode
        layers: List[Any] = [
            StreamableConv1d(
                config.audio_channels,
                config.num_filters,
                config.kernel_size,
                pad_mode=pad_mode,
            )
        ]
        channels = config.num_filters
        for ratio in reversed(config.upsampling_ratios):
            dilation = 1
            for _ in range(config.num_residual_layers):
                layers.append(
                    SeanetResnetBlock(
                        channels,
                        config.residual_kernel_size,
                        dilation,
                        config.compress,
                        pad_mode,
                    )
                )
                dilation *= config.dilation_growth_rate
            layers.append(_Elu())
            layers.append(
                StreamableConv1d(
                    channels,
                    channels * 2,
                    ratio * 2,
                    stride=ratio,
                    pad_mode=pad_mode,
                )
            )
            channels *= 2
        layers.append(_Elu())
        layers.append(
            StreamableConv1d(
                channels, config.hidden_size, config.last_kernel_size, pad_mode=pad_mode
            )
        )
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderAttention(nn.Module):
    """Sliding-window causal attention with separate q/k/v projections."""

    def __init__(self, config: TokenizerEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        bias = config.attention_bias
        inner = self.num_heads * self.head_dim
        kv_inner = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_inner, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_inner, bias=bias)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=bias)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        batch, length, _ = x.shape
        queries = (
            self.q_proj(x)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        queries = self.rope(queries)
        keys = self.rope(keys)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class EncoderMLP(nn.Module):
    """Two-layer feed-forward with exact (erf) GELU, per ``hidden_act``."""

    def __init__(self, config: TokenizerEncoderConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class EncoderTransformerLayer(nn.Module):
    """Pre-norm layer with LayerNorm and learned residual scales."""

    def __init__(self, config: TokenizerEncoderConfig) -> None:
        super().__init__()
        self.self_attn = EncoderAttention(config)
        self.mlp = EncoderMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps
        )
        self.self_attn_layer_scale = LayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )
        self.mlp_layer_scale = LayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn_layer_scale(
            self.self_attn(self.input_layernorm(x), mask=mask)
        )
        return x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))


class EncoderTransformer(nn.Module):
    """Transformer over encoder frames, in NCL like the surrounding convs."""

    def __init__(self, config: TokenizerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = [
            EncoderTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        mask = sliding_causal_mask(
            x.shape[1], x.shape[1], self.config.sliding_window, x.dtype
        )
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class EncoderEuclideanCodebook(nn.Module):
    """Nearest-codeword lookup over a folded codebook table."""

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.embed = nn.Embedding(codebook_size, dim)

    def encode(self, x: mx.array) -> mx.array:
        table = self.embed.weight.astype(mx.float32)
        half_sq = (table**2).sum(axis=-1) / 2
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        return mx.argmin(half_sq - flat @ table.T, axis=-1).reshape(x.shape[:-1])

    def decode(self, codes: mx.array) -> mx.array:
        return self.embed(codes)


class EncoderVectorQuantization(nn.Module):
    """One residual stage. ``codebook_dim == dim`` here, so no projections."""

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.codebook = EncoderEuclideanCodebook(dim, codebook_size)

    def encode(self, x: mx.array) -> mx.array:
        return self.codebook.encode(mx.swapaxes(x, -1, -2))

    def decode(self, codes: mx.array) -> mx.array:
        return mx.swapaxes(self.codebook.decode(codes), -1, -2)


class EncoderResidualVectorQuantizer(nn.Module):
    """Residual VQ stack with 1x1 input/output projections."""

    def __init__(
        self,
        dim: int,
        input_dim: int,
        output_dim: int,
        num_quantizers: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, dim, 1, bias=False)
        self.output_proj = nn.Conv1d(dim, output_dim, 1, bias=False)
        self.layers = [
            EncoderVectorQuantization(dim, codebook_size) for _ in range(num_quantizers)
        ]

    def encode(self, x: mx.array) -> mx.array:
        projected = mx.swapaxes(self.input_proj(mx.swapaxes(x, -1, -2)), -1, -2)
        codes = []
        residual = projected
        for layer in self.layers:
            indices = layer.encode(residual)
            residual = (
                residual.astype(mx.float32) - layer.decode(indices).astype(mx.float32)
            ).astype(projected.dtype)
            codes.append(indices)
        # [num_quantizers, batch, time] -> [batch, num_quantizers, time]
        return mx.swapaxes(mx.stack(codes, axis=0), 0, 1)


class EncoderSplitResidualVectorQuantizer(nn.Module):
    """Semantic stage plus the acoustic remainder, both fed the same input."""

    def __init__(self, config: TokenizerEncoderConfig) -> None:
        super().__init__()
        dim = config.vector_quantization_hidden_dimension
        self.num_quantizers = config.num_quantizers
        self.semantic_residual_vector_quantizer = EncoderResidualVectorQuantizer(
            dim,
            config.hidden_size,
            config.hidden_size,
            config.num_semantic_quantizers,
            config.codebook_size,
        )
        self.acoustic_residual_vector_quantizer = EncoderResidualVectorQuantizer(
            dim,
            config.hidden_size,
            config.hidden_size,
            config.num_quantizers - config.num_semantic_quantizers,
            config.codebook_size,
        )

    def encode(self, x: mx.array) -> mx.array:
        codes = self.semantic_residual_vector_quantizer.encode(x)
        if self.acoustic_residual_vector_quantizer.layers:
            codes = mx.concatenate(
                [codes, self.acoustic_residual_vector_quantizer.encode(x)], axis=1
            )
        return codes


class Qwen3TTSSpeechTokenizerEncoder(nn.Module):
    """Waveform -> codec frames.

    Attribute names match the checkpoint (and ``MimiModel``): the inner
    ``encoder`` is the convolutional trunk, followed by a transformer, a strided
    downsample to the 12.5 Hz frame rate, and the split residual quantiser.
    """

    def __init__(
        self,
        config: TokenizerEncoderConfig,
        valid_num_quantizers: int = 16,
    ) -> None:
        super().__init__()
        self.config = config
        self.valid_num_quantizers = valid_num_quantizers
        self.encoder = SeanetEncoder(config)
        self.encoder_transformer = EncoderTransformer(config)
        encoder_frame_rate = config.sampling_rate / math.prod(config.upsampling_ratios)
        stride = int(encoder_frame_rate / config.frame_rate)
        # ``edge`` padding, unlike the SEANet trunk's zeros.
        self.downsample = StreamableConv1d(
            config.hidden_size,
            config.hidden_size,
            2 * stride,
            stride=stride,
            bias=False,
            pad_mode="edge",
        )
        self.quantizer = EncoderSplitResidualVectorQuantizer(config)

    def encode(self, audio: mx.array) -> mx.array:
        """``[batch, 1, samples]`` -> ``[batch, valid_num_quantizers, frames]``."""
        x = mx.swapaxes(audio, -1, -2)
        x = self.encoder(x)
        x = self.encoder_transformer(x)
        x = self.downsample(x)
        codes = self.quantizer.encode(mx.swapaxes(x, -1, -2))
        return codes[:, : self.valid_num_quantizers, :]


class Qwen3TTSSpeechTokenizer(nn.Module):
    """Speech tokenizer wrapper. The encoder is optional: only Base voice
    cloning needs waveform -> codes."""

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = Qwen3TTSSpeechTokenizerDecoder(config.decoder_config)
        self.encoder = (
            Qwen3TTSSpeechTokenizerEncoder(
                config.encoder_config,
                valid_num_quantizers=config.encoder_valid_num_quantizers,
            )
            if config.encoder_config is not None
            else None
        )
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

    @property
    def sample_rate(self) -> int:
        return self.config.output_sample_rate

    @property
    def has_encoder(self) -> bool:
        return self.encoder is not None

    def encode(self, audio: mx.array) -> mx.array:
        """``[batch, 1, samples]`` waveform -> ``[batch, groups, frames]`` codes."""
        if self.encoder is None:
            raise ValueError(
                "This speech tokenizer has no encoder; only Base checkpoints "
                "ship one, and only voice cloning needs it"
            )
        return self.encoder.encode(audio)

    def decode(self, audio_codes: mx.array) -> Tuple[mx.array, mx.array]:
        """``[batch, time, groups]`` codes -> ``([batch, samples], [batch])``.

        Lengths come from the count of non-padding group-0 codes, matching the
        reference's trimming rule.
        """
        codes = mx.transpose(audio_codes, (0, 2, 1))
        waveform = self.decoder(codes).squeeze(1)
        lengths = (audio_codes[..., 0] > 0).sum(axis=-1) * self.decode_upsample_rate
        return waveform, lengths

    def streaming_decode(self, audio_codes: mx.array, chunk_tokens: int = 100):
        """Yield waveform chunks for ``[batch, time, groups]`` codes."""
        codes = mx.transpose(audio_codes, (0, 2, 1))
        total = codes.shape[-1]
        self.decoder.reset_streaming_state()
        start = 0
        while start < total:
            end = min(start + chunk_tokens, total)
            yield self.decoder.streaming_step(codes[..., start:end])
            start = end

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Fold codebook statistics into lookup tables; keep names otherwise.

        Both halves store each codebook as a running sum plus a usage count,
        and the usable table is their ratio with the denominator floored. The
        decoder spells this ``._codebook.embedding_sum`` and the encoder
        ``.codebook.embed_sum``, so both spellings are folded to
        ``.codebook.embed.weight``. ``initialized`` is training state and is
        dropped.

        Every other key passes through unchanged, because this port mirrors the
        checkpoint's own module layout. Convolution layout is fixed separately
        by :func:`~sglang_omni.models.qwen3_tts.mlx.weights.align_conv_weights`.

        When this tokenizer was built without an encoder, the checkpoint's
        encoder subtree is dropped so a strict load still succeeds.
        """
        sanitized: Dict[str, mx.array] = {}
        codebooks: Dict[str, Dict[str, mx.array]] = {}
        skip_encoder = self.encoder is None

        for key, value in weights.items():
            if skip_encoder and key.startswith("encoder."):
                continue
            base = None
            for marker in ("._codebook.", ".codebook."):
                if marker in key:
                    base, _, stat = key.rpartition(marker)
                    if stat == "cluster_usage":
                        codebooks.setdefault(base, {})["usage"] = value
                    elif stat in ("embedding_sum", "embed_sum"):
                        codebooks.setdefault(base, {})["sum"] = value
                    break
            if base is None:
                sanitized[key] = value

        for base, entry in codebooks.items():
            if "sum" not in entry or "usage" not in entry:
                continue
            usage = mx.clip(entry["usage"][:, None], 1e-5, None)
            sanitized[f"{base}.codebook.embed.weight"] = entry["sum"] / usage

        return sanitized
