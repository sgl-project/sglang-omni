# SPDX-License-Identifier: Apache-2.0
# The architecture follows the Apache-2.0 Transformers Nemotron implementation
# maintained in ../hf_compat (Copyright 2026 The HuggingFace Inc. team).
"""FastConformer and greedy RNN-T inference using native MLX operations."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def subsampled_lengths(lengths, config):
    for _ in range(int(math.log2(config.subsampling_factor))):
        lengths = lengths // config.subsampling_conv_stride + 1
    return lengths


class CausalConv2d(nn.Conv2d):
    def __call__(self, x):
        kernel = self.weight.shape[1]
        stride = self.stride[0]
        return super().__call__(
            mx.pad(
                x, ((0, 0), (kernel - 1, stride - 1), (kernel - 1, stride - 1), (0, 0))
            )
        )


class SubsamplingLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        channels = c.subsampling_conv_channels
        self.depthwise_conv = CausalConv2d(
            channels,
            channels,
            c.subsampling_conv_kernel_size,
            stride=c.subsampling_conv_stride,
            groups=channels,
        )
        self.pointwise_conv = nn.Conv2d(channels, channels, 1)

    def __call__(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Subsampling(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_in = CausalConv2d(
            1,
            c.subsampling_conv_channels,
            c.subsampling_conv_kernel_size,
            stride=c.subsampling_conv_stride,
        )
        self.layers = [
            SubsamplingLayer(c) for _ in range(int(math.log2(c.subsampling_factor)) - 1)
        ]
        self.linear = nn.Linear(c.subsampling_out_hidden_size, c.hidden_size)
        self._stride = c.subsampling_conv_stride

    def __call__(self, features, lengths):
        x = features[..., None]
        for conv in [self.conv_in, *self.layers]:
            x = conv(x)
            lengths = lengths // self._stride + 1
            valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
            x = nn.relu(x * valid[:, :, None, None])
        # NeMo flattens channels before frequency, whereas MLX convolutions use NHWC.
        x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.linear(x)


class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.linear1 = nn.Linear(
            c.hidden_size, c.intermediate_size, bias=c.attention_bias
        )
        self.linear2 = nn.Linear(
            c.intermediate_size, c.hidden_size, bias=c.attention_bias
        )

    def __call__(self, x):
        return self.linear2(nn.silu(self.linear1(x)))


class Convolution(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.hidden_size
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1, bias=c.convolution_bias)
        self.depthwise_conv = nn.Conv1d(
            d, d, c.conv_kernel_size, groups=d, bias=c.convolution_bias
        )
        self.norm = nn.LayerNorm(d)
        self.pointwise_conv2 = nn.Conv1d(d, d, 1, bias=c.convolution_bias)
        self._left_pad = c.conv_kernel_size - 1

    def __call__(self, x, valid):
        a, b = mx.split(self.pointwise_conv1(x), 2, axis=-1)
        x = a * mx.sigmoid(b) * valid[..., None]
        x = self.depthwise_conv(mx.pad(x, ((0, 0), (self._left_pad, 0), (0, 0))))
        return self.pointwise_conv2(nn.silu(self.norm(x)))


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.hidden_size
        self._heads = c.num_attention_heads
        self._head_dim = d // self._heads
        self.q_proj = nn.Linear(d, d, bias=c.attention_bias)
        self.k_proj = nn.Linear(d, d, bias=c.attention_bias)
        self.v_proj = nn.Linear(d, d, bias=c.attention_bias)
        self.o_proj = nn.Linear(d, d, bias=c.attention_bias)
        self.relative_k_proj = nn.Linear(d, d, bias=False)
        self.bias_u = mx.zeros((self._heads, self._head_dim))
        self.bias_v = mx.zeros((self._heads, self._head_dim))

    def __call__(self, x, positions, mask):
        b, t, d = x.shape

        def split(y):
            return y.reshape(b, -1, self._heads, self._head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        p = (
            self.relative_k_proj(positions)
            .reshape(1, -1, self._heads, self._head_dim)
            .transpose(0, 2, 3, 1)
        )
        bd = (q + self.bias_v[None, :, None, :]) @ p
        # Transformer-XL relative shift, identical to the Torch reference.
        pos_len = bd.shape[-1]
        bd = mx.pad(bd, ((0, 0), (0, 0), (0, 0), (1, 0)))
        bd = bd.reshape(b, self._heads, -1, t)[:, :, 1:, :].reshape(
            b, self._heads, t, pos_len
        )[..., :t]
        scale = self._head_dim**-0.5
        bias = mx.where(mask, bd * scale, -1e9)
        y = mx.fast.scaled_dot_product_attention(
            q + self.bias_u[None, :, None, :],
            k,
            v,
            scale=scale,
            mask=bias,
        )
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.feed_forward1 = FeedForward(c)
        self.feed_forward2 = FeedForward(c)
        self.self_attn = Attention(c)
        self.conv = Convolution(c)
        self.norm_feed_forward1 = nn.LayerNorm(c.hidden_size)
        self.norm_self_att = nn.LayerNorm(c.hidden_size)
        self.norm_conv = nn.LayerNorm(c.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(c.hidden_size)
        self.norm_out = nn.LayerNorm(c.hidden_size)

    def __call__(self, x, positions, mask, valid):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), positions, mask)
        x = x + self.conv(self.norm_conv(x), valid)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self._config = c
        self.subsampling = Subsampling(c)
        self.layers = [Block(c) for _ in range(c.num_hidden_layers)]

    def __call__(self, features, lengths, lookahead):
        c = self._config
        x = self.subsampling(features, lengths)
        if c.scale_input:
            x = x * math.sqrt(c.hidden_size)
        t = x.shape[1]
        if t > c.max_position_embeddings:
            raise ValueError("Audio exceeds the encoder positional capacity")
        output_lengths = subsampled_lengths(lengths, c)
        valid = mx.arange(t)[None, :] < output_lengths[:, None]
        chunk_size = lookahead + 1
        chunks = mx.arange(t) // chunk_size
        diff = chunks[:, None] - chunks[None, :]
        allowed = (diff >= 0) & (diff <= (c.sliding_window - 1) // chunk_size)
        mask = allowed[None, None, :, :] & valid[:, None, None, :]
        freq = mx.exp(
            -math.log(10000)
            * mx.arange(0, c.hidden_size, 2).astype(mx.float32)
            / c.hidden_size
        )
        angles = mx.arange(t - 1, -t, -1).astype(mx.float32)[:, None] * freq[None, :]
        positions = mx.stack([mx.sin(angles), mx.cos(angles)], axis=-1).reshape(
            1, 2 * t - 1, c.hidden_size
        )
        for layer in self.layers:
            x = layer(x, positions, mask, valid)
            mx.eval(x)
        return x, output_lengths


class LSTM(nn.Module):
    """Torch gate order (i, f, g, o), with request-owned hidden/cell state."""

    def __init__(self, size, layers):
        super().__init__()
        self._layers = layers
        self._size = size
        for i in range(layers):
            self[f"weight_ih_l{i}"] = mx.zeros((4 * size, size))
            self[f"weight_hh_l{i}"] = mx.zeros((4 * size, size))
            self[f"bias_ih_l{i}"] = mx.zeros((4 * size,))
            self[f"bias_hh_l{i}"] = mx.zeros((4 * size,))

    def __call__(self, x, state=None):
        if state is None:
            state = [(mx.zeros_like(x), mx.zeros_like(x)) for _ in range(self._layers)]
        updated = []
        for i, (h, c) in enumerate(state):
            gates = x @ self[f"weight_ih_l{i}"].T + h @ self[f"weight_hh_l{i}"].T
            gates = gates + self[f"bias_ih_l{i}"] + self[f"bias_hh_l{i}"]
            gi, gf, gg, go = mx.split(gates, 4, axis=-1)
            c = mx.sigmoid(gf) * c + mx.sigmoid(gi) * mx.tanh(gg)
            x = mx.sigmoid(go) * mx.tanh(c)
            updated.append((x, c))
        return x, updated


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embedding = nn.Embedding(c.vocab_size, c.decoder_hidden_size)
        self.lstm = LSTM(c.decoder_hidden_size, c.num_decoder_layers)
        self.decoder_projector = nn.Linear(c.decoder_hidden_size, c.decoder_hidden_size)

    def __call__(self, token, state=None):
        x, state = self.lstm(self.embedding(mx.array([token])), state)
        return self.decoder_projector(x), state


class PromptProjector(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.linear_1 = nn.Linear(
            c.encoder_config.hidden_size + c.num_prompts, c.prompt_intermediate_size
        )
        self.linear_2 = nn.Linear(
            c.prompt_intermediate_size, c.encoder_config.hidden_size
        )

    def __call__(self, x):
        return self.linear_2(nn.relu(self.linear_1(x)))


class Joint(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.head = nn.Linear(c.decoder_hidden_size, c.vocab_size)

    def __call__(self, encoder, decoder):
        return self.head(nn.relu(encoder + decoder))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.encoder = Encoder(config.encoder_config)
        self.prompt_projector = PromptProjector(config)
        self.encoder_projector = nn.Linear(
            config.encoder_config.hidden_size, config.decoder_hidden_size
        )
        self.decoder = Decoder(config)
        self.joint = Joint(config)

    def encode(self, features, lengths, prompt_ids, lookahead):
        x, output_lengths = self.encoder(features, lengths, lookahead)
        one_hot = (
            prompt_ids[:, None] == mx.arange(self._config.num_prompts)[None, :]
        ).astype(x.dtype)
        one_hot = mx.broadcast_to(
            one_hot[:, None, :], (*x.shape[:2], one_hot.shape[-1])
        )
        x = self.encoder_projector(
            self.prompt_projector(mx.concatenate([x, one_hot], axis=-1))
        )
        mx.eval(x, output_lengths)
        return x, output_lengths

    def decode(self, encoded, length, max_new_tokens=None):
        c = self._config
        # Generation limits count RNN-T steps (including blanks), like Transformers.
        limit = (
            max_new_tokens
            if max_new_tokens is not None
            else encoded.shape[0] * c.max_symbols_per_step - 1
        )
        decoder, state = self.decoder(c.blank_token_id)
        tokens = [c.blank_token_id]
        steps = 0
        for frame in range(length):
            for _ in range(c.max_symbols_per_step):
                if steps >= limit:
                    return tokens
                token = int(
                    mx.argmax(
                        self.joint(encoded[frame : frame + 1], decoder), axis=-1
                    ).item()
                )
                tokens.append(token)
                steps += 1
                if token == c.blank_token_id:
                    break
                decoder, state = self.decoder(token, state)
                mx.eval(decoder, state)
        return tokens


def sanitize_weights(weights):
    """Convert official Torch OIHW/OIK kernels to MLX OHWI/OKI layout."""
    return {
        name: (
            value.transpose(0, 2, 3, 1)
            if value.ndim == 4
            else (
                value.transpose(0, 2, 1)
                if value.ndim == 3 and name.endswith("weight")
                else value
            )
        )
        for name, value in weights.items()
    }
