# SPDX-License-Identifier: Apache-2.0
"""Voxtral audio code decoder implemented with MLX layers."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from sglang_omni_mlx.voxtral_tts.config import AudioTokenizerConfig

SEMANTIC_CODEBOOK_EPSILON = 1e-5
ATTENTION_QUERY_CHUNK_SIZE = 1024


class CausalConv1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        use_bias: bool,
        padding_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.conv = nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=stride, bias=use_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        padding_left = self.kernel_size - self.stride
        target_length = math.ceil(hidden_states.shape[1] / self.stride) * self.stride
        padding_right = target_length - hidden_states.shape[1]
        hidden_states = mx.pad(
            hidden_states,
            ((0, 0), (padding_left, padding_right), (0, 0)),
            mode=self.padding_mode,
        )
        return self.conv(hidden_states)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            channels, channels, kernel_size, stride=stride, bias=use_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv(hidden_states)
        total_padding = self.kernel_size - self.stride
        left_padding = 0
        right_padding = total_padding
        end_index = hidden_states.shape[1] - right_padding
        return hidden_states[:, left_padding:end_index, :]


class AudioAttention(nn.Module):
    def __init__(self, config: AudioTokenizerConfig, window_size: int) -> None:
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.head_count = config.head_count
        self.kv_head_count = config.kv_head_count
        self.head_dimension = config.head_dimension
        self.wq = nn.Linear(
            config.dimension, self.head_count * self.head_dimension, bias=False
        )
        self.wk = nn.Linear(
            config.dimension, self.kv_head_count * self.head_dimension, bias=False
        )
        self.wv = nn.Linear(
            config.dimension, self.kv_head_count * self.head_dimension, bias=False
        )
        self.wo = nn.Linear(
            self.head_count * self.head_dimension,
            config.dimension,
            bias=config.use_biases,
        )
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_count * self.head_dimension, eps=config.qk_norm_epsilon
            )
            self.k_norm = nn.RMSNorm(
                self.kv_head_count * self.head_dimension, eps=config.qk_norm_epsilon
            )
        else:
            pass

    @staticmethod
    def get_alibi_slopes(head_count: int) -> list[float]:
        def power_of_two_slopes(count: int) -> list[float]:
            base = 2.0 ** (-8.0 / count)
            return [base**index for index in range(count)]

        if math.log2(head_count).is_integer():
            return power_of_two_slopes(head_count)
        else:
            pass
        lower_power = 2 ** math.floor(math.log2(head_count))
        return (
            power_of_two_slopes(lower_power)
            + power_of_two_slopes(2 * lower_power)[::2][: head_count - lower_power]
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape
        queries = self.wq(hidden_states)
        keys = self.wk(hidden_states)
        values = self.wv(hidden_states)
        if self.config.qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        else:
            pass
        queries = queries.reshape(
            batch_size, sequence_length, self.head_count, self.head_dimension
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.kv_head_count, self.head_dimension
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.kv_head_count, self.head_dimension
        ).transpose(0, 2, 1, 3)
        repeats = self.head_count // self.kv_head_count
        if repeats > 1:
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)
        else:
            pass
        alibi_slopes = mx.array(self.get_alibi_slopes(self.head_count))
        if self.config.causal:
            right_window_tokens = 0
        else:
            right_window_tokens = self.window_size
        attention_outputs: list[mx.array] = []
        for query_start_position in range(
            0, sequence_length, ATTENTION_QUERY_CHUNK_SIZE
        ):
            query_end_position = min(
                query_start_position + ATTENTION_QUERY_CHUNK_SIZE, sequence_length
            )
            key_start_position = max(0, query_start_position - self.window_size)
            key_end_position = min(
                sequence_length, query_end_position + right_window_tokens
            )
            query_positions = mx.arange(query_start_position, query_end_position)
            key_positions = mx.arange(key_start_position, key_end_position)
            relative_positions = key_positions[None, :] - query_positions[:, None]
            attention_bias = alibi_slopes.astype(hidden_states.dtype)[
                :, None, None
            ] * relative_positions[None, :, :].astype(hidden_states.dtype)
            if self.config.causal:
                attention_bias = mx.where(
                    relative_positions[None, :, :] > 0,
                    -float("inf"),
                    attention_bias,
                )
            else:
                pass
            outside_window = (relative_positions < -self.window_size) | (
                relative_positions > right_window_tokens
            )
            attention_bias = mx.where(
                outside_window[None, :, :], -float("inf"), attention_bias
            )
            attention_outputs.append(
                mx.fast.scaled_dot_product_attention(
                    queries[:, :, query_start_position:query_end_position, :],
                    keys[:, :, key_start_position:key_end_position, :],
                    values[:, :, key_start_position:key_end_position, :],
                    scale=self.head_dimension**-0.5,
                    mask=attention_bias[None, :, :, :],
                )
            )
        attended = mx.concatenate(attention_outputs, axis=2)
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.head_count * self.head_dimension
        )
        return self.wo(attended)


class AudioTransformerBlock(nn.Module):
    def __init__(
        self, config: AudioTokenizerConfig, layer_id: int, window_size: int
    ) -> None:
        super().__init__()
        self.attention = AudioAttention(config, window_size)
        self.feed_forward = AudioFeedForward(config)
        self.attention_norm = nn.RMSNorm(config.dimension, eps=config.norm_epsilon)
        self.ffn_norm = nn.RMSNorm(config.dimension, eps=config.norm_epsilon)
        self.attention_scale: mx.array | None = None
        self.ffn_scale: mx.array | None = None
        if config.layer_scale:
            if config.layer_scale_initial is not None:
                initial_scale = config.layer_scale_initial
            elif layer_id < 18:
                initial_scale = 0.1
            elif layer_id <= 24:
                initial_scale = 1e-5
            else:
                initial_scale = 1e-6
            self.attention_scale = mx.full((config.dimension,), initial_scale)
            self.ffn_scale = mx.full((config.dimension,), initial_scale)
        else:
            pass

    def __call__(self, hidden_states: mx.array) -> mx.array:
        attention_output = self.attention(self.attention_norm(hidden_states))
        if self.attention_scale is not None:
            attention_output = self.attention_scale * attention_output
        else:
            pass
        hidden_states = hidden_states + attention_output
        feed_forward_output = self.feed_forward(self.ffn_norm(hidden_states))
        if self.ffn_scale is not None:
            feed_forward_output = self.ffn_scale * feed_forward_output
        else:
            pass
        return hidden_states + feed_forward_output


class AudioFeedForward(nn.Module):
    def __init__(self, config: AudioTokenizerConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dimension, config.hidden_dimension, bias=False)
        self.w2 = nn.Linear(
            config.hidden_dimension, config.dimension, bias=config.use_biases
        )
        self.w3 = nn.Linear(config.dimension, config.hidden_dimension, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class AudioTokenizerDecoder(nn.Module):
    def __init__(self, config: AudioTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dimension = config.semantic_dimension + config.acoustic_dimension
        self.decoder_blocks: list[nn.Module] = []
        current_window_size = config.sliding_window_size
        for encoder_stride in config.encoder_convolution_strides:
            if config.half_attention_window_on_downsample and encoder_stride > 1:
                current_window_size //= 2
            else:
                pass
        self.decoder_blocks.append(
            CausalConv1d(
                self.latent_dimension,
                config.dimension,
                config.convolution_kernels[0],
                config.convolution_strides[0],
                False,
                padding_mode="edge",
            )
        )
        if (
            config.half_attention_window_on_downsample
            and config.convolution_strides[0] > 1
        ):
            current_window_size *= 2
        else:
            pass
        layer_id = 0
        for block_index, layer_count in enumerate(config.transformer_lengths):
            self.decoder_blocks.append(
                AudioTransformerStack(
                    config, layer_count, current_window_size, layer_id
                )
            )
            layer_id += layer_count
            next_index = block_index + 1
            if next_index < len(config.transformer_lengths) and (
                config.convolution_kernels[next_index] != 1
                or config.convolution_strides[next_index] != 1
            ):
                self.decoder_blocks.append(
                    CausalConvTranspose1d(
                        config.dimension,
                        config.convolution_kernels[next_index],
                        config.convolution_strides[next_index],
                        False,
                    )
                )
                if (
                    config.half_attention_window_on_downsample
                    and config.convolution_strides[next_index] > 1
                ):
                    current_window_size *= 2
                else:
                    pass
            else:
                pass
        self.semantic_embedding = mx.zeros(
            (config.semantic_codebook_size, config.semantic_dimension)
        )
        self.output_proj = CausalConv1d(
            config.dimension,
            config.patch_size,
            config.patch_projection_kernel_size,
            1,
            False,
            padding_mode="reflect",
        )
        self.downsample_factor = config.patch_size * math.prod(
            config.convolution_strides
        )
        self.sampling_rate = config.sampling_rate

    def __call__(self, audio_codes: mx.array) -> mx.array:
        semantic_codes = audio_codes[:, 0, :]
        decoder_dtype = self.output_proj.conv.weight.dtype
        semantic_features = self.semantic_embedding[semantic_codes].transpose(0, 2, 1)
        semantic_features = semantic_features.astype(decoder_dtype)
        acoustic_features = (
            audio_codes[:, 1:, :].astype(mx.float32)
            * (2.0 / (self.config.acoustic_codebook_size - 1))
            - 1.0
        ).astype(decoder_dtype)
        hidden_states = mx.concatenate([semantic_features, acoustic_features], axis=1)
        hidden_states = hidden_states.transpose(0, 2, 1)
        for decoder_block in self.decoder_blocks:
            hidden_states = decoder_block(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        batch_size, frame_count, _ = hidden_states.shape
        return hidden_states.reshape(
            batch_size, 1, frame_count * self.config.patch_size
        )


class AudioTransformerStack(nn.Module):
    def __init__(
        self,
        config: AudioTokenizerConfig,
        layer_count: int,
        window_size: int,
        first_layer_id: int,
    ) -> None:
        super().__init__()
        self.layers = [
            AudioTransformerBlock(config, first_layer_id + layer_index, window_size)
            for layer_index in range(layer_count)
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
