# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
    _incremental_transformer,
    incremental_causal_conv1d,
    incremental_causal_transconv1d,
)


def _random_partitions(total: int, seed: int) -> list[int]:
    generator = random.Random(seed)
    partitions = []
    while total:
        length = generator.randint(1, min(4, total))
        partitions.append(length)
        total -= length
    return partitions


class _CausalConv(nn.Module):
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


class _CausalTransConv(nn.Module):
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


class _RotaryEmbedding(nn.Module):
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


class _Attention(nn.Module):
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


class _TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, window_size: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = _Attention(hidden_size, head_dim, window_size)
        self.self_attn_layer_scale = nn.Identity()
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.mlp_layer_scale = nn.Identity()


class _Transformer(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden_size: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_size)
        self.layers = nn.ModuleList([_TransformerLayer(hidden_size, 2, 4)])
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, latent_dim)
        self.rotary_emb = _RotaryEmbedding(2)
        self.window_size = 4


class _Quantizer(nn.Module):
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes[:, :2].to(torch.float32) / 16.0


class _ConvNeXt(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = _CausalConv(channels, channels, 7, groups=channels)
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


class _ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.act1 = nn.Tanh()
        self.conv1 = _CausalConv(channels, channels, 7, dilation=dilation)
        self.act2 = nn.Tanh()
        self.conv2 = _CausalConv(channels, channels, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(self.act1(hidden_states))
        hidden_states = self.conv2(self.act2(hidden_states))
        return hidden_states + residual


class _DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                nn.Tanh(),
                _CausalTransConv(4, 2, 4, 2),
                _ResidualUnit(2, 1),
                _ResidualUnit(2, 3),
                _ResidualUnit(2, 9),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.block:
            hidden_states = module(hidden_states)
        return hidden_states


class _Decoder(nn.Module):
    total_upsample = 4

    def __init__(self) -> None:
        super().__init__()
        self.quantizer = _Quantizer()
        self.pre_conv = _CausalConv(2, 2, 3)
        self.pre_transformer = _Transformer()
        self.upsample = nn.ModuleList(
            [nn.ModuleList([_CausalTransConv(2, 2, 2, 2), _ConvNeXt(2)])]
        )
        self.decoder = nn.ModuleList(
            [
                _CausalConv(2, 4, 7),
                _DecoderBlock(),
                nn.Tanh(),
                _CausalConv(2, 1, 7),
            ]
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        hidden_states = self.quantizer.decode(codes)
        hidden_states = self.pre_conv(hidden_states).transpose(1, 2)
        hidden_states = _full_transformer(self.pre_transformer, hidden_states).permute(
            0, 2, 1
        )
        for modules in self.upsample:
            for module in modules:
                hidden_states = module(hidden_states)
        waveform = hidden_states
        for module in self.decoder:
            waveform = module(waveform)
        return waveform.clamp(min=-1, max=1)


def _full_transformer(
    transformer: _Transformer, hidden_states: torch.Tensor
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


@pytest.mark.parametrize("partitions", [[9], [1] * 9, [1, 8], [8, 1], [3, 2, 4]])
def test_incremental_causal_conv_matches_whole(
    partitions: list[int],
) -> None:
    torch.manual_seed(1)
    module = _CausalConv(2, 3, 7, dilation=3)
    inputs = torch.randn(1, 2, sum(partitions))
    expected = module(inputs)
    state = Qwen3TTSIncrementalCodecState()
    actual = []
    offset = 0
    for length in partitions:
        actual.append(
            incremental_causal_conv1d(
                module, inputs[..., offset : offset + length], state, "conv"
            )
        )
        offset += length

    torch.testing.assert_close(torch.cat(actual, dim=-1), expected)
    assert state.conv_histories["conv"].shape[-1] == module.padding


@pytest.mark.parametrize("partitions", [[9], [1] * 9, [1, 8], [8, 1], [3, 2, 4]])
def test_incremental_causal_transconv_matches_whole(
    partitions: list[int],
) -> None:
    torch.manual_seed(2)
    module = _CausalTransConv(2, 3, 8, 4)
    inputs = torch.randn(1, 2, sum(partitions))
    expected = module(inputs)
    state = Qwen3TTSIncrementalCodecState()
    actual = []
    offset = 0
    for length in partitions:
        actual.append(
            incremental_causal_transconv1d(
                module, inputs[..., offset : offset + length], state, "transconv"
            )
        )
        offset += length

    torch.testing.assert_close(torch.cat(actual, dim=-1), expected)
    assert state.transconv_overlaps["transconv"].shape[-1] == module.right_pad


@pytest.mark.parametrize("partitions", [[11], [1] * 11, [1, 10], [10, 1], [3, 2, 6]])
def test_incremental_transformer_matches_whole_across_window(
    partitions: list[int],
) -> None:
    torch.manual_seed(3)
    transformer = _Transformer()
    inputs = torch.randn(1, sum(partitions), 2)
    expected = _full_transformer(transformer, inputs)
    state = Qwen3TTSIncrementalCodecState()
    actual = []
    offset = 0
    for length in partitions:
        actual.append(
            _incremental_transformer(
                transformer, inputs[:, offset : offset + length], state
            )
        )
        state.frame_position += length
        offset += length

    torch.testing.assert_close(torch.cat(actual, dim=1), expected)
    assert state.transformer_context_length == transformer.window_size - 1
    assert all(
        item.shape[-2] == transformer.window_size - 1
        for item in state.transformer_keys.values()
    )


def test_incremental_codec_state_clone_owns_tensor_storage() -> None:
    state = Qwen3TTSIncrementalCodecState(
        frame_position=3,
        transformer_context_length=2,
        transformer_keys={0: torch.ones(1, 1, 2, 2)},
        transformer_values={0: torch.ones(1, 1, 2, 2)},
        conv_histories={"conv": torch.ones(1, 1, 2)},
        transconv_overlaps={"transconv": torch.ones(1, 1, 2)},
    )

    cloned = state.clone()
    cloned.transformer_keys[0].zero_()
    cloned.transformer_values[0].zero_()
    cloned.conv_histories["conv"].zero_()
    cloned.transconv_overlaps["transconv"].zero_()

    assert state.transformer_keys[0].count_nonzero() == 4
    assert state.transformer_values[0].count_nonzero() == 4
    assert state.conv_histories["conv"].count_nonzero() == 2
    assert state.transconv_overlaps["transconv"].count_nonzero() == 2


@pytest.mark.parametrize(
    "partitions",
    [[11], [1] * 11, [1, 10], [10, 1], [3, 2, 6], _random_partitions(96, 5)],
)
def test_incremental_decoder_matches_whole_for_arbitrary_partitions(
    partitions: list[int],
) -> None:
    torch.manual_seed(4)
    decoder = _Decoder()
    codes = torch.randint(0, 16, (1, 2, sum(partitions)))
    expected = decoder(codes)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    state = Qwen3TTSIncrementalCodecState()
    actual = []
    offset = 0
    for length in partitions:
        actual.append(incremental.decode(codes[..., offset : offset + length], state))
        offset += length

    actual_waveform = torch.cat(actual, dim=-1)
    torch.testing.assert_close(actual_waveform, expected, rtol=2e-5, atol=2e-6)
    assert actual_waveform.shape[-1] == sum(partitions) * decoder.total_upsample
    assert state.frame_position == sum(partitions)
    assert state.transformer_context_length == decoder.pre_transformer.window_size - 1


def test_incremental_decoder_reference_prefix_matches_generated_waveform() -> None:
    torch.manual_seed(5)
    decoder = _Decoder()
    codes = torch.randint(0, 16, (1, 2, 9))
    reference_frames = 4
    expected = decoder(codes)[..., reference_frames * decoder.total_upsample :]
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    state = Qwen3TTSIncrementalCodecState()

    initial = incremental.decode(codes[..., :7], state)
    initial = initial[..., reference_frames * decoder.total_upsample :]
    final = incremental.decode(codes[..., 7:], state)
    actual = torch.cat((initial, final), dim=-1)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual.shape[-1] == (9 - reference_frames) * decoder.total_upsample


def test_incremental_decoder_rejects_batch_greater_than_one() -> None:
    decoder = _Decoder()
    incremental = Qwen3TTSIncrementalDecoder(decoder)

    with pytest.raises(ValueError, match="\[1, Q, T\]"):
        incremental.decode(
            torch.ones(2, 2, 1, dtype=torch.long),
            Qwen3TTSIncrementalCodecState(),
        )
