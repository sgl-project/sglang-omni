# SPDX-License-Identifier: Apache-2.0
"""Stateful incremental decoder for the Qwen3-TTS speech tokenizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.attention.rope import (
    apply_rope_inplace,
    apply_rope_inplace_with_kvcache,
)
from sglang.kernels.ops.layernorm import rmsnorm
from sglang.srt.utils.custom_op import register_custom_op

from sglang_omni.platforms import current_platform
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
    channels_last_weight,
    is_channels_last_conv_device,
)
from sglang_omni.utils.merged_projections import merge_projection_weights

if current_platform.is_cuda():
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
    from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported

    from sglang_omni.kernels.bias_snake_beta import (
        SnakeBetaCoefficients,
        fused_bias_snake_beta,
    )
    from sglang_omni.kernels.codec_state import history_exchange, overlap_exchange
    from sglang_omni.kernels.single_output_conv import clamped_single_output_conv
else:
    pass

if TYPE_CHECKING:
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2CausalConvNet,
        Qwen3TTSTokenizerV2CausalTransConvNet,
        Qwen3TTSTokenizerV2ConvNeXtBlock,
        Qwen3TTSTokenizerV2Decoder,
        Qwen3TTSTokenizerV2DecoderAttention,
        Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
        Qwen3TTSTokenizerV2DecoderRMSNorm,
        Qwen3TTSTokenizerV2DecoderTransformerLayer,
        Qwen3TTSTokenizerV2DecoderTransformerModel,
    )
else:
    pass


@dataclass(frozen=True)
class Qwen3TTSIncrementalCodecStateSpec:
    """Static description of one stream's incremental state.

    ``conv_histories`` and ``transconv_overlaps`` are ``(key, channels,
    length)`` triples in the order ``Qwen3TTSIncrementalDecoder.decode``
    visits them. Zero-length entries are kept so every consumer sees the same
    key set regardless of kernel size.
    """

    conv_histories: tuple[tuple[str, int, int], ...]
    transconv_overlaps: tuple[tuple[str, int, int], ...]
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    retained_context: int

    def bytes_per_stream(self, dtype: torch.dtype) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        elements = sum(channels * length for _, channels, length in self.conv_histories)
        elements += sum(
            channels * length for _, channels, length in self.transconv_overlaps
        )
        elements += (
            2
            * self.num_layers
            * self.num_key_value_heads
            * self.retained_context
            * self.head_dim
        )
        return int(elements * itemsize)


@dataclass(frozen=True)
class Qwen3TTSKVRings:
    """Every layer's keys and values, read and written in place: cohort row i keeps frame p of
    its stream at ring row slot_rows[i] + p % ring_frames, and frame p of a decode it does not
    keep goes to trash_slot_row + p % ring_frames, a slot no stream reads."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    ring_frames: int
    slot_rows: torch.Tensor
    trash_slot_row: int


@dataclass
class Qwen3TTSIncrementalCodecState:
    frame_position: int = 0
    transformer_context_length: int = 0
    transformer_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    transformer_values: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_overlaps: dict[str, torch.Tensor] = field(default_factory=dict)
    # Note (Qihao Liu): absolute frame position per batch row. ``None`` means
    # every row shares ``frame_position``, which is the single-request case.
    # A cohort assembled from a state arena sets this so streams at different
    # playback positions can execute in one launch.
    frame_positions: torch.Tensor | None = None
    # note (ratish): rows' slots in an arena's conv pools; None when rows own their state.
    slot_index: torch.Tensor | None = None
    # note (ratish): the slot graph padding rows share, which the conv exchanges skip; -1 is none.
    pad_slot: int = -1
    # note (ratish): an arena's K/V rings; None when the K/V live in transformer_keys and values.
    kv_rings: Qwen3TTSKVRings | None = None

    def conv_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.slot_index is None:
            return torch.arange(batch_size, device=device)
        else:
            return self.slot_index

    def row_frame_positions(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.frame_positions is None:
            return torch.full(
                (batch_size,), self.frame_position, device=device, dtype=torch.long
            )
        else:
            pass
        if int(self.frame_positions.shape[0]) != batch_size:
            raise ValueError(
                "Qwen3-TTS incremental codec state carries "
                f"{int(self.frame_positions.shape[0])} row positions for a batch "
                f"of {batch_size}"
            )
        else:
            pass
        return self.frame_positions.to(device=device, dtype=torch.long)

    def advance(self, fresh_frames: int) -> None:
        self.frame_position += fresh_frames
        if self.frame_positions is not None:
            self.frame_positions = self.frame_positions + fresh_frames
        else:
            pass

    def clone(self) -> Qwen3TTSIncrementalCodecState:
        return Qwen3TTSIncrementalCodecState(
            frame_position=self.frame_position,
            transformer_context_length=self.transformer_context_length,
            frame_positions=(
                None if self.frame_positions is None else self.frame_positions.clone()
            ),
            slot_index=None if self.slot_index is None else self.slot_index.clone(),
            pad_slot=self.pad_slot,
            transformer_keys={
                key: value.clone() for key, value in self.transformer_keys.items()
            },
            transformer_values={
                key: value.clone() for key, value in self.transformer_values.items()
            },
            conv_histories={
                key: value.clone() for key, value in self.conv_histories.items()
            },
            transconv_overlaps={
                key: value.clone() for key, value in self.transconv_overlaps.items()
            },
        )


def conv_history(
    state: Qwen3TTSIncrementalCodecState,
    key: str,
    hidden_states: torch.Tensor,
    batch_size: int,
    channels: int,
    history_size: int,
) -> torch.Tensor:
    """The conv's (B, C, history) state, zeros before the stream's first decode."""
    history = state.conv_histories.get(key)
    if history is None:
        return hidden_states.new_zeros(batch_size, channels, history_size)
    elif history.shape[:-1] != (batch_size, channels):
        raise ValueError(f"incremental causal Conv1d state shape changed for {key}")
    else:
        return history


def incremental_causal_conv1d(
    module: "Qwen3TTSTokenizerV2CausalConvNet",
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    key: str,
) -> torch.Tensor:
    conv = module.conv
    stride = int(conv.stride[0])
    if stride != 1:
        raise ValueError(f"incremental causal Conv1d requires stride=1, got {stride}")
    else:
        pass
    history_size = int(module.padding)
    history = conv_history(
        state,
        key,
        hidden_states,
        int(hidden_states.shape[0]),
        int(hidden_states.shape[1]),
        history_size,
    )

    combined = torch.cat((history, hidden_states), dim=-1)
    output = conv(combined).contiguous()
    if output.shape[-1] != hidden_states.shape[-1]:
        raise RuntimeError(
            f"incremental causal Conv1d changed temporal length for {key}"
        )
    else:
        pass
    state.conv_histories[key] = (
        combined[..., -history_size:].clone()
        if history_size
        else combined[..., :0].clone()
    )
    return output


def incremental_causal_transconv1d(
    module: "Qwen3TTSTokenizerV2CausalTransConvNet",
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    key: str,
) -> torch.Tensor:
    conv = module.conv
    stride = int(conv.stride[0])
    right_pad = int(module.right_pad)
    output = F.conv_transpose1d(
        hidden_states,
        conv.weight,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        output_padding=conv.output_padding,
        groups=conv.groups,
        dilation=conv.dilation,
    )
    overlap = state.transconv_overlaps.get(key)
    if overlap is not None:
        if overlap.shape[:-1] != output.shape[:-1]:
            raise ValueError(
                f"incremental causal ConvTranspose1d state shape changed for {key}"
            )
        else:
            pass
        overlap_length = int(overlap.shape[-1])
        output[..., :overlap_length] += overlap
    else:
        pass

    emit_length = int(hidden_states.shape[-1]) * stride
    emitted = output[..., :emit_length]
    tail = output[..., emit_length:]
    if int(tail.shape[-1]) != right_pad:
        raise RuntimeError(
            f"incremental causal ConvTranspose1d produced the wrong overlap for {key}"
        )
    else:
        pass
    state.transconv_overlaps[key] = tail.clone()
    if conv.bias is not None:
        emitted = emitted + conv.bias.view(1, -1, 1)
    else:
        pass
    return emitted.contiguous()


def add_into(out: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> None:
    # note (ratish): Dynamo rejects a non-contiguous out= tensor; inductor fuses this copy.
    if torch.compiler.is_dynamo_compiling():
        out.copy_(first + second)
    else:
        torch.add(first, second, out=out)


def padded_input(
    module: "Qwen3TTSTokenizerV2CausalConvNet", frames_like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, length, channels = frames_like.shape
    history_samples = int(module.padding)
    # note (ratish): the tail completes the last dilation phase of the conv's folded input.
    tail_samples = (-length) % int(module.conv.dilation[0])
    padded = frames_like.new_empty(
        batch_size, history_samples + length + tail_samples, channels
    )
    return padded, padded[:, history_samples : history_samples + length]


def slot_causal_conv(
    module: "Qwen3TTSTokenizerV2CausalConvNet",
    weight: torch.Tensor,
    padded: torch.Tensor,
    history_pool: torch.Tensor,
    slot_index: torch.Tensor,
    pad_slot: int,
    length: int,
) -> torch.Tensor:
    if history_pool.shape[1] > 0:
        history_exchange(
            padded,
            history_pool,
            slot_index,
            new_samples=length,
            is_cold=False,
            pad_slot=pad_slot,
        )
    else:
        pass
    # note (ratish): the consumer adds the bias, fused into its activation.
    return channels_last_conv1d(padded, module.conv, weight, None, length)


def slot_causal_transconv(
    module: "Qwen3TTSTokenizerV2CausalTransConvNet",
    weight: torch.Tensor,
    frames: torch.Tensor,
    overlap_pool: torch.Tensor,
    slot_index: torch.Tensor,
    pad_slot: int,
) -> torch.Tensor:
    conv = module.conv
    # note (ratish): the consumer adds the bias after the overlap, so overlaps carry it once.
    output = channels_last_conv_transpose1d(frames, conv, weight, None)
    emitted_samples = int(frames.shape[1]) * int(conv.stride[0])
    if overlap_pool.shape[1] > 0:
        overlap_exchange(
            output,
            overlap_pool,
            slot_index,
            emitted_samples=emitted_samples,
            is_cold=False,
            pad_slot=pad_slot,
        )
    else:
        pass
    return output[:, :emitted_samples]


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    else:
        pass
    batch, heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, heads, groups, length, head_dim
    )
    return hidden_states.reshape(batch, heads * groups, length, head_dim)


def attention_positions(
    state: Qwen3TTSIncrementalCodecState,
    batch_size: int,
    fresh_frames: int,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """The retained K/V length, then each row's absolute key positions, retained keys
    first, and query positions."""
    frame_positions = state.row_frame_positions(batch_size, device)
    # Note (Qihao Liu): the retained buffer is right-aligned: slot j of a buffer
    # of length P holds absolute position frame_position - P + j. Deriving P
    # from the buffer rather than from transformer_context_length keeps this
    # correct when an arena hands over a full-width buffer that a cold stream
    # has not filled.
    prior_key = state.transformer_keys.get(0)
    prior_length = 0 if prior_key is None else int(prior_key.shape[-2])
    key_offsets = torch.arange(
        -prior_length, fresh_frames, device=device, dtype=torch.long
    )
    query_offsets = torch.arange(fresh_frames, device=device, dtype=torch.long)
    key_positions = frame_positions.unsqueeze(1) + key_offsets.unsqueeze(0)
    query_positions = frame_positions.unsqueeze(1) + query_offsets.unsqueeze(0)
    return prior_length, key_positions, query_positions


def allowed_keys(
    key_positions: torch.Tensor, query_positions: torch.Tensor, sliding_window: int
) -> torch.Tensor:
    """(batch, queries, keys): whether each query attends each key."""
    keys_by_row = key_positions.unsqueeze(1)
    queries_by_row = query_positions.unsqueeze(2)
    allowed = keys_by_row <= queries_by_row
    if sliding_window > 0:
        allowed &= keys_by_row > (queries_by_row - sliding_window)
    else:
        pass
    # Note (Qihao Liu): a right-aligned K/V buffer that a stream has not filled
    # yet holds zeros whose nominal absolute position is negative. Those slots
    # would otherwise satisfy both tests above for an early query, so mask them
    # explicitly. This is what lets a cold and a warm stream share one cohort.
    allowed &= (key_positions >= 0).unsqueeze(1)
    return allowed


def incremental_attention(
    attention: "Qwen3TTSTokenizerV2DecoderAttention",
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    state: Qwen3TTSIncrementalCodecState,
    layer_index: int,
    key_positions: torch.Tensor,
    query_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    head_dim = int(attention.head_dim)
    query = attention.q_norm(attention.q_proj(hidden_states)).view(
        *input_shape, -1, head_dim
    )
    key = attention.k_norm(attention.k_proj(hidden_states)).view(
        *input_shape, -1, head_dim
    )
    value = attention.v_proj(hidden_states).view(*input_shape, -1, head_dim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query, key = apply_rotary_pos_emb(
        query, key, position_embeddings[0], position_embeddings[1]
    )

    prior_key = state.transformer_keys.get(layer_index)
    prior_value = state.transformer_values.get(layer_index)
    if (prior_key is None) != (prior_value is None):
        raise RuntimeError(f"incomplete transformer state for layer {layer_index}")
    else:
        pass
    if prior_key is not None:
        key = torch.cat((prior_key, key), dim=-2)
        value = torch.cat((prior_value, value), dim=-2)
    else:
        pass

    if int(key.shape[-2]) != int(key_positions.shape[-1]):
        raise RuntimeError(
            f"incremental transformer state length {int(key.shape[-2])} does not "
            f"match {int(key_positions.shape[-1])} key positions"
        )
    else:
        pass
    repeated_key = repeat_kv(key, int(attention.num_key_value_groups))
    repeated_value = repeat_kv(value, int(attention.num_key_value_groups))
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * float(
        attention.scaling
    )
    allowed = allowed_keys(
        key_positions, query_positions, int(attention.sliding_window)
    )
    scores = scores.masked_fill(
        ~allowed.unsqueeze(1),
        torch.finfo(scores.dtype).min,
    )
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probabilities, repeated_value)
    output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return attention.o_proj(output), key, value


def frames_ending_at(
    sequence: torch.Tensor, end_frames: torch.Tensor, count: int, dim: int
) -> torch.Tensor:
    """Each row's count entries of sequence along dim that end at end_frames[row]."""
    index = (end_frames - count).unsqueeze(1) + torch.arange(
        count, device=sequence.device
    )
    shape = [1] * sequence.ndim
    shape[0], shape[dim] = int(sequence.shape[0]), count
    expanded = list(sequence.shape)
    expanded[dim] = count
    return sequence.gather(dim, index.view(shape).expand(expanded))


def retained_frames(
    sequence: torch.Tensor, retained_end: torch.Tensor | None, retained_context: int
) -> torch.Tensor:
    """A layer's keys or values kept for the next decode: the last retained_context
    frames, ending at each row's retained_end when given."""
    if retained_end is not None:
        return frames_ending_at(sequence, retained_end, retained_context, -2)
    elif retained_context:
        return sequence[..., -retained_context:, :].clone()
    else:
        return sequence[..., :0, :].clone()


def incremental_transformer(
    transformer: "Qwen3TTSTokenizerV2DecoderTransformerModel",
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    valid_frames: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden_states = transformer.input_proj(hidden_states)
    batch_size = int(hidden_states.shape[0])
    fresh_frames = int(hidden_states.shape[1])
    prior_length, key_positions, query_positions = attention_positions(
        state, batch_size, fresh_frames, hidden_states.device
    )
    position_embeddings = transformer.rotary_emb(hidden_states, query_positions)

    next_keys: dict[int, torch.Tensor] = {}
    next_values: dict[int, torch.Tensor] = {}
    window_size = int(transformer.window_size)
    retained_context = max(0, window_size - 1)
    retained_end = None if valid_frames is None else prior_length + valid_frames
    for layer_index, layer in enumerate(transformer.layers):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attended, key, value = incremental_attention(
            layer.self_attn,
            normalized,
            position_embeddings,
            state,
            layer_index,
            key_positions,
            query_positions,
        )
        hidden_states = residual + layer.self_attn_layer_scale(attended)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + layer.mlp_layer_scale(hidden_states)
        next_keys[layer_index] = retained_frames(key, retained_end, retained_context)
        next_values[layer_index] = retained_frames(
            value, retained_end, retained_context
        )

    state.transformer_keys = next_keys
    state.transformer_values = next_values
    return transformer.output_proj(transformer.norm(hidden_states))


def ring_window_attention(
    query: torch.Tensor,
    key_ring: torch.Tensor,
    value_ring: torch.Tensor,
    page_table: torch.Tensor,
    window_lengths: torch.Tensor,
    softmax_scale: float,
    retained_frames: int,
) -> torch.Tensor:
    """FA3 of (B, L, heads, head_dim) queries over each row's window of ring rows, its L new
    frames last, each query seeing itself and the retained_frames before it."""
    return flash_attn_with_kvcache(
        query,
        key_ring,
        value_ring,
        page_table=page_table,
        cache_seqlens=window_lengths,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(retained_frames, 0),
    )


def ring_window_attention_fake(
    query: torch.Tensor,
    key_ring: torch.Tensor,
    value_ring: torch.Tensor,
    page_table: torch.Tensor,
    window_lengths: torch.Tensor,
    softmax_scale: float,
    retained_frames: int,
) -> torch.Tensor:
    return query.new_empty(query.shape)


if current_platform.is_cuda():
    # note (ratish): FA3 has no fake implementation, so a compiled decode calls it as this op.
    compiled_ring_window_attention = register_custom_op(
        ring_window_attention,
        op_name="qwen3_tts_codec_ring_window_attention",
        mutates_args=[],
        fake_impl=ring_window_attention_fake,
    )
else:
    pass


class FusedIncrementalTransformer:
    """incremental_transformer with one qkv and one gate-up GEMM per layer, and RMSNorm, rotary,
    SwiGLU and LayerScale plus residual one kernel each; on an arena cohort the K/V stay in the
    arena's rings (ring_step)."""

    def __init__(
        self, transformer: "Qwen3TTSTokenizerV2DecoderTransformerModel"
    ) -> None:
        rotary = transformer.rotary_emb
        assert (
            not transformer.config.attention_bias
        ), "the decoder attention has no bias"
        assert transformer.config.hidden_act == "silu", "the decoder MLP is SwiGLU"
        assert rotary.rope_type == "default", "cos and sin come from inv_freq"
        assert transformer.window_size > 1, "the decoder retains a sliding window"
        self.transformer = transformer
        self.inverse_frequencies = rotary.inv_freq.float()
        self.attention_scaling = float(rotary.attention_scaling)
        assert all(
            isinstance(layer.self_attn.q_norm, torch.nn.Identity)
            and isinstance(layer.self_attn.k_norm, torch.nn.Identity)
            for layer in transformer.layers
        ), "the decoder attention has no q or k norm"
        merge_projection_weights(transformer.layers)
        # note (ratish): FA3 runs on SM 8.x and 9.x with CUDA 12.3 or newer; other GPUs gather
        # each row's ring window for SDPA.
        self.is_paged_attention = _is_fa3_supported()

    @staticmethod
    def normalize(
        hidden_states: torch.Tensor, norm: "Qwen3TTSTokenizerV2DecoderRMSNorm"
    ) -> torch.Tensor:
        # note (ratish): Dynamo cannot trace the fused kernel's dispatch,
        # so the precompiled shapes trace the same norm in torch ops.
        if torch.compiler.is_dynamo_compiling():
            normalized = hidden_states.float()
            normalized = normalized * torch.rsqrt(
                normalized.pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon
            )
            return (normalized * norm.weight).to(hidden_states.dtype)
        else:
            return rmsnorm(hidden_states, norm.weight, norm.variance_epsilon)

    def cos_sin_rows(
        self, query_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The rope kernels' cos/sin table, one row per query, and each query's row."""
        # note (ratish): one cos/sin row per query, from its absolute frame position,
        # which a stream's reference prefix can carry past max_position_embeddings.
        rows = query_positions.numel()
        angles = query_positions.reshape(rows, 1).float() * self.inverse_frequencies
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1) * (
            self.attention_scaling
        )
        return cos_sin, torch.arange(rows, device=query_positions.device)

    def projected_qkv(
        self,
        layer: "Qwen3TTSTokenizerV2DecoderTransformerLayer",
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The layer's (rows, heads, head_dim) query and key before rotary and its (rows,
        kv_heads * head_dim) value, from one qkv GEMM on the normed rows."""
        attention = layer.self_attn
        head_dim = int(attention.head_dim)
        rows = int(hidden_states.shape[0])
        normalized = self.normalize(hidden_states, layer.input_layernorm)
        query, key, value = F.linear(normalized, attention.qkv_weight).split(
            [
                attention.q_proj.out_features,
                attention.k_proj.out_features,
                attention.v_proj.out_features,
            ],
            dim=-1,
        )
        return query.view(rows, -1, head_dim), key.view(rows, -1, head_dim), value

    @staticmethod
    def attend(
        layer: "Qwen3TTSTokenizerV2DecoderTransformerLayer",
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA of (B, heads, L, head_dim) queries over (B, kv_heads, S, head_dim) keys and
        values under a (B, 1, L, S) mask, as (B * L, heads * head_dim)."""
        attention = layer.self_attn
        attended = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=attention_mask,
            scale=float(attention.scaling),
            enable_gqa=int(attention.num_key_value_groups) > 1,
        )
        return attended.transpose(1, 2).reshape(
            -1, int(attended.shape[1]) * int(attended.shape[3])
        )

    def layer_output(
        self,
        layer: "Qwen3TTSTokenizerV2DecoderTransformerLayer",
        residual: torch.Tensor,
        attended: torch.Tensor,
    ) -> torch.Tensor:
        """The layer's attention projection, then its SwiGLU MLP, each added to the residual
        by its layer scale in one kernel."""
        hidden_states = torch.addcmul(
            residual,
            layer.self_attn_layer_scale.scale,
            F.linear(attended, layer.self_attn.o_proj.weight),
        )
        normalized = self.normalize(hidden_states, layer.post_attention_layernorm)
        gate_up = F.linear(normalized, layer.mlp.gate_up_weight)
        # note (ratish): torch ops under Dynamo, as in normalize.
        if torch.compiler.is_dynamo_compiling():
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
        else:
            activated = silu_and_mul(gate_up)
        return torch.addcmul(
            hidden_states,
            layer.mlp_layer_scale.scale,
            F.linear(activated, layer.mlp.down_proj.weight),
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.kv_rings is not None:
            return self.ring_step(hidden_states, state, valid_frames)
        else:
            pass
        transformer = self.transformer
        hidden_states = transformer.input_proj(hidden_states)
        batch_size = int(hidden_states.shape[0])
        fresh_frames = int(hidden_states.shape[1])
        hidden_size = int(hidden_states.shape[2])
        rows = batch_size * fresh_frames
        device = hidden_states.device
        prior_length, key_positions, query_positions = attention_positions(
            state, batch_size, fresh_frames, device
        )
        attention_mask = allowed_keys(
            key_positions, query_positions, int(transformer.window_size)
        ).unsqueeze(1)
        cos_sin, rotary_rows = self.cos_sin_rows(query_positions)
        hidden_states = hidden_states.reshape(rows, hidden_size)
        next_keys: dict[int, torch.Tensor] = {}
        next_values: dict[int, torch.Tensor] = {}
        retained_context = int(transformer.window_size) - 1
        retained_end = None if valid_frames is None else prior_length + valid_frames
        for layer_index, layer in enumerate(transformer.layers):
            head_dim = int(layer.self_attn.head_dim)
            query, key, value = self.projected_qkv(layer, hidden_states)
            apply_rope_inplace(query, key, cos_sin, rotary_rows, is_neox=True)
            key = key.view(batch_size, fresh_frames, -1, head_dim).transpose(1, 2)
            value = value.view(batch_size, fresh_frames, -1, head_dim).transpose(1, 2)
            prior_key = state.transformer_keys.get(layer_index)
            prior_value = state.transformer_values.get(layer_index)
            if (prior_key is None) != (prior_value is None):
                raise RuntimeError(
                    f"incomplete transformer state for layer {layer_index}"
                )
            elif prior_key is not None:
                key = torch.cat((prior_key, key), dim=-2)
                value = torch.cat((prior_value, value), dim=-2)
            else:
                pass
            attended = self.attend(
                layer,
                query.view(batch_size, fresh_frames, -1, head_dim).transpose(1, 2),
                key,
                value,
                attention_mask,
            )
            hidden_states = self.layer_output(layer, hidden_states, attended)
            next_keys[layer_index] = retained_frames(
                key, retained_end, retained_context
            )
            next_values[layer_index] = retained_frames(
                value, retained_end, retained_context
            )

        state.transformer_keys = next_keys
        state.transformer_values = next_values
        hidden_states = self.normalize(hidden_states, transformer.norm)
        return transformer.output_proj(
            hidden_states.view(batch_size, fresh_frames, hidden_size)
        )

    def ring_step(
        self,
        hidden_states: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        valid_frames: torch.Tensor | None,
    ) -> torch.Tensor:
        """The step for a cohort whose K/V live in an arena's rings: in place when a decode's
        window and new frames fit the ring, else on its gathered prior window."""
        transformer = self.transformer
        rings = state.kv_rings
        hidden_states = transformer.input_proj(hidden_states)
        batch_size = int(hidden_states.shape[0])
        fresh_frames = int(hidden_states.shape[1])
        hidden_size = int(hidden_states.shape[2])
        rows = batch_size * fresh_frames
        device = hidden_states.device
        window_size = int(transformer.window_size)
        retained_context = window_size - 1
        start_positions = state.row_frame_positions(batch_size, device)
        query_positions = start_positions.unsqueeze(1) + torch.arange(
            fresh_frames, device=device
        )
        slot_rows = rings.slot_rows.unsqueeze(1)
        frame_rows = slot_rows + query_positions % rings.ring_frames
        cos_sin, rotary_rows = self.cos_sin_rows(query_positions)
        is_in_place = (
            valid_frames is None
            and fresh_frames <= rings.ring_frames - retained_context
        )
        if is_in_place:
            write_rows = frame_rows.reshape(rows)
            window_starts = (start_positions - retained_context).clamp(min=0)
            window_rows = (
                slot_rows
                + (
                    window_starts.unsqueeze(1)
                    + torch.arange(rings.ring_frames, device=device)
                )
                % rings.ring_frames
            )
            window_lengths = start_positions - window_starts + fresh_frames
            if self.is_paged_attention:
                page_table = window_rows.to(torch.int32)
                window_lengths = window_lengths.to(torch.int32)
            else:
                query_offsets = (window_lengths - fresh_frames).unsqueeze(
                    1
                ) + torch.arange(fresh_frames, device=device)
                distances = query_offsets.unsqueeze(2) - torch.arange(
                    rings.ring_frames, device=device
                )
                attention_mask = (
                    (distances >= 0) & (distances < window_size)
                ).unsqueeze(1)
        else:
            key_positions = start_positions.unsqueeze(1) + torch.arange(
                -retained_context, fresh_frames, device=device
            )
            attention_mask = allowed_keys(
                key_positions, query_positions, window_size
            ).unsqueeze(1)
            prior_rows = (
                slot_rows + key_positions[:, :retained_context] % rings.ring_frames
            )
            valid_ends = start_positions + (
                fresh_frames if valid_frames is None else valid_frames
            )
            # note (ratish): a decode wider than the ring past its window stores only the frames
            # the next decode reads, so no two of its frames share a ring row; the rest spread
            # over the trash slot's rows, since a long first chunk discards hundreds a row.
            is_kept = (
                query_positions >= (valid_ends - retained_context).unsqueeze(1)
            ) & (query_positions < valid_ends.unsqueeze(1))
            write_rows = torch.where(
                is_kept,
                frame_rows,
                rings.trash_slot_row + query_positions % rings.ring_frames,
            ).reshape(rows)
        hidden_states = hidden_states.reshape(rows, hidden_size)
        for layer_index, layer in enumerate(transformer.layers):
            head_dim = int(layer.self_attn.head_dim)
            key_ring = rings.keys[layer_index]
            value_ring = rings.values[layer_index]
            query, key, value = self.projected_qkv(layer, hidden_states)
            if not is_in_place:
                # note (ratish): read before the store, which can reuse a prior frame's ring row.
                prior_keys = key_ring[prior_rows]
                prior_values = value_ring[prior_rows]
            else:
                pass
            apply_rope_inplace_with_kvcache(
                query,
                key,
                value.view(rows, -1, head_dim),
                key_ring,
                value_ring,
                cos_sin,
                rotary_rows,
                write_rows,
                is_neox=True,
            )
            key_value_heads = int(key.shape[1])
            query = query.view(batch_size, fresh_frames, -1, head_dim)
            if is_in_place and self.is_paged_attention:
                window_attention = (
                    compiled_ring_window_attention
                    if torch.compiler.is_dynamo_compiling()
                    else ring_window_attention
                )
                attended = window_attention(
                    query,
                    key_ring.view(-1, 1, key_value_heads, head_dim),
                    value_ring.view(-1, 1, key_value_heads, head_dim),
                    page_table,
                    window_lengths,
                    float(layer.self_attn.scaling),
                    retained_context,
                ).reshape(rows, -1)
            elif is_in_place:
                attended = self.attend(
                    layer,
                    query.transpose(1, 2),
                    key_ring[window_rows]
                    .view(batch_size, -1, key_value_heads, head_dim)
                    .transpose(1, 2),
                    value_ring[window_rows]
                    .view(batch_size, -1, key_value_heads, head_dim)
                    .transpose(1, 2),
                    attention_mask,
                )
            else:
                attended = self.attend(
                    layer,
                    query.transpose(1, 2),
                    torch.cat(
                        (
                            prior_keys.view(
                                batch_size, retained_context, key_value_heads, head_dim
                            ).transpose(1, 2),
                            key.view(
                                batch_size, fresh_frames, key_value_heads, head_dim
                            ).transpose(1, 2),
                        ),
                        dim=-2,
                    ),
                    torch.cat(
                        (
                            prior_values.view(
                                batch_size, retained_context, key_value_heads, head_dim
                            ).transpose(1, 2),
                            value.view(
                                batch_size, fresh_frames, key_value_heads, head_dim
                            ).transpose(1, 2),
                        ),
                        dim=-2,
                    ),
                    attention_mask,
                )
            hidden_states = self.layer_output(layer, hidden_states, attended)
        hidden_states = self.normalize(hidden_states, transformer.norm)
        return transformer.output_proj(
            hidden_states.view(batch_size, fresh_frames, hidden_size)
        )


def incremental_convnext(
    module: "Qwen3TTSTokenizerV2ConvNeXtBlock",
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    key: str,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = incremental_causal_conv1d(
        module.dwconv, hidden_states, state, f"{key}.dwconv"
    ).permute(0, 2, 1)
    hidden_states = module.norm(hidden_states)
    hidden_states = module.pwconv1(hidden_states)
    hidden_states = module.act(hidden_states)
    hidden_states = module.pwconv2(hidden_states)
    hidden_states = module.gamma * hidden_states
    return residual + hidden_states.permute(0, 2, 1)


def incremental_residual_unit(
    module: "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit",
    hidden_states: torch.Tensor,
    state: Qwen3TTSIncrementalCodecState,
    key: str,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = incremental_causal_conv1d(
        module.conv1, module.act1(hidden_states), state, f"{key}.conv1"
    )
    hidden_states = incremental_causal_conv1d(
        module.conv2, module.act2(hidden_states), state, f"{key}.conv2"
    )
    return hidden_states + residual


class Qwen3TTSIncrementalDecoder:
    def __init__(self, decoder: "Qwen3TTSTokenizerV2Decoder") -> None:
        self.require_attrs(
            decoder,
            "decoder",
            "quantizer",
            "pre_conv",
            "pre_transformer",
            "upsample",
            "decoder",
            "total_upsample",
        )
        if len(decoder.decoder) < 3:
            raise TypeError("unsupported Qwen3-TTS decoder layout")
        else:
            pass
        self.require_attrs(decoder.pre_conv, "pre_conv", "conv", "padding")
        self.require_attrs(
            decoder.pre_transformer,
            "pre_transformer",
            "input_proj",
            "layers",
            "norm",
            "output_proj",
            "rotary_emb",
            "window_size",
        )
        for layer_index, layer in enumerate(decoder.pre_transformer.layers):
            self.require_attrs(
                layer,
                f"pre_transformer.layers.{layer_index}",
                "input_layernorm",
                "self_attn",
                "self_attn_layer_scale",
                "post_attention_layernorm",
                "mlp",
                "mlp_layer_scale",
            )
            self.require_attrs(
                layer.self_attn,
                f"pre_transformer.layers.{layer_index}.self_attn",
                "head_dim",
                "num_key_value_groups",
                "scaling",
                "sliding_window",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_norm",
                "k_norm",
            )
        for stage_index, blocks in enumerate(decoder.upsample):
            if len(blocks) != 2:
                raise TypeError("unsupported Qwen3-TTS upsample layout")
            else:
                pass
            self.require_attrs(
                blocks[0], f"upsample.{stage_index}.0", "conv", "right_pad"
            )
            self.require_attrs(
                blocks[1],
                f"upsample.{stage_index}.1",
                "dwconv",
                "norm",
                "pwconv1",
                "act",
                "pwconv2",
                "gamma",
            )
        self.require_attrs(decoder.decoder[0], "decoder.0", "conv", "padding")
        self.require_attrs(decoder.decoder[-1], "decoder.final", "conv", "padding")
        for block_index, block in enumerate(decoder.decoder[1:-2], start=1):
            if not hasattr(block, "block") or len(block.block) < 2:
                raise TypeError("unsupported Qwen3-TTS decoder block layout")
            else:
                pass
            self.require_attrs(
                block.block[1],
                f"decoder.{block_index}.1",
                "conv",
                "right_pad",
            )
            for residual_index, residual in enumerate(block.block[2:]):
                self.require_attrs(
                    residual,
                    f"decoder.{block_index}.{residual_index + 2}",
                    "act1",
                    "conv1",
                    "act2",
                    "conv2",
                )
        self.decoder = decoder
        self.total_upsample = int(decoder.total_upsample)
        # note (ratish): on CUDA the convs run channels last on weights kept here in
        # that layout; the shared modules keep theirs for the whole-utterance decode.
        if is_channels_last_conv_device(decoder.pre_conv.conv.weight.device):
            self.channels_last_weights: dict[str, torch.Tensor] | None = {
                "pre_conv": channels_last_weight(decoder.pre_conv.conv),
                "decoder.0": channels_last_weight(decoder.decoder[0].conv),
            }
            for stage_index, blocks in enumerate(decoder.upsample):
                self.channels_last_weights[f"upsample.{stage_index}.transconv"] = (
                    channels_last_weight(blocks[0].conv)
                )
            activations = [decoder.decoder[-2]]
            for block_index, block in enumerate(decoder.decoder[1:-2], start=1):
                self.channels_last_weights[f"decoder.{block_index}.transconv"] = (
                    channels_last_weight(block.block[1].conv)
                )
                activations.append(block.block[0])
                for residual_index, residual in enumerate(block.block[2:]):
                    key = f"decoder.{block_index}.residual.{residual_index}"
                    self.channels_last_weights[f"{key}.conv1"] = channels_last_weight(
                        residual.conv1.conv
                    )
                    self.channels_last_weights[f"{key}.conv2"] = channels_last_weight(
                        residual.conv2.conv
                    )
                    activations.extend((residual.act1, residual.act2))
            self.snake_coefficients: dict[torch.nn.Module, SnakeBetaCoefficients] = {
                activation: SnakeBetaCoefficients.from_activation(activation)
                for activation in activations
            }
        else:
            self.channels_last_weights = None
            self.snake_coefficients = {}
        transformer_weight = decoder.pre_transformer.input_proj.weight
        if (
            current_platform.is_cuda()
            and transformer_weight.device.type == "cuda"
            and transformer_weight.dtype in (torch.bfloat16, torch.float16)
        ):
            self.fused_transformer: FusedIncrementalTransformer | None = (
                FusedIncrementalTransformer(decoder.pre_transformer)
            )
        else:
            self.fused_transformer = None
        self._state_spec: Qwen3TTSIncrementalCodecStateSpec | None = (
            None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self.compiled_kernel: (
            Callable[[torch.Tensor, Qwen3TTSIncrementalCodecState], torch.Tensor] | None
        ) = None
        self.compiled_shapes: set[tuple[int, int]] = set()
        self.first_chunk_reach: dict[int, int] = {}

    @staticmethod
    def require_attrs(module: object, path: str, *names: str) -> None:
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            raise TypeError(
                "unsupported Qwen3-TTS decoder layout; "
                f"{path} is missing {', '.join(missing)}"
            )
        else:
            pass

    def state_spec(self) -> Qwen3TTSIncrementalCodecStateSpec:
        """Describe one stream's state without running a decode.

        Note (Qihao Liu): the lazy buffers in ``decode`` only materialize once
        activations have flowed through, so a state arena cannot preallocate
        from them. This walks the validated module tree in the same order
        ``decode`` does and derives every key and shape statically.
        """
        if (
            self._state_spec is None
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self._state_spec = (
                self.build_state_spec()
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        else:
            pass
        return (
            self._state_spec
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    def build_state_spec(self) -> Qwen3TTSIncrementalCodecStateSpec:
        decoder = self.decoder
        conv: list[tuple[str, int, int]] = []
        transconv: list[tuple[str, int, int]] = []

        def add_conv(module: "Qwen3TTSTokenizerV2CausalConvNet", key: str) -> None:
            conv.append((key, int(module.conv.in_channels), int(module.padding)))

        def add_transconv(
            module: "Qwen3TTSTokenizerV2CausalTransConvNet", key: str
        ) -> None:
            transconv.append(
                (key, int(module.conv.out_channels), int(module.right_pad))
            )

        add_conv(decoder.pre_conv, "pre_conv")
        for stage_index, blocks in enumerate(decoder.upsample):
            add_transconv(blocks[0], f"upsample.{stage_index}.transconv")
            add_conv(blocks[1].dwconv, f"upsample.{stage_index}.convnext.dwconv")
        add_conv(decoder.decoder[0], "decoder.0")
        for block_index, decoder_block in enumerate(decoder.decoder[1:-2], start=1):
            add_transconv(decoder_block.block[1], f"decoder.{block_index}.transconv")
            for residual_index, residual in enumerate(decoder_block.block[2:]):
                key = f"decoder.{block_index}.residual.{residual_index}"
                add_conv(residual.conv1, f"{key}.conv1")
                add_conv(residual.conv2, f"{key}.conv2")
        add_conv(decoder.decoder[-1], "decoder.final")

        transformer = decoder.pre_transformer
        attention = transformer.layers[0].self_attn
        head_dim = int(attention.head_dim)
        key_features = int(attention.k_proj.out_features)
        if key_features % head_dim:
            raise TypeError(
                "unsupported Qwen3-TTS decoder layout; k_proj width "
                f"{key_features} is not a multiple of head_dim {head_dim}"
            )
        else:
            pass
        return Qwen3TTSIncrementalCodecStateSpec(
            conv_histories=tuple(conv),
            transconv_overlaps=tuple(transconv),
            num_layers=len(transformer.layers),
            num_key_value_heads=key_features // head_dim,
            head_dim=head_dim,
            retained_context=max(0, int(transformer.window_size) - 1),
        )

    def conv_state_shape(
        self, rows: int, channels: int, samples: int
    ) -> tuple[int, int, int]:
        if self.channels_last_weights is not None:
            # note (ratish): the slot stack's exchange kernels read whole channel spans.
            return (rows, samples, channels)
        else:
            return (rows, channels, samples)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        has_key_value_buffers: bool = True,
    ) -> Qwen3TTSIncrementalCodecState:
        """Allocate a zeroed state with full-width buffers, the transformer's K/V among them
        unless has_key_value_buffers is False.

        Note (Qihao Liu): unlike the lazily grown state, the Transformer K/V
        buffers start at their retained width. A stream that has not filled
        them yet reads zeros at negative nominal positions, which
        ``_incremental_attention`` masks out, so a cold and a warm stream share
        one execution shape.
        """
        spec = self.state_spec()
        state = Qwen3TTSIncrementalCodecState(
            frame_positions=torch.zeros(batch_size, device=device, dtype=torch.long),
            transformer_context_length=spec.retained_context,
        )
        for key, channels, length in spec.conv_histories:
            state.conv_histories[key] = torch.zeros(
                self.conv_state_shape(batch_size, channels, length),
                device=device,
                dtype=dtype,
            )
        for key, channels, length in spec.transconv_overlaps:
            state.transconv_overlaps[key] = torch.zeros(
                self.conv_state_shape(batch_size, channels, length),
                device=device,
                dtype=dtype,
            )
        for layer_index in range(spec.num_layers if has_key_value_buffers else 0):
            shape = (
                batch_size,
                spec.num_key_value_heads,
                spec.retained_context,
                spec.head_dim,
            )
            state.transformer_keys[layer_index] = torch.zeros(
                shape, device=device, dtype=dtype
            )
            state.transformer_values[layer_index] = torch.zeros(
                shape, device=device, dtype=dtype
            )
        return state

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        *,
        compiled: bool = False,
    ) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires codes shaped [B, Q, T]"
            )
        else:
            pass
        if int(codes.shape[0]) < 1:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires at least one row"
            )
        else:
            pass
        fresh_frames = int(codes.shape[-1])
        if fresh_frames <= 0:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires fresh frames"
            )
        else:
            pass
        # note (luojiaxuan): the compiled kernel is opt-in per call. Only the
        # graph runner asks for it, at capture time, on static buffers; every
        # eager decode stays off Dynamo, whose value guards would otherwise
        # recompile once per stream position and hit the recompile limit.
        shape = (int(codes.shape[0]), fresh_frames)
        if compiled and shape not in self.compiled_shapes:
            raise RuntimeError(
                f"Qwen3-TTS incremental codec shape {shape} was not precompiled"
            )
        else:
            pass
        kernel = self.compiled_kernel if compiled else self.decode_tensors
        waveform = kernel(codes, state)
        expected_samples = fresh_frames * self.total_upsample
        if int(waveform.shape[-1]) != expected_samples:
            raise RuntimeError(
                "Qwen3-TTS incremental codec decoder returned the wrong sample count"
            )
        else:
            pass
        state.transformer_context_length = min(
            self.state_spec().retained_context,
            state.transformer_context_length + fresh_frames,
        )
        state.advance(fresh_frames)
        return waveform

    def tail_frames(self, emit_frames: int) -> int:
        """How many transformer outputs the last emit_frames frames' samples and the conv-stack
        state depend on, walked back from the waveform: a causal conv adds its history, and a
        transposed conv of kernel k and stride s maps n samples to (n + k - 1) // s inputs.
        """
        decoder = self.decoder
        samples = int(emit_frames) * self.total_upsample + int(
            decoder.decoder[-1].padding
        )
        for decoder_block in reversed(decoder.decoder[1:-2]):
            for residual_unit in reversed(decoder_block.block[2:]):
                samples += int(residual_unit.conv2.padding) + int(
                    residual_unit.conv1.padding
                )
            transconv = decoder_block.block[1].conv
            samples = (samples + int(transconv.kernel_size[0]) - 1) // int(
                transconv.stride[0]
            )
        samples += int(decoder.decoder[0].padding)
        for blocks in reversed(decoder.upsample):
            samples += int(blocks[1].dwconv.padding)
            transconv = blocks[0].conv
            samples = (samples + int(transconv.kernel_size[0]) - 1) // int(
                transconv.stride[0]
            )
        return samples

    def first_chunk_reference_frames(self, emit_frames: int) -> int:
        """How many frames before its last emit_frames a stream's first decode needs: its
        emitted samples and kept state read the transformer outputs of the last
        tail_frames(emit_frames) frames, each sliding-window layer reaches window - 1
        frames further back, and pre_conv its padding before that."""
        # note (ratish): the vocoder plans every stream's first chunk with this, and the
        # module walk costs about 50 us a call, so each emit count is walked once.
        if emit_frames not in self.first_chunk_reach:
            transformer = self.decoder.pre_transformer
            self.first_chunk_reach[emit_frames] = (
                self.tail_frames(emit_frames)
                - int(emit_frames)
                + len(transformer.layers) * (int(transformer.window_size) - 1)
                + int(self.decoder.pre_conv.padding)
            )
        else:
            pass
        return self.first_chunk_reach[emit_frames]

    @torch.no_grad()
    def decode_tail(
        self,
        codes: torch.Tensor,
        valid_frames: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        emit_frames: int,
    ) -> torch.Tensor:
        """Decode row i's valid_frames[i] frames with the conv stack on the last
        tail_frames(emit_frames) outputs only, and return the last emit_frames frames' samples.
        Needs arena state (per-row positions, full-width K/V), tail_frames <= valid_frames[i].
        """
        if state.frame_positions is None:
            raise ValueError("Qwen3-TTS tail decoding needs per-row frame positions")
        else:
            pass
        tail_frames = self.tail_frames(emit_frames)
        hidden_states = self.frame_tensors(codes, state, valid_frames)
        time_dim = 1 if self.channels_last_weights is not None else 2
        window = frames_ending_at(hidden_states, valid_frames, tail_frames, time_dim)
        waveform = self.conv_stack_tensors(window, state)
        state.transformer_context_length = self.state_spec().retained_context
        state.frame_positions = state.frame_positions + valid_frames
        return waveform[..., -int(emit_frames) * self.total_upsample :]

    def decode_tensors(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
    ) -> torch.Tensor:
        """The tensor-only decode step: every op here is traceable and capturable.

        The frame counter and retained-context bookkeeping live in ``decode``:
        they are the Python scalars that change every step, and inside the
        traced region each distinct value would force a recompile. With
        channels_last_weights, activations flow as (B, L, C) from the quantizer
        to the final conv.
        """
        if self.channels_last_weights is not None and not state.conv_histories:
            # note (ratish): the slot stack exchanges in place, so an empty state gets zeroed pools.
            spec = self.state_spec()
            weight = self.decoder.pre_conv.conv.weight
            for pools, entries in (
                (state.conv_histories, spec.conv_histories),
                (state.transconv_overlaps, spec.transconv_overlaps),
            ):
                for key, channels, samples in entries:
                    pools[key] = weight.new_zeros(
                        self.conv_state_shape(int(codes.shape[0]), channels, samples)
                    )
        else:
            pass
        return self.conv_stack_tensors(self.frame_tensors(codes, state), state)

    def frame_tensors(
        self,
        codes: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Quantizer, pre_conv and transformer in the decoder's layout; with valid_frames,
        row i's pre_conv history and kept K/V end at valid_frames[i]."""
        quantized = self.decoder.quantizer.decode(codes)
        if self.channels_last_weights is not None:
            transformer_inputs = self.slot_pre_conv(quantized, state, valid_frames)
        else:
            prior_history = (
                state.conv_histories["pre_conv"] if valid_frames is not None else None
            )
            hidden_states = incremental_causal_conv1d(
                self.decoder.pre_conv, quantized, state, "pre_conv"
            )
            if valid_frames is not None:
                history_size = int(prior_history.shape[-1])
                state.conv_histories["pre_conv"] = frames_ending_at(
                    torch.cat((prior_history, quantized), dim=-1),
                    history_size + valid_frames,
                    history_size,
                    -1,
                )
            else:
                pass
            transformer_inputs = hidden_states.transpose(1, 2)
        if self.fused_transformer is not None:
            transformer_outputs = self.fused_transformer(
                transformer_inputs, state, valid_frames
            )
        else:
            transformer_outputs = incremental_transformer(
                self.decoder.pre_transformer, transformer_inputs, state, valid_frames
            )
        if self.channels_last_weights is not None:
            return transformer_outputs
        else:
            return transformer_outputs.permute(0, 2, 1)

    def slot_pre_conv(
        self,
        quantized: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        valid_frames: torch.Tensor | None,
    ) -> torch.Tensor:
        module = self.decoder.pre_conv
        history_pool = state.conv_histories["pre_conv"]
        batch_size, _, length = quantized.shape
        slot_index = state.conv_slots(batch_size, quantized.device)
        frames = quantized.transpose(1, 2)
        padded, padded_frames = padded_input(module, frames)
        padded_frames.copy_(frames)
        if valid_frames is None:
            history_exchange(
                padded,
                history_pool,
                slot_index,
                new_samples=length,
                is_cold=False,
                pad_slot=state.pad_slot,
            )
        else:
            history_samples = int(module.padding)
            padded[:, :history_samples].copy_(history_pool.index_select(0, slot_index))
            history_pool.index_copy_(
                0,
                slot_index,
                frames_ending_at(
                    padded, history_samples + valid_frames, history_samples, 1
                ),
            )
        return channels_last_conv1d(
            padded,
            module.conv,
            self.channels_last_weights["pre_conv"],
            module.conv.bias,
            length,
        )

    def conv_stack_tensors(
        self, hidden_states: torch.Tensor, state: Qwen3TTSIncrementalCodecState
    ) -> torch.Tensor:
        """Everything after the transformer, from its outputs to the clamped waveform."""
        if self.channels_last_weights is not None:
            return self.slot_conv_stack(hidden_states, state)
        else:
            pass
        for stage_index, blocks in enumerate(self.decoder.upsample):
            hidden_states = incremental_causal_transconv1d(
                blocks[0], hidden_states, state, f"upsample.{stage_index}.transconv"
            )
            hidden_states = incremental_convnext(
                blocks[1], hidden_states, state, f"upsample.{stage_index}.convnext"
            )

        waveform = incremental_causal_conv1d(
            self.decoder.decoder[0], hidden_states, state, "decoder.0"
        )
        for block_index, decoder_block in enumerate(
            self.decoder.decoder[1:-2], start=1
        ):
            waveform = incremental_causal_transconv1d(
                decoder_block.block[1],
                decoder_block.block[0](waveform),
                state,
                f"decoder.{block_index}.transconv",
            )
            for residual_index, residual_unit in enumerate(decoder_block.block[2:]):
                waveform = incremental_residual_unit(
                    residual_unit,
                    waveform,
                    state,
                    f"decoder.{block_index}.residual.{residual_index}",
                )
        waveform = incremental_causal_conv1d(
            self.decoder.decoder[-1],
            self.decoder.decoder[-2](waveform),
            state,
            "decoder.final",
        )
        return waveform.clamp(min=-1, max=1)

    def slot_conv_stack(
        self, hidden_states: torch.Tensor, state: Qwen3TTSIncrementalCodecState
    ) -> torch.Tensor:
        decoder = self.decoder
        coefficients = self.snake_coefficients
        slot_index = state.conv_slots(int(hidden_states.shape[0]), hidden_states.device)
        padded, length = self.slot_upsample(hidden_states, state, slot_index)
        decoder_output = slot_causal_conv(
            decoder.decoder[0],
            self.channels_last_weights["decoder.0"],
            padded,
            state.conv_histories["decoder.0"],
            slot_index,
            state.pad_slot,
            length,
        )
        decoder_blocks = decoder.decoder[1:-2]
        activated = decoder_output.new_empty(decoder_output.shape)
        # note (ratish): eager rounding would land further from fp32 than main's compiled adds.
        fused_bias_snake_beta(
            decoder_output,
            decoder.decoder[0].conv.bias,
            coefficients[decoder_blocks[0].block[0]],
            activated,
            should_round_like_eager=False,
        )
        for block_index, decoder_block in enumerate(decoder_blocks, start=1):
            conv_output, residual = self.slot_decoder_block(
                decoder_block, f"decoder.{block_index}", activated, state, slot_index
            )
            bias = decoder_block.block[-1].conv2.conv.bias
            if block_index < len(decoder_blocks):
                activated = conv_output.new_empty(conv_output.shape)
                fused_bias_snake_beta(
                    conv_output,
                    bias,
                    coefficients[decoder_blocks[block_index].block[0]],
                    activated,
                    residual=residual,
                    should_round_like_eager=False,
                )
            else:
                padded, padded_frames = padded_input(decoder.decoder[-1], conv_output)
                fused_bias_snake_beta(
                    conv_output,
                    bias,
                    coefficients[decoder.decoder[-2]],
                    padded_frames,
                    residual=residual,
                    should_round_like_eager=False,
                )
        length = int(conv_output.shape[1])
        history_exchange(
            padded,
            state.conv_histories["decoder.final"],
            slot_index,
            new_samples=length,
            is_cold=False,
            pad_slot=state.pad_slot,
        )
        # note (ratish): cuDNN has no tensor-core engine for one output channel.
        return clamped_single_output_conv(padded, decoder.decoder[-1].conv, length)

    def slot_upsample(
        self,
        hidden_states: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        slot_index: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        decoder = self.decoder
        weights = self.channels_last_weights
        last_stage_index = len(decoder.upsample) - 1
        for stage_index, blocks in enumerate(decoder.upsample):
            transconv, convnext = blocks[0], blocks[1]
            key = f"upsample.{stage_index}"
            frames = slot_causal_transconv(
                transconv,
                weights[f"{key}.transconv"],
                hidden_states,
                state.transconv_overlaps[f"{key}.transconv"],
                slot_index,
                state.pad_slot,
            )
            # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel,
            # faster than cuDNN's channels-last depthwise and free of its per-shape setup.
            batch_size, length, channels = frames.shape
            history_samples = int(convnext.dwconv.padding)
            depthwise_input = frames.new_empty(
                batch_size, channels, history_samples + length
            )
            residual = depthwise_input[:, :, history_samples:].transpose(1, 2)
            add_into(residual, frames, transconv.conv.bias)
            history_exchange(
                depthwise_input.transpose(1, 2),
                state.conv_histories[f"{key}.convnext.dwconv"],
                slot_index,
                new_samples=length,
                is_cold=False,
                pad_slot=state.pad_slot,
            )
            normed = convnext.norm(
                convnext.dwconv.conv(depthwise_input).transpose(1, 2)
            )
            scaled = convnext.gamma * convnext.pwconv2(
                convnext.act(convnext.pwconv1(normed))
            )
            if stage_index < last_stage_index:
                hidden_states = residual + scaled
            else:
                padded, padded_frames = padded_input(decoder.decoder[0], scaled)
                add_into(padded_frames, residual, scaled)
        return padded, length

    def slot_decoder_block(
        self,
        decoder_block: torch.nn.Module,
        key: str,
        activated: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        slot_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.channels_last_weights
        coefficients = self.snake_coefficients
        histories = state.conv_histories
        transconv = decoder_block.block[1]
        frames = slot_causal_transconv(
            transconv,
            weights[f"{key}.transconv"],
            activated,
            state.transconv_overlaps[f"{key}.transconv"],
            slot_index,
            state.pad_slot,
        )
        residual_units = decoder_block.block[2:]
        length = int(frames.shape[1])
        residual = frames.new_empty(frames.shape)
        padded, padded_frames = padded_input(residual_units[0].conv1, frames)
        fused_bias_snake_beta(
            frames,
            transconv.conv.bias,
            coefficients[residual_units[0].act1],
            padded_frames,
            sum_out=residual,
            should_round_like_eager=False,
        )
        for residual_index, unit in enumerate(residual_units):
            unit_key = f"{key}.residual.{residual_index}"
            conv1_output = slot_causal_conv(
                unit.conv1,
                weights[f"{unit_key}.conv1"],
                padded,
                histories[f"{unit_key}.conv1"],
                slot_index,
                state.pad_slot,
                length,
            )
            padded, padded_frames = padded_input(unit.conv2, frames)
            fused_bias_snake_beta(
                conv1_output,
                unit.conv1.conv.bias,
                coefficients[unit.act2],
                padded_frames,
                should_round_like_eager=False,
            )
            conv2_output = slot_causal_conv(
                unit.conv2,
                weights[f"{unit_key}.conv2"],
                padded,
                histories[f"{unit_key}.conv2"],
                slot_index,
                state.pad_slot,
                length,
            )
            if residual_index + 1 < len(residual_units):
                next_unit = residual_units[residual_index + 1]
                padded, padded_frames = padded_input(next_unit.conv1, frames)
                fused_bias_snake_beta(
                    conv2_output,
                    unit.conv2.conv.bias,
                    coefficients[next_unit.act1],
                    padded_frames,
                    residual=residual,
                    sum_out=residual,
                    should_round_like_eager=False,
                )
            else:
                pass
        return conv2_output, residual

    def precompile(
        self, codes: torch.Tensor, state: Qwen3TTSIncrementalCodecState
    ) -> None:
        """Trace the tensor-only step on these inputs and admit their shape.

        ``decode`` routes a shape through the compiled kernel only after it was
        precompiled here, so an unforeseen shape at serving time takes the eager
        path instead of a multi-second compile on a request's critical path.

        Pass the codes and state of a real decode, in its grad-disabling
        context: Dynamo guards on both, and the step advances the state.
        """
        if self.compiled_kernel is None:
            self.compiled_kernel = torch.compile(
                self.decode_tensors, dynamic=False, fullgraph=True
            )
        else:
            pass
        shape = (int(codes.shape[0]), int(codes.shape[-1]))
        if shape in self.compiled_shapes:
            return
        else:
            pass
        # note (luojiaxuan): one trace per (batch, width) pair, past the default
        # 8; ``decode`` never compiles a new shape, so the limit only matters here.
        with torch._dynamo.config.patch(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            recompile_limit=max(
                torch._dynamo.config.recompile_limit, 64
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        ):
            self.compiled_kernel(codes, state)
        self.compiled_shapes.add(shape)
