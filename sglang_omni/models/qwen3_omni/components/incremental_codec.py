# SPDX-License-Identifier: Apache-2.0
"""The code2wav conv stack decoded incrementally from per-stream slots in a codec state pool:
each causal conv reads its history from the rows' slots and stores the newest, and each
transposed conv adds the slots' overlap and stores the samples the next frames still add to."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
)


def conv_stack_state_layout(
    codec: Qwen3OmniMoeCode2Wav,
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """The conv stack's (key, channels, length) conv histories and transposed-conv overlaps,
    in the order slot_conv_stack visits them."""
    conv_histories: list[tuple[str, int, int]] = []
    transconv_overlaps: list[tuple[str, int, int]] = []

    def add_conv(module: Qwen3OmniMoeCausalConvNet, state_key: str) -> None:
        conv_histories.append((state_key, module.conv.in_channels, module.padding))

    def add_transconv(module: Qwen3OmniMoeCausalTransConvNet, state_key: str) -> None:
        transconv_overlaps.append(
            (state_key, module.conv.out_channels, module.right_pad)
        )

    for stage_index, blocks in enumerate(codec.upsample):
        add_transconv(blocks[0], f"upsample.{stage_index}.transconv")
        add_conv(blocks[1].dwconv, f"upsample.{stage_index}.convnext.dwconv")
    add_conv(codec.decoder[0], "decoder.0")
    for block_index, decoder_block in enumerate(codec.decoder[1:-2], start=1):
        add_transconv(decoder_block.block[1], f"decoder.{block_index}.transconv")
        for residual_index, residual in enumerate(decoder_block.block[2:]):
            residual_key = f"decoder.{block_index}.residual.{residual_index}"
            add_conv(residual.conv1, f"{residual_key}.conv1")
            add_conv(residual.conv2, f"{residual_key}.conv2")
    add_conv(codec.decoder[-1], "decoder.final")
    return conv_histories, transconv_overlaps


@dataclass(frozen=True, kw_only=True)
class SlotDecode:
    """The pool slots a decode's rows read and write, and whether every row starts its stream."""

    pool: CodecStatePool
    slot_index: torch.Tensor
    is_cold: bool


def with_history(
    frames: torch.Tensor,
    slot_decode: SlotDecode,
    state_key: str,
    history_samples: int,
) -> torch.Tensor:
    """The (B, history + L, C) frames with each slot's history in front (zeros when the rows
    start their streams); the last history samples become the slots' history."""
    histories = slot_decode.pool.histories
    length = frames.shape[1]
    if state_key not in histories:
        return frames
    elif slot_decode.is_cold:
        padded = F.pad(frames, (0, 0, history_samples, 0))
    else:
        history = histories[state_key].index_select(0, slot_decode.slot_index)
        padded = torch.cat((history, frames), dim=1)
    histories[state_key].index_copy_(0, slot_decode.slot_index, padded[:, length:])
    return padded


def slot_causal_conv(
    module: Qwen3OmniMoeCausalConvNet,
    frames: torch.Tensor,
    slot_decode: SlotDecode,
    state_key: str,
) -> torch.Tensor:
    """The causal conv's (B, L, C_out) outputs for the rows' new (B, L, C) frames."""
    conv = module.conv
    length = frames.shape[1]
    padded = with_history(frames, slot_decode, state_key, module.padding)
    padded = F.pad(padded, (0, 0, 0, (-length) % conv.dilation[0]))
    return channels_last_conv1d(padded, conv, conv.weight, length)


def slot_causal_transconv(
    module: Qwen3OmniMoeCausalTransConvNet,
    frames: torch.Tensor,
    slot_decode: SlotDecode,
    state_key: str,
    *,
    left_trim: int,
) -> torch.Tensor:
    """The transposed conv's emitted samples after left_trim, the slots' overlap added in
    front and the samples the next frames add to kept; the bias follows the overlap, so
    overlapping samples carry it once."""
    conv = module.conv
    overlaps = slot_decode.pool.overlaps
    output = channels_last_conv_transpose1d(frames, conv, conv.weight, None)
    emitted_samples = frames.shape[1] * conv.stride[0]
    if state_key not in overlaps:
        pass
    elif slot_decode.is_cold:
        overlaps[state_key].index_copy_(
            0, slot_decode.slot_index, output[:, emitted_samples:]
        )
    else:
        overlap_samples = output.shape[1] - emitted_samples
        output[:, :overlap_samples] += overlaps[state_key].index_select(
            0, slot_decode.slot_index
        )
        overlaps[state_key].index_copy_(
            0, slot_decode.slot_index, output[:, emitted_samples:]
        )
    return output[:, left_trim:emitted_samples] + conv.bias


def channels_last_activation(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """A (B, C, L) activation module run on (B, L, C) activations."""
    return module(hidden_states.transpose(1, 2)).transpose(1, 2)


def slot_conv_stack(
    codec: Qwen3OmniMoeCode2Wav,
    hidden_states: torch.Tensor,
    slot_decode: SlotDecode,
) -> torch.Tensor:
    """Everything after the transformer, from its (B, L, C) outputs to the clamped (B, 1,
    samples) waveform, on channels-last activations. A stream's first decode drops each
    transposed conv's left pad, as the whole-sequence forward does."""
    for stage_index, blocks in enumerate(codec.upsample):
        transconv, convnext = blocks[0], blocks[1]
        hidden_states = slot_causal_transconv(
            transconv,
            hidden_states,
            slot_decode,
            f"upsample.{stage_index}.transconv",
            left_trim=transconv.left_pad if slot_decode.is_cold else 0,
        )
        padded = with_history(
            hidden_states,
            slot_decode,
            f"upsample.{stage_index}.convnext.dwconv",
            convnext.dwconv.padding,
        )
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel, faster
        # than cuDNN's channels-last depthwise and free of its per-shape setup.
        normed = convnext.norm(
            convnext.dwconv.conv(padded.transpose(1, 2).contiguous()).transpose(1, 2)
        )
        hidden_states = hidden_states + convnext.gamma * convnext.pwconv2(
            convnext.act(convnext.pwconv1(normed))
        )
    hidden_states = slot_causal_conv(
        codec.decoder[0], hidden_states, slot_decode, "decoder.0"
    )
    for block_index, decoder_block in enumerate(codec.decoder[1:-2], start=1):
        transconv = decoder_block.block[1]
        hidden_states = slot_causal_transconv(
            transconv,
            channels_last_activation(decoder_block.block[0], hidden_states),
            slot_decode,
            f"decoder.{block_index}.transconv",
            left_trim=transconv.left_pad if slot_decode.is_cold else 0,
        )
        for residual_index, unit in enumerate(decoder_block.block[2:]):
            residual_key = f"decoder.{block_index}.residual.{residual_index}"
            output = slot_causal_conv(
                unit.conv1,
                channels_last_activation(unit.act1, hidden_states),
                slot_decode,
                f"{residual_key}.conv1",
            )
            output = slot_causal_conv(
                unit.conv2,
                channels_last_activation(unit.act2, output),
                slot_decode,
                f"{residual_key}.conv2",
            )
            hidden_states = output + hidden_states
    waveform = slot_causal_conv(
        codec.decoder[-1],
        channels_last_activation(codec.decoder[-2], hidden_states),
        slot_decode,
        "decoder.final",
    )
    return waveform.transpose(1, 2).clamp(min=-1, max=1)
