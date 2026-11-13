# SPDX-License-Identifier: Apache-2.0
"""The code2wav conv stack decoded incrementally from per-stream slots in a codec state pool:
every producer writes its frames into the next causal conv's padded input, whose history the
rows' slots supply and take back, and each transposed conv adds the slots' overlap and stores
the samples the next frames still add to, both exchanged in place."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.kernels.bias_snake_beta import (
    SnakeBetaCoefficients,
    fused_bias_snake_beta,
)
from sglang_omni.kernels.codec_state import history_exchange, overlap_exchange
from sglang_omni.kernels.single_output_conv import clamped_single_output_conv
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


def padded_input(
    module: Qwen3OmniMoeCausalConvNet, frames_like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A causal conv's empty (B, history + L + tail, C) input for (B, L, C) frames shaped as
    frames_like, and its view of the L frame samples the producer writes."""
    batch_size, length, channels = frames_like.shape
    history_samples = module.padding
    tail_samples = (-length) % module.conv.dilation[0]
    padded = frames_like.new_empty(
        batch_size, history_samples + length + tail_samples, channels
    )
    return padded, padded[:, history_samples : history_samples + length]


def slot_causal_conv(
    module: Qwen3OmniMoeCausalConvNet,
    padded: torch.Tensor,
    slot_decode: SlotDecode,
    state_key: str,
    length: int,
) -> torch.Tensor:
    """The causal conv's (B, length, C_out) outputs without its bias, which the consumer adds,
    for a padded_input whose frames the producer wrote, its history exchanged with the slots in
    place."""
    conv = module.conv
    histories = slot_decode.pool.histories
    if state_key in histories:
        history_exchange(
            padded,
            histories[state_key],
            slot_decode.slot_index,
            new_samples=length,
            is_cold=slot_decode.is_cold,
        )
    else:
        pass
    return channels_last_conv1d(padded, conv, conv.weight, None, length)


def slot_causal_transconv(
    module: Qwen3OmniMoeCausalTransConvNet,
    frames: torch.Tensor,
    slot_decode: SlotDecode,
    state_key: str,
    *,
    left_trim: int,
) -> torch.Tensor:
    """The transposed conv's emitted samples after left_trim without its bias, the slots'
    overlap added in front and the samples the next frames add to kept; the consumer adds the
    bias after the overlap, so overlapping samples carry it once."""
    conv = module.conv
    overlaps = slot_decode.pool.overlaps
    output = channels_last_conv_transpose1d(frames, conv, conv.weight, None)
    emitted_samples = frames.shape[1] * conv.stride[0]
    if state_key in overlaps:
        overlap_exchange(
            output,
            overlaps[state_key],
            slot_decode.slot_index,
            emitted_samples=emitted_samples,
            is_cold=slot_decode.is_cold,
        )
    else:
        pass
    return output[:, left_trim:emitted_samples]


def slot_conv_stack(
    codec: Qwen3OmniMoeCode2Wav,
    hidden_states: torch.Tensor,
    slot_decode: SlotDecode,
    snake_coefficients: dict[torch.nn.Module, SnakeBetaCoefficients],
) -> torch.Tensor:
    """Everything after the transformer, from its (B, L, C) outputs to the clamped (B, 1,
    samples) waveform, on channels-last activations, every SnakeBeta fused with the adds in
    front of it and read from snake_coefficients. A stream's first decode drops each
    transposed conv's left pad, as the whole-sequence forward does."""
    last_stage_index = len(codec.upsample) - 1
    for stage_index, blocks in enumerate(codec.upsample):
        transconv, convnext = blocks[0], blocks[1]
        frames = slot_causal_transconv(
            transconv,
            hidden_states,
            slot_decode,
            f"upsample.{stage_index}.transconv",
            left_trim=transconv.left_pad if slot_decode.is_cold else 0,
        )
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel, faster
        # than cuDNN's channels-last depthwise and free of its per-shape setup.
        batch_size, length, channels = frames.shape
        history_samples = convnext.dwconv.padding
        depthwise_input = frames.new_empty(
            batch_size, channels, history_samples + length
        )
        residual = depthwise_input[:, :, history_samples:].transpose(1, 2)
        torch.add(frames, transconv.conv.bias, out=residual)
        history_exchange(
            depthwise_input.transpose(1, 2),
            slot_decode.pool.histories[f"upsample.{stage_index}.convnext.dwconv"],
            slot_decode.slot_index,
            new_samples=length,
            is_cold=slot_decode.is_cold,
        )
        normed = convnext.norm(convnext.dwconv.conv(depthwise_input).transpose(1, 2))
        scaled = convnext.gamma * convnext.pwconv2(
            convnext.act(convnext.pwconv1(normed))
        )
        if stage_index < last_stage_index:
            hidden_states = residual + scaled
        else:
            padded, padded_frames = padded_input(codec.decoder[0], scaled)
            torch.add(residual, scaled, out=padded_frames)
    decoder_output = slot_causal_conv(
        codec.decoder[0], padded, slot_decode, "decoder.0", length
    )
    decoder_blocks = codec.decoder[1:-2]
    activated = decoder_output.new_empty(decoder_output.shape)
    fused_bias_snake_beta(
        decoder_output,
        codec.decoder[0].conv.bias,
        snake_coefficients[decoder_blocks[0].block[0]],
        activated,
    )
    for block_index, decoder_block in enumerate(decoder_blocks, start=1):
        transconv = decoder_block.block[1]
        frames = slot_causal_transconv(
            transconv,
            activated,
            slot_decode,
            f"decoder.{block_index}.transconv",
            left_trim=transconv.left_pad if slot_decode.is_cold else 0,
        )
        residual_units = decoder_block.block[2:]
        length = frames.shape[1]
        residual = frames.new_empty(frames.shape)
        padded, padded_frames = padded_input(residual_units[0].conv1, frames)
        fused_bias_snake_beta(
            frames,
            transconv.conv.bias,
            snake_coefficients[residual_units[0].act1],
            padded_frames,
            sum_out=residual,
        )
        for residual_index, unit in enumerate(residual_units):
            residual_key = f"decoder.{block_index}.residual.{residual_index}"
            conv1_output = slot_causal_conv(
                unit.conv1,
                padded,
                slot_decode,
                f"{residual_key}.conv1",
                length,
            )
            padded, padded_frames = padded_input(unit.conv2, frames)
            fused_bias_snake_beta(
                conv1_output,
                unit.conv1.conv.bias,
                snake_coefficients[unit.act2],
                padded_frames,
            )
            conv2_output = slot_causal_conv(
                unit.conv2,
                padded,
                slot_decode,
                f"{residual_key}.conv2",
                length,
            )
            if residual_index + 1 < len(residual_units):
                next_unit = residual_units[residual_index + 1]
                padded, padded_frames = padded_input(next_unit.conv1, frames)
                fused_bias_snake_beta(
                    conv2_output,
                    unit.conv2.conv.bias,
                    snake_coefficients[next_unit.act1],
                    padded_frames,
                    residual=residual,
                    sum_out=residual,
                )
            elif block_index < len(decoder_blocks):
                activated = conv2_output.new_empty(conv2_output.shape)
                fused_bias_snake_beta(
                    conv2_output,
                    unit.conv2.conv.bias,
                    snake_coefficients[decoder_blocks[block_index].block[0]],
                    activated,
                    residual=residual,
                )
            else:
                padded, padded_frames = padded_input(codec.decoder[-1], frames)
                fused_bias_snake_beta(
                    conv2_output,
                    unit.conv2.conv.bias,
                    snake_coefficients[codec.decoder[-2]],
                    padded_frames,
                    residual=residual,
                )
    history_exchange(
        padded,
        slot_decode.pool.histories["decoder.final"],
        slot_decode.slot_index,
        new_samples=length,
        is_cold=slot_decode.is_cold,
    )
    return clamped_single_output_conv(padded, codec.decoder[-1].conv, length)
