# SPDX-License-Identifier: Apache-2.0
"""Incremental decoding for the dots.tts AudioVAE decoder.

The decoder is causal: an output sample depends only on latent frames up to a
small lookahead. A stream that keeps the recent inputs of every stage can
decode only its new frames instead of re-decoding a whole window each step.
"""

from __future__ import annotations

import itertools
import math
import operator
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang_omni.models.dots_tts.alias_free import (
    FusedAliasFree,
    alias_free_channels_last,
    residual_bias_add,
    residual_bias_add_channels_last,
)
from sglang_omni.models.dots_tts.codec_state_arena import DotsCodecStateArena
from sglang_omni.utils.channels_last_conv import (
    channels_last_weight,
    is_channels_last_conv_device,
)

if TYPE_CHECKING:
    # note (0xtoward): dots.tts is optional on Apple; CPU scheduler tests still import this module.
    from dots_tts.modules.backbone.layers import Conv1d
    from dots_tts.modules.vocoder.alias_free_act import Activation1d
    from dots_tts.modules.vocoder.bigvgan import AMPBlock1, Decoder
    from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
else:
    pass


class DotsIncrementalDecoder:
    """Stage geometry and the cold and warm forwards of the AudioVAE decoder.

    The decoder keeps no stream state. The forwards read and write the
    per-slot history in a DotsCodecStateArena.
    """

    def __init__(
        self, inference: VocoderInference, *, channels_last: bool = True
    ) -> None:
        decoder = inference.vocoder.decoder
        self.decoder: Decoder = decoder
        self.lookahead: int = int(
            inference._decoder_stream_lookahead()
        )  # noqa: leading-underscore  # upstream spelling
        # note (0xtoward): conv_pre is centred, so its left history is twice the
        # lookahead. Each output frame also waits for lookahead future frames.
        self.conv_pre_context: int = int(decoder.conv_pre.kernel_size[0]) - 1
        if self.conv_pre_context != 2 * self.lookahead:
            raise ValueError(
                f"conv_pre kernel {self.conv_pre_context + 1} is not centred on "
                f"lookahead {self.lookahead}"
            )
        else:
            pass
        self.num_kernels: int = int(decoder.num_kernels)
        if any(type(block).__name__ != "AMPBlock1" for block in decoder.resblocks):
            raise ValueError("The incremental decoder expects AMPBlock1 resblocks")
        else:
            pass
        # note (0xtoward): dots' causal ConvTranspose1d overwrites stride with a plain int.
        self.upsample_strides: list[int] = [
            int(
                stage[0].stride[0]
                if isinstance(stage[0].stride, tuple)
                else stage[0].stride
            )
            for stage in decoder.ups
        ]
        # note (0xtoward): output samples per latent frame after each upsampling stage.
        self.upsample_factors: list[int] = list(
            itertools.accumulate(self.upsample_strides, operator.mul)
        )
        # note (0xtoward): how many of its own input samples each upsampling
        # stage must see on the left. The last stage also covers the output
        # activation and conv_post.
        self.stage_contexts: list[int] = []
        last_stage = len(self.upsample_strides) - 1
        for stage in range(len(self.upsample_strides)):
            blocks = decoder.resblocks[
                stage * self.num_kernels : (stage + 1) * self.num_kernels
            ]
            context = max(
                int(inference._ampblock_left_context(block)) for block in blocks
            )  # noqa: leading-underscore  # upstream spelling
            if stage == last_stage:
                context += int(
                    inference._activation_left_context(decoder.activation_post)
                )  # noqa: leading-underscore  # upstream spelling
                context += int(
                    inference._conv1d_left_context(decoder.conv_post)
                )  # noqa: leading-underscore  # upstream spelling
            else:
                pass
            self.stage_contexts.append(context)
        # note (0xtoward): a slot is warm once its history covers every stage's
        # context, measured in latent frames. Until then it decodes cold.
        self.warm_history_frames: int = max(
            [1]
            + [
                math.ceil(context / factor)
                for context, factor in zip(self.stage_contexts, self.upsample_factors)
            ]
        )
        self.use_tanh: bool = bool(decoder.h.get("use_tanh_at_final", True))
        self.device: torch.device = decoder.conv_pre.weight.device
        self.dtype: torch.dtype = decoder.conv_pre.weight.dtype
        self.latent_channels: int = int(decoder.conv_pre.in_channels)
        # note (0xtoward): a warm slot's history covers every op's receptive field,
        # so the warm path can run each conv and activation on valid samples only,
        # in (B, T, C) layout: no padding, slicing or cuDNN layout transposes.
        self.block_contexts: list[int] = [
            int(
                inference._ampblock_left_context(block)
            )  # noqa: leading-underscore  # upstream spelling
            for block in decoder.resblocks[: self.num_kernels]
        ]
        # note (0xtoward): warm steps and cold windows both run in (B, T, C) layout
        # on this path, so cuDNN needs no layout transposes around the convs.
        self.use_channels_last: bool = (
            channels_last
            and is_channels_last_conv_device(self.device)
            and self.dtype == torch.float32
            and all(
                isinstance(module, FusedAliasFree)
                for module in [
                    *(
                        activation
                        for block in decoder.resblocks
                        for activation in block.activations
                    ),
                    decoder.activation_post,
                ]
            )
        )
        self.channels_last_weights: dict[int, torch.Tensor] = {}
        if self.use_channels_last:
            for module in decoder.modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                    self.channels_last_weights[id(module)] = channels_last_weight(
                        module
                    )
                else:
                    pass
        else:
            pass

    def new_state_arena(self, num_slots: int) -> DotsCodecStateArena:
        """Allocate zeroed history for num_slots streams, shaped for this decoder."""
        return DotsCodecStateArena(
            num_slots=num_slots,
            latent_channels=self.latent_channels,
            conv_pre_context=self.conv_pre_context,
            upsample_channels=[int(stage[0].in_channels) for stage in self.decoder.ups],
            stage_channels=[int(stage[0].out_channels) for stage in self.decoder.ups],
            stage_contexts=self.stage_contexts,
            device=self.device,
            dtype=self.dtype,
        )

    def run_stage(self, index: int, value: torch.Tensor) -> torch.Tensor:
        """Run upsampling stage index's resblocks, plus the output head after the last stage."""
        total = None
        for block in self.decoder.resblocks[
            index * self.num_kernels : (index + 1) * self.num_kernels
        ]:
            output = run_block(block, value)
            total = output if total is None else total + output
        value = total / self.num_kernels
        if index == len(self.upsample_strides) - 1:
            value = causal_conv(
                self.decoder.conv_post, self.decoder.activation_post(value)
            )
            value = (
                torch.tanh(value)
                if self.use_tanh
                else torch.clamp(value, min=-1.0, max=1.0)
            )
        else:
            pass
        return value

    def cold_forward(
        self,
        arena: DotsCodecStateArena,
        window: torch.Tensor,
        slot_index: torch.Tensor,
        stable: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Decode left-aligned latent windows [B, C, L] and record each slot's history.

        valid is the number of real frames per row. stable is the frame count
        whose output no later input can change. History ends at the stable
        frames, so the next warm step continues exactly there.
        """
        record_context(arena.conv_pre_history, window, slot_index, valid)
        if self.use_channels_last:
            return self.cold_channels_last(arena, window, slot_index, stable)
        else:
            pass
        value = self.decoder.conv_pre(window)
        previous_factor = 1
        for index, upsample in enumerate(self.decoder.ups):
            record_context(
                arena.upsample_histories[index],
                value,
                slot_index,
                stable * previous_factor,
            )
            value = upsample[0](value)
            record_context(
                arena.stage_histories[index],
                value,
                slot_index,
                stable * self.upsample_factors[index],
            )
            value = self.run_stage(index, value)
            previous_factor = self.upsample_factors[index]
        return value

    def cold_channels_last(
        self,
        arena: DotsCodecStateArena,
        window: torch.Tensor,
        slot_index: torch.Tensor,
        stable: torch.Tensor,
    ) -> torch.Tensor:
        """cold_forward after conv_pre history, in (B, T, C) layout without layout transforms.

        Every conv and activation pads like the native module, so the window
        decodes from the stream start exactly as the NCL path does.
        """
        value = padded_conv(
            self.decoder.conv_pre,
            window.transpose(1, 2).contiguous(),
            self.channels_last_weights,
            with_bias=True,
        )
        previous_factor = 1
        for index, upsample in enumerate(self.decoder.ups):
            record_context_channels_last(
                arena.upsample_histories[index],
                value,
                slot_index,
                stable * previous_factor,
            )
            conv = upsample[0]
            stride = self.upsample_strides[index]
            # note (0xtoward): the causal transposed conv drops its last stride outputs.
            value = (
                F.conv_transpose2d(
                    value.transpose(1, 2).unsqueeze(2),
                    self.channels_last_weights[id(conv)].unsqueeze(2),
                    conv.bias,
                    stride=(1, stride),
                )
                .squeeze(2)
                .transpose(1, 2)[:, :-stride]
            )
            record_context_channels_last(
                arena.stage_histories[index],
                value,
                slot_index,
                stable * self.upsample_factors[index],
            )
            value = self.run_stage_channels_last(index, value, padded=True)
            previous_factor = self.upsample_factors[index]
        return value.transpose(1, 2)

    def warm_forward(
        self,
        arena: DotsCodecStateArena,
        frames: torch.Tensor,
        slot_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode new latent frames [B, C, n] into n frames of audio.

        Each stage prepends the slot's history to its new input, keeps the
        tail as the next history and drops the outputs that only the history
        produced. The audio ends lookahead frames before the input boundary.
        """
        joined = torch.cat(
            [arena.conv_pre_history.index_select(0, slot_index), frames], dim=-1
        )
        arena.conv_pre_history[slot_index] = joined[..., -self.conv_pre_context :]
        value = F.conv1d(
            joined, self.decoder.conv_pre.weight, self.decoder.conv_pre.bias
        )
        if self.use_channels_last:
            return self.warm_channels_last(arena, value.transpose(1, 2), slot_index)
        else:
            pass
        for index, upsample in enumerate(self.decoder.ups):
            joined = torch.cat(
                [arena.upsample_histories[index].index_select(0, slot_index), value],
                dim=-1,
            )
            arena.upsample_histories[index][slot_index] = joined[..., -1:]
            value = upsample[0](joined)[..., self.upsample_strides[index] :]
            joined = torch.cat(
                [arena.stage_histories[index].index_select(0, slot_index), value],
                dim=-1,
            )
            arena.stage_histories[index][slot_index] = joined[
                ..., -self.stage_contexts[index] :
            ]
            value = self.run_stage(index, joined)[..., -value.shape[-1] :]
        return value

    def warm_channels_last(
        self,
        arena: DotsCodecStateArena,
        value: torch.Tensor,
        slot_index: torch.Tensor,
    ) -> torch.Tensor:
        """Warm stages on a (B, n, C) conv_pre output, every op on valid samples only."""
        for index, upsample in enumerate(self.decoder.ups):
            conv = upsample[0]
            joined = torch.cat(
                [
                    arena.upsample_histories[index]
                    .index_select(0, slot_index)
                    .transpose(1, 2),
                    value,
                ],
                dim=1,
            )
            arena.upsample_histories[index][slot_index] = joined[:, -1:].transpose(1, 2)
            stride = self.upsample_strides[index]
            # note (0xtoward): the causal transposed conv drops its last stride
            # outputs; the first stride outputs belong to the history sample.
            upsampled = (
                F.conv_transpose2d(
                    joined.transpose(1, 2).unsqueeze(2),
                    self.channels_last_weights[id(conv)].unsqueeze(2),
                    conv.bias,
                    stride=(1, stride),
                )
                .squeeze(2)
                .transpose(1, 2)[:, stride:-stride]
            )
            joined = torch.cat(
                [
                    arena.stage_histories[index]
                    .index_select(0, slot_index)
                    .transpose(1, 2),
                    upsampled,
                ],
                dim=1,
            )
            arena.stage_histories[index][slot_index] = joined[
                :, -self.stage_contexts[index] :
            ].transpose(1, 2)
            value = self.run_stage_channels_last(index, joined, padded=False)
        return value.transpose(1, 2)

    def run_stage_channels_last(
        self, index: int, value: torch.Tensor, *, padded: bool
    ) -> torch.Tensor:
        """run_stage on a (B, T, C) input.

        padded takes an input that starts at the stream start and returns T
        outputs; otherwise the input carries every block's full history and
        only the last outputs are computed.
        """
        widest = max(self.block_contexts)
        total = None
        for block, context in zip(
            self.decoder.resblocks[
                index * self.num_kernels : (index + 1) * self.num_kernels
            ],
            self.block_contexts,
        ):
            output = run_block_channels_last(
                block,
                value if padded else value[:, widest - context :],
                self.channels_last_weights,
                padded=padded,
            )
            total = output if total is None else total + output
        value = total / self.num_kernels
        if index == len(self.upsample_strides) - 1:
            conv = padded_conv if padded else valid_conv
            value = conv(
                self.decoder.conv_post,
                alias_free_channels_last(
                    self.decoder.activation_post, value, None, padded=padded
                ),
                self.channels_last_weights,
                with_bias=True,
            )
            value = (
                torch.tanh(value)
                if self.use_tanh
                else torch.clamp(value, min=-1.0, max=1.0)
            )
        else:
            pass
        return value


def record_context(
    history: torch.Tensor,
    value: torch.Tensor,
    slot_index: torch.Tensor,
    end: torch.Tensor,
) -> None:
    """Store value[..., end - width : end] of each row as its slot's history.

    Positions before the start of the stream are stored as zeros, which is
    what the causal padding of the first window used.
    """
    width = history.shape[-1]
    positions = end.unsqueeze(1) - width + torch.arange(width, device=value.device)
    picked = value.gather(
        -1, positions.clamp(min=0).unsqueeze(1).expand(-1, value.shape[1], -1)
    )
    history[slot_index] = picked * (positions >= 0).unsqueeze(1).to(picked.dtype)


def record_context_channels_last(
    history: torch.Tensor,
    value: torch.Tensor,
    slot_index: torch.Tensor,
    end: torch.Tensor,
) -> None:
    """record_context for a (B, T, C) value; the arena keeps each slot's (C, width) rows."""
    width = history.shape[-1]
    positions = end.unsqueeze(1) - width + torch.arange(width, device=value.device)
    picked = value.gather(
        1, positions.clamp(min=0).unsqueeze(2).expand(-1, -1, value.shape[2])
    )
    history[slot_index] = (
        picked * (positions >= 0).unsqueeze(2).to(picked.dtype)
    ).transpose(1, 2)


def causal_conv(
    conv: Conv1d, value: torch.Tensor, *, with_bias: bool = True
) -> torch.Tensor:
    """Causal Conv1d through cuDNN padding plus a view, instead of a padded copy of the input."""
    output = F.conv1d(
        value,
        conv.weight,
        conv.bias if with_bias else None,
        padding=conv.left_padding,
        dilation=conv.dilation,
    )
    return output[..., : value.shape[-1]]


def activate(
    activation: Activation1d | FusedAliasFree,
    value: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Activate value + bias, adding the bias inside the fused kernel when there is one."""
    if isinstance(activation, FusedAliasFree):
        return activation(value, bias=bias)
    elif bias is not None:
        return activation(value + bias.view(1, -1, 1))
    else:
        return activation(value)


def run_block(block: AMPBlock1, value: torch.Tensor) -> torch.Tensor:
    """AMPBlock1.forward with copy-free causal convs and conv biases folded into the next kernel."""
    activations = block.activations
    for first, second, first_activation, second_activation in zip(
        block.convs1, block.convs2, activations[::2], activations[1::2]
    ):
        hidden = causal_conv(first, first_activation(value), with_bias=False)
        activated = activate(second_activation, hidden, first.bias)
        value = residual_bias_add(
            causal_conv(second, activated, with_bias=False), second.bias, value
        )
    return value


def valid_conv(
    conv: Conv1d,
    value: torch.Tensor,
    channels_last_weights: dict[int, torch.Tensor],
    *,
    with_bias: bool,
) -> torch.Tensor:
    """Conv1d of a (B, T, C) activation without padding, as one channels-last cuDNN call."""
    output = F.conv2d(
        value.transpose(1, 2).unsqueeze(2),
        channels_last_weights[id(conv)].unsqueeze(2),
        conv.bias if with_bias else None,
        dilation=(1, conv.dilation[0]),
    )
    return output.squeeze(2).transpose(1, 2)


def padded_conv(
    conv: Conv1d,
    value: torch.Tensor,
    channels_last_weights: dict[int, torch.Tensor],
    *,
    with_bias: bool,
) -> torch.Tensor:
    """Conv1d of a (B, T, C) activation with the module's own zero padding, as one channels-last cuDNN call."""
    frames = int(value.shape[1])
    padding = int(conv.left_padding) if conv.causal else int(conv.padding[0])
    output = F.conv2d(
        value.transpose(1, 2).unsqueeze(2),
        channels_last_weights[id(conv)].unsqueeze(2),
        conv.bias if with_bias else None,
        dilation=(1, conv.dilation[0]),
        padding=(0, padding),
    )
    # note (0xtoward): a causal conv pads both sides by its left context; the
    # first T outputs are the causal ones.
    return output.squeeze(2).transpose(1, 2)[:, :frames]


def run_block_channels_last(
    block: AMPBlock1,
    value: torch.Tensor,
    channels_last_weights: dict[int, torch.Tensor],
    *,
    padded: bool,
) -> torch.Tensor:
    """run_block on a (B, T, C) input.

    padded runs from the stream start with the native padding and keeps T
    outputs; otherwise only valid samples are computed and each conv pair
    shortens the input by its receptive field.
    """
    conv = padded_conv if padded else valid_conv
    activations = block.activations
    for first, second, first_activation, second_activation in zip(
        block.convs1, block.convs2, activations[::2], activations[1::2]
    ):
        hidden = conv(
            first,
            alias_free_channels_last(first_activation, value, None, padded=padded),
            channels_last_weights,
            with_bias=False,
        )
        convolved = conv(
            second,
            alias_free_channels_last(
                second_activation, hidden, first.bias, padded=padded
            ),
            channels_last_weights,
            with_bias=False,
        )
        value = residual_bias_add_channels_last(
            convolved, second.bias, value[:, -convolved.shape[1] :]
        )
    return value


__all__ = ["DotsIncrementalDecoder"]
