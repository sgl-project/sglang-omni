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

from sglang_omni.models.dots_tts.codec_state_arena import DotsCodecStateArena

if TYPE_CHECKING:
    # note (0xtoward): dots.tts is optional on Apple; CPU scheduler tests still import this module.
    from dots_tts.modules.backbone.layers import Conv1d
    from dots_tts.modules.vocoder.bigvgan import AMPBlock1, Decoder
    from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
else:
    pass


class DotsIncrementalDecoder:
    """Stage geometry and the cold and warm forwards of the AudioVAE decoder.

    The decoder keeps no stream state. The forwards read and write the
    per-slot history in a DotsCodecStateArena.
    """

    def __init__(self, inference: VocoderInference) -> None:
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


def causal_conv(conv: Conv1d, value: torch.Tensor) -> torch.Tensor:
    """Causal Conv1d through cuDNN padding plus a view, instead of a padded copy of the input."""
    output = F.conv1d(
        value, conv.weight, conv.bias, padding=conv.left_padding, dilation=conv.dilation
    )
    return output[..., : value.shape[-1]]


def run_block(block: AMPBlock1, value: torch.Tensor) -> torch.Tensor:
    """AMPBlock1.forward with causal_conv in place of the padded-copy convolutions."""
    activations = block.activations
    for first, second, first_activation, second_activation in zip(
        block.convs1, block.convs2, activations[::2], activations[1::2]
    ):
        hidden = causal_conv(first, first_activation(value))
        value = causal_conv(second, second_activation(hidden)) + value
    return value


__all__ = ["DotsIncrementalDecoder"]
