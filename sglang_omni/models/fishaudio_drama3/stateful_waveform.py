# SPDX-License-Identifier: Apache-2.0
"""Incremental waveform execution using the existing channels-last decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_drama3.stateful_state import DecoderStateArena
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

if triton is not None:

    @triton.jit
    def snake_history_kernel(
        hidden_states,
        alpha,
        inverse_alpha,
        history,
        slot_index,
        padded,
        frame_count: tl.constexpr,
        channel_count: tl.constexpr,
        context_frame_count: tl.constexpr,
        padded_frame_count: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        channel = offsets % channel_count
        frame = offsets // channel_count
        slot = tl.load(slot_index)
        source_frame = frame - context_frame_count
        valid_source = (source_frame >= 0) & (source_frame < frame_count)
        values = tl.load(
            hidden_states + source_frame * channel_count + channel,
            mask=valid_source & (frame < padded_frame_count),
            other=0,
        ).to(tl.float32)
        scales = tl.load(alpha + channel)
        inverses = tl.load(inverse_alpha + channel)
        sine = libdevice.sin(scales * values)
        activated = values + inverses * sine * sine
        previous = tl.load(
            history + (slot * channel_count + channel) * context_frame_count + frame,
            mask=frame < context_frame_count,
            other=0,
        )
        result = tl.where(
            frame < context_frame_count, previous, tl.where(valid_source, activated, 0)
        )
        tl.store(padded + offsets, result, mask=frame < padded_frame_count)

else:
    pass


class StatefulWaveformDecoder:
    def __init__(
        self,
        decoder: ChannelsLastDecoder,
        arena: DecoderStateArena,
        *,
        fuse_snake_history: bool = False,
    ) -> None:
        if fuse_snake_history and triton is None:
            raise ValueError("Snake/history fusion requires Triton")
        else:
            pass
        self.decoder = decoder
        self.arena = arena
        self.fuse_snake_history = fuse_snake_history

    def decode_module(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        slot_index: torch.Tensor,
        path: str,
    ) -> torch.Tensor:
        module_name = type(module).__name__
        if module_name == "CausalConvNet":
            convolution = module.conv
            frame_count = hidden_states.shape[1]
            context_length = (convolution.kernel_size[0] - 1) * convolution.dilation[0]
            if context_length:
                history = self.arena.histories[path]
                hidden_states = torch.cat(
                    (
                        history.index_select(0, slot_index).transpose(1, 2),
                        hidden_states,
                    ),
                    dim=1,
                )
                history.index_copy_(
                    0, slot_index, hidden_states[:, -context_length:].transpose(1, 2)
                )
            else:
                pass
            alignment = (-hidden_states.shape[1]) % convolution.dilation[0]
            hidden_states = F.pad(hidden_states, (0, 0, 0, alignment))
            return channels_last_conv1d(
                hidden_states,
                convolution,
                self.decoder.conv_weights[id(convolution)],
                bias=convolution.bias,
                length=frame_count,
            )
        elif module_name == "CausalTransConvNet":
            convolution = module.conv
            output_length = hidden_states.shape[1] * convolution.stride[0]
            overlap_length = convolution.kernel_size[0] - convolution.stride[0]
            waveform = channels_last_conv_transpose1d(
                hidden_states,
                convolution,
                self.decoder.conv_weights[id(convolution)],
                None,
            )
            if overlap_length:
                carry = self.arena.histories[path]
                waveform[:, :overlap_length] += carry.index_select(
                    0, slot_index
                ).transpose(1, 2)
                carry.index_copy_(
                    0, slot_index, waveform[:, output_length:].transpose(1, 2)
                )
            else:
                pass
            waveform = waveform[:, :output_length]
            if convolution.bias is not None:
                waveform = waveform + convolution.bias[None, None, :]
            else:
                pass
            return waveform.contiguous()
        elif module_name == "ResidualUnit":
            return hidden_states + self.decode_module(
                module.block, hidden_states, slot_index, f"{path}.block"
            )
        elif module_name in ("Decoder", "DecoderBlock"):
            return self.decode_module(
                module.model if module_name == "Decoder" else module.block,
                hidden_states,
                slot_index,
                f"{path}.layers",
            )
        elif isinstance(module, nn.Sequential):
            child_index = 0
            while child_index < len(module):
                child = module[child_index]
                following = (
                    module[child_index + 1] if child_index + 1 < len(module) else None
                )
                if (
                    self.fuse_snake_history
                    and type(child).__name__ == "Snake1d"
                    and type(following).__name__ == "CausalConvNet"
                    and hidden_states.shape[0] == 1
                    and following.conv.kernel_size[0] > 1
                ):
                    convolution = following.conv
                    frame_count = hidden_states.shape[1]
                    channel_count = hidden_states.shape[2]
                    context_frame_count = (
                        convolution.kernel_size[0] - 1
                    ) * convolution.dilation[0]
                    padded_frame_count = context_frame_count + frame_count
                    padded_frame_count += (-padded_frame_count) % convolution.dilation[
                        0
                    ]
                    padded = torch.empty(
                        (1, padded_frame_count, channel_count),
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    history = self.arena.histories[f"{path}.{child_index + 1}"]
                    alpha, inverse_alpha = self.decoder.snake_alphas[id(child)]
                    snake_history_kernel[
                        (triton.cdiv(padded_frame_count * channel_count, 256),)
                    ](
                        hidden_states.contiguous(),
                        alpha,
                        inverse_alpha,
                        history,
                        slot_index,
                        padded,
                        frame_count,
                        channel_count,
                        context_frame_count,
                        padded_frame_count,
                        256,
                    )
                    history.index_copy_(
                        0,
                        slot_index,
                        padded[
                            :, frame_count : context_frame_count + frame_count
                        ].transpose(1, 2),
                    )
                    hidden_states = channels_last_conv1d(
                        padded,
                        convolution,
                        self.decoder.conv_weights[id(convolution)],
                        bias=convolution.bias,
                        length=frame_count,
                    )
                    child_index += 2
                else:
                    hidden_states = self.decode_module(
                        child, hidden_states, slot_index, f"{path}.{child_index}"
                    )
                    child_index += 1
            return hidden_states
        else:
            return self.decoder.apply(module, hidden_states)
