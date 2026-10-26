# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F
from dac.nn.layers import Snake1d

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import (
    DAC,
    CausalConvNet,
    CausalTransConvNet,
    DecoderBlock,
    ResidualUnit,
)
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
    channels_last_weight,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

SNAKE_BLOCK_ROWS = 32
SNAKE_BLOCK_CHANNELS = 64

if triton is not None:

    @triton.jit(do_not_specialize=["rows", "frames", "batch_frames", "pad_right"])
    def snake_channels_last_kernel(
        x_ptr: torch.Tensor,
        alpha_ptr: torch.Tensor,
        inverse_alpha_ptr: torch.Tensor,
        y_ptr: torch.Tensor,
        rows: int,
        frames: int,
        channels: int,
        batch_frames: int,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
        PAD_LEFT: tl.constexpr = 0,
        pad_right: int = 0,
    ) -> None:
        row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        row_mask = row < rows
        channel_mask = channel < channels
        mask = row_mask[:, None] & channel_mask[None, :]
        output_frames = frames + PAD_LEFT + pad_right
        source_frame = row % output_frames - PAD_LEFT
        source_mask = (source_frame >= 0) & (source_frame < frames)
        source = ((row // output_frames) * batch_frames + source_frame) * channels
        x = tl.load(
            x_ptr + source[:, None] + channel[None, :],
            mask=mask & source_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        alpha = tl.load(alpha_ptr + channel, mask=channel_mask, other=1.0)
        inverse_alpha = tl.load(
            inverse_alpha_ptr + channel, mask=channel_mask, other=1.0
        )
        sine = libdevice.sin(alpha[None, :] * x)
        y = x + inverse_alpha[None, :] * sine * sine
        tl.store(
            y_ptr + row[:, None] * channels + channel[None, :],
            tl.where(source_mask[:, None], y, 0.0).to(y_ptr.dtype.element_ty),
            mask=mask,
        )

else:
    pass


class ChannelsLastDecoder:
    def __init__(self, codec: DAC, *, dtype: torch.dtype) -> None:
        if triton is None:
            raise RuntimeError(
                "the channels-last codec decoder needs Triton; set "
                "stages.vocoder.factory.decode_dtype to float32 to decode without it"
            )
        else:
            pass
        self.codec = codec
        self.snake_alphas: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for module in codec.decoder.modules():
            if isinstance(module, Snake1d):
                alpha = module.alpha.detach().float().reshape(-1).contiguous()
                self.snake_alphas[id(module)] = (alpha, (alpha + 1e-9).reciprocal())
            else:
                pass
        codec.quantizer.to(dtype)
        codec.decoder.to(dtype)
        self.conv_weights: dict[int, torch.Tensor] = {
            id(module): channels_last_weight(module)
            for module in codec.decoder.modules()
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
        }

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        latent = self.codec.quantizer.decode(indices)
        hidden_states = latent.transpose(1, 2).contiguous()
        hidden_states = self.apply_sequence(self.codec.decoder.model, hidden_states)
        return hidden_states.transpose(1, 2)

    def apply_sequence(
        self, sequence: torch.nn.Sequential, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        modules = iter(sequence)
        module = next(modules, None)
        while module is not None:
            if isinstance(module, Snake1d):
                following = next(modules, None)
                if isinstance(following, CausalConvNet):
                    assert following.stride == 1
                    frame_count = hidden_states.shape[1]
                    alignment = (
                        -(following.padding + frame_count)
                    ) % following.conv.dilation[0]
                    padded = self.snake(
                        module, hidden_states, following.padding, alignment
                    )
                    hidden_states = self.causal_conv(
                        following, padded, original_frame_count=frame_count
                    )
                    module = next(modules, None)
                else:
                    hidden_states = self.apply(module, hidden_states)
                    module = following
            else:
                hidden_states = self.apply(module, hidden_states)
                module = next(modules, None)
        return hidden_states

    def apply(
        self, module: torch.nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(module, CausalConvNet):
            return self.causal_conv(module, hidden_states)
        elif isinstance(module, CausalTransConvNet):
            return self.causal_transposed_conv(module, hidden_states)
        elif isinstance(module, Snake1d):
            return self.snake(module, hidden_states)
        elif isinstance(module, ResidualUnit):
            residual = self.apply_sequence(module.block, hidden_states)
            return hidden_states[:, : residual.shape[1]] + residual
        elif isinstance(module, DecoderBlock):
            return self.apply_sequence(module.block, hidden_states)
        elif isinstance(module, torch.nn.Tanh):
            return torch.tanh(hidden_states)
        else:
            raise TypeError(
                f"no channels-last decoding for {type(module).__name__} in the codec"
            )

    def causal_conv(
        self,
        module: CausalConvNet,
        hidden_states: torch.Tensor,
        *,
        original_frame_count: int | None = None,
    ) -> torch.Tensor:
        assert module.stride == 1
        if original_frame_count is None:
            frame_count = hidden_states.shape[1]
            dilation = module.conv.dilation[0]
            alignment = (-(module.padding + frame_count)) % dilation
            padded = F.pad(hidden_states, (0, 0, module.padding, alignment))
        else:
            frame_count = original_frame_count
            padded = hidden_states
        return channels_last_conv1d(
            padded,
            module.conv,
            self.conv_weights[id(module.conv)],
            module.conv.bias,
            frame_count,
        )

    def causal_transposed_conv(
        self, module: CausalTransConvNet, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        output = channels_last_conv_transpose1d(
            hidden_states,
            module.conv,
            self.conv_weights[id(module.conv)],
            module.conv.bias,
        )
        return output[:, : output.shape[1] - (module.kernel_size - module.stride)]

    def snake(
        self,
        module: Snake1d,
        hidden_states: torch.Tensor,
        pad_left: int = 0,
        pad_right: int = 0,
    ) -> torch.Tensor:
        alpha, inverse_alpha = self.snake_alphas[id(module)]
        batch_size, frame_count, channel_count = hidden_states.shape
        assert hidden_states.stride(2) == 1
        assert hidden_states.stride(1) == channel_count
        assert batch_size == 1 or hidden_states.stride(0) % channel_count == 0
        output_frame_count = frame_count + pad_left + pad_right
        output = torch.empty(
            (batch_size, output_frame_count, channel_count),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        rows = batch_size * output_frame_count
        snake_channels_last_kernel[
            (
                triton.cdiv(rows, SNAKE_BLOCK_ROWS),
                triton.cdiv(channel_count, SNAKE_BLOCK_CHANNELS),
            )
        ](
            hidden_states,
            alpha,
            inverse_alpha,
            output,
            rows,
            frame_count,
            channel_count,
            hidden_states.stride(0) // channel_count,
            BLOCK_ROWS=SNAKE_BLOCK_ROWS,
            BLOCK_CHANNELS=SNAKE_BLOCK_CHANNELS,
            PAD_LEFT=pad_left,
            pad_right=pad_right,
        )
        return output
