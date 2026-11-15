# SPDX-License-Identifier: Apache-2.0
"""Codec encoder bitwise equal to the module, with snakes fused into conv padding."""

from __future__ import annotations

import torch
from dac.nn.layers import Snake1d

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import (
    CausalConvNet,
    Encoder,
    EncoderBlock,
    ResidualUnit,
    get_extra_padding_for_conv1d,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None

SNAKE_BLOCK_POSITIONS = 1024

if triton is not None:

    @triton.jit(do_not_specialize=["length", "pad_right"])
    def snake_into_padding_kernel(
        x_ptr: torch.Tensor,
        alpha_ptr: torch.Tensor,
        inverse_alpha_ptr: torch.Tensor,
        y_ptr: torch.Tensor,
        length: int,
        channel_count: int,
        x_batch_stride: int,
        x_channel_stride: int,
        BLOCK_POSITIONS: tl.constexpr,
        PAD_LEFT: tl.constexpr,
        pad_right: int,
    ) -> None:
        """Snake of one channel's BLOCK_POSITIONS positions of a (B, C, L) x, zero
        padded as a causal conv pads its input."""
        row = tl.program_id(1).to(tl.int64)
        batch = row // channel_count
        channel = row % channel_count
        output_length = length + PAD_LEFT + pad_right
        position = tl.program_id(0) * BLOCK_POSITIONS + tl.arange(0, BLOCK_POSITIONS)
        source = position - PAD_LEFT
        source_mask = (source >= 0) & (source < length)
        x = tl.load(
            x_ptr + batch * x_batch_stride + channel * x_channel_stride + source,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
        alpha = tl.load(alpha_ptr + channel)
        inverse_alpha = tl.load(inverse_alpha_ptr + channel)
        sine = libdevice.sin(alpha * x)
        # note (Wenyao Gao): with fma contraction this order rounds exactly as the
        # scripted snake's fused kernel; the other orders differ in about a tenth of
        # the elements.
        y = x + inverse_alpha * (sine * sine)
        tl.store(
            y_ptr + row * output_length + position,
            tl.where(source_mask, y, 0.0).to(y_ptr.dtype.element_ty),
            mask=position < output_length,
        )

else:
    pass


class CodecEncoder:
    def __init__(self, encoder: Encoder, autocast_dtype: torch.dtype | None) -> None:
        if autocast_dtype is not None and triton is None:
            raise RuntimeError(
                "the Drama 3 codec encoder needs Triton under autocast; set "
                "stages.preprocessing.factory.reference_encode_dtype to float32 to "
                "encode without it"
            )
        else:
            pass
        self.encoder = encoder
        self.autocast_dtype = autocast_dtype
        self.snake_alphas: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for module in encoder.modules():
            if isinstance(module, ResidualUnit) and not module.causal:
                raise ValueError("the Drama 3 codec encoder runs causal residual units")
            elif isinstance(module, Snake1d):
                alpha = module.alpha.detach().float().reshape(-1).contiguous()
                self.snake_alphas[id(module)] = (alpha, (alpha + 1e-9).reciprocal())
            else:
                pass

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return self.apply_sequence(self.encoder.block, audio)

    def apply_sequence(
        self, sequence: torch.nn.Sequential, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        modules = list(sequence)
        index = 0
        while index < len(modules):
            module = modules[index]
            next_module = modules[index + 1] if index + 1 < len(modules) else None
            if (
                isinstance(module, Snake1d)
                and isinstance(next_module, CausalConvNet)
                and hidden_states.dtype == self.autocast_dtype
            ):
                hidden_states = next_module.conv(
                    self.snake_into_padding(module, next_module, hidden_states)
                )
                index += 2
            elif isinstance(module, ResidualUnit):
                residual = self.apply_sequence(module.block, hidden_states)
                hidden_states = hidden_states[..., : residual.shape[-1]] + residual
                index += 1
            elif isinstance(module, EncoderBlock):
                hidden_states = self.apply_sequence(module.block, hidden_states)
                index += 1
            else:
                hidden_states = module(hidden_states)
                index += 1
        return hidden_states

    def snake_into_padding(
        self, snake: Snake1d, convolution: CausalConvNet, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        alpha, inverse_alpha = self.snake_alphas[id(snake)]
        batch_size, channel_count, length = hidden_states.shape
        assert hidden_states.stride(2) == 1
        pad_right = get_extra_padding_for_conv1d(
            hidden_states,
            convolution.kernel_size,
            convolution.stride,
            convolution.padding,
        )
        output_length = length + convolution.padding + pad_right
        output = torch.empty(
            (batch_size, channel_count, output_length),
            dtype=self.autocast_dtype,
            device=hidden_states.device,
        )
        snake_into_padding_kernel[
            (
                triton.cdiv(output_length, SNAKE_BLOCK_POSITIONS),
                batch_size * channel_count,
            )
        ](
            hidden_states,
            alpha,
            inverse_alpha,
            output,
            length,
            channel_count,
            hidden_states.stride(0),
            hidden_states.stride(1),
            BLOCK_POSITIONS=SNAKE_BLOCK_POSITIONS,
            PAD_LEFT=convolution.padding,
            pad_right=pad_right,
        )
        return output
