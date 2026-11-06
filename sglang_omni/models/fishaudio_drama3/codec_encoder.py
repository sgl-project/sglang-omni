# SPDX-License-Identifier: Apache-2.0
"""The codec encoder as the module runs it under autocast, by the same cuDNN calls on
the same values: each snake that feeds a conv writes the conv's zero padded input in
the autocast dtype in one kernel, in place of a float32 snake, a pad copy and a cast."""

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

SNAKE_BLOCK = 1024

if triton is not None:

    @triton.jit(do_not_specialize=["length", "pad_right"])
    def snake_into_padding_kernel(
        x_ptr: torch.Tensor,
        alpha_ptr: torch.Tensor,
        inverse_alpha_ptr: torch.Tensor,
        y_ptr: torch.Tensor,
        length: int,
        channels: int,
        x_batch_stride: int,
        x_channel_stride: int,
        BLOCK: tl.constexpr,
        PAD_LEFT: tl.constexpr,
        pad_right: int,
    ) -> None:
        """One channel's BLOCK samples of snake(x) for (B, C, L) x, written PAD_LEFT
        samples in with zeros around them: the padded input of a causal conv.

        Keep inverse_alpha * (sine * sine): with fma contraction it rounds exactly
        as the scripted snake's fused kernel, and the other orders differ in about a
        tenth of the elements."""
        row = tl.program_id(1).to(tl.int64)
        batch = row // channels
        channel = row % channels
        output_length = length + PAD_LEFT + pad_right
        position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
        y = x + inverse_alpha * (sine * sine)
        tl.store(
            y_ptr + row * output_length + position,
            tl.where(source_mask, y, 0.0).to(y_ptr.dtype.element_ty),
            mask=position < output_length,
        )

else:
    pass


class CodecEncoder:
    """encoder(audio) with every Snake1d that feeds a CausalConvNet an autocast-dtype
    tensor fused into that conv's padded input; every other module runs itself."""

    def __init__(self, encoder: Encoder, autocast_dtype: torch.dtype | None) -> None:
        if autocast_dtype is not None and triton is None:
            raise RuntimeError("the Drama 3 codec encoder needs Triton under autocast")
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
            following = modules[index + 1] if index + 1 < len(modules) else None
            if (
                isinstance(module, Snake1d)
                and isinstance(following, CausalConvNet)
                and hidden_states.dtype == self.autocast_dtype
            ):
                hidden_states = following.conv(
                    self.snake_into_padding(module, following, hidden_states)
                )
                index += 2
            else:
                hidden_states = self.apply(module, hidden_states)
                index += 1
        return hidden_states

    def apply(
        self, module: torch.nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(module, ResidualUnit):
            residual = self.apply_sequence(module.block, hidden_states)
            return hidden_states[..., : residual.shape[-1]] + residual
        elif isinstance(module, EncoderBlock):
            return self.apply_sequence(module.block, hidden_states)
        else:
            return module(hidden_states)

    def snake_into_padding(
        self, module: Snake1d, conv: CausalConvNet, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """snake(hidden_states) of (B, C, L), zero padded along L as conv's forward
        pads its input, in the autocast dtype."""
        alpha, inverse_alpha = self.snake_alphas[id(module)]
        batch_size, channel_count, length = hidden_states.shape
        assert hidden_states.stride(2) == 1
        pad_right = get_extra_padding_for_conv1d(
            hidden_states, conv.kernel_size, conv.stride, conv.padding
        )
        output_length = length + conv.padding + pad_right
        output = torch.empty(
            (batch_size, channel_count, output_length),
            dtype=self.autocast_dtype,
            device=hidden_states.device,
        )
        snake_into_padding_kernel[
            (triton.cdiv(output_length, SNAKE_BLOCK), batch_size * channel_count)
        ](
            hidden_states,
            alpha,
            inverse_alpha,
            output,
            length,
            channel_count,
            hidden_states.stride(0),
            hidden_states.stride(1),
            BLOCK=SNAKE_BLOCK,
            PAD_LEFT=conv.padding,
            pad_right=pad_right,
        )
        return output
