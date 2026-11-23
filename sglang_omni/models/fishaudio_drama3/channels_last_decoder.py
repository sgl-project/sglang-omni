# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F
from dac.nn.layers import Snake1d

from sglang_omni.models.fishaudio_drama3.codec_transformer import codec_transformer
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
        bias_ptr: torch.Tensor,
        y_ptr: torch.Tensor,
        rows: int,
        frames: int,
        channels: int,
        batch_frames: int,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
        PAD_LEFT: tl.constexpr = 0,
        pad_right: int = 0,
        HAS_BIAS: tl.constexpr = False,
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
        if HAS_BIAS:
            # note (Wenyao Gao): rounded to the activation dtype, as a separate bias
            # add rounds it.
            bias = tl.load(bias_ptr + channel, mask=channel_mask, other=0.0).to(
                tl.float32
            )
            x = (x + bias[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
        else:
            pass
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

    @triton.jit(do_not_specialize=["rows", "rows_per_batch"])
    def residual_bias_add_kernel(
        x_ptr: torch.Tensor,
        x_bias_ptr: torch.Tensor,
        residual_ptr: torch.Tensor,
        residual_bias_ptr: torch.Tensor,
        y_ptr: torch.Tensor,
        rows: int,
        channels: int,
        rows_per_batch: int,
        x_batch_stride: int,
        residual_batch_stride: int,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
        HAS_X_BIAS: tl.constexpr,
        HAS_RESIDUAL_BIAS: tl.constexpr,
    ) -> None:
        """y = (x + x_bias) + (residual + residual_bias), each sum rounded to the
        activation dtype as the separate adds round it."""
        row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        row_mask = row < rows
        channel_mask = channel < channels
        mask = row_mask[:, None] & channel_mask[None, :]
        batch_index = row // rows_per_batch
        row_in_batch = row % rows_per_batch
        dtype = y_ptr.dtype.element_ty
        x = tl.load(
            x_ptr
            + (batch_index * x_batch_stride + row_in_batch * channels)[:, None]
            + channel[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_X_BIAS:
            x_bias = tl.load(x_bias_ptr + channel, mask=channel_mask, other=0.0).to(
                tl.float32
            )
            x = (x + x_bias[None, :]).to(dtype).to(tl.float32)
        else:
            pass
        residual = tl.load(
            residual_ptr
            + (batch_index * residual_batch_stride + row_in_batch * channels)[:, None]
            + channel[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_RESIDUAL_BIAS:
            residual_bias = tl.load(
                residual_bias_ptr + channel, mask=channel_mask, other=0.0
            ).to(tl.float32)
            residual = (residual + residual_bias[None, :]).to(dtype).to(tl.float32)
        else:
            pass
        tl.store(
            y_ptr + row[:, None] * channels + channel[None, :],
            (x + residual).to(dtype),
            mask=mask,
        )

else:
    pass


def conv_stack_left_context(
    sequence: torch.nn.Sequential, context_positions: int = 0
) -> int:
    """Input positions before the first wanted output that the causal conv stack
    reads to produce context_positions positions at its output: the convs' left
    padding, scaled down through each transposed conv's stride."""
    for module in reversed(list(sequence)):
        if isinstance(module, CausalConvNet):
            context_positions += module.padding
        elif isinstance(module, CausalTransConvNet):
            stride = module.stride
            context_positions = -(-context_positions // stride) - (
                -(module.kernel_size - 1) // stride
            )
        elif isinstance(module, (ResidualUnit, DecoderBlock)):
            context_positions = conv_stack_left_context(module.block, context_positions)
        else:
            pass
    return context_positions


def is_single_channel_tail(module: CausalConvNet) -> bool:
    """Whether snake_conv_tanh can serve the conv."""
    conv = module.conv
    return (
        conv.out_channels == 1
        and conv.dilation[0] == 1
        and conv.groups == 1
        and conv.stride[0] == 1
        and conv.bias is not None
    )


def tail_weight(conv: torch.nn.Conv1d) -> torch.Tensor:
    """The (C, taps + 1) weight of a conv onto one channel, padded by a zero column
    to an even GEMM width."""
    return F.pad(conv.weight.data[0], (0, 1)).contiguous()


class ChannelsLastDecoder:
    """The codec decoder over (B, L, C) half activations. Each conv runs without its
    bias, and the next snake or residual kernel adds it as the pending bias, which
    saves one pass over the activation per conv."""

    def __init__(self, codec: DAC, *, dtype: torch.dtype) -> None:
        if triton is None:
            raise RuntimeError(
                "the channels-last codec decoder needs Triton; set "
                "stages.vocoder.factory.decode_dtype to float32 to decode without it"
            )
        else:
            pass
        self.codec = codec
        self.hops_per_frame = int(codec.frame_length) // int(codec.hop_length)
        self.samples_per_hop = int(codec.hop_length)
        self.conv_margin_hops = conv_stack_left_context(codec.decoder.model)
        self.snake_alphas: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for module in codec.decoder.modules():
            if isinstance(module, Snake1d):
                alpha = module.alpha.detach().float().reshape(-1).contiguous()
                self.snake_alphas[id(module)] = (alpha, (alpha + 1e-9).reciprocal())
            else:
                pass
        codec.quantizer.to(dtype)
        codec.decoder.to(dtype)
        self.transformer = codec_transformer(codec.quantizer.post_module)
        self.conv_weights: dict[int, torch.Tensor] = {
            id(module): channels_last_weight(module)
            for module in codec.decoder.modules()
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
        }
        self.conv_biases: dict[int, torch.Tensor | None] = {
            id(module): (
                None
                if module.bias is None
                else module.bias.data.reshape(-1).contiguous()
            )
            for module in codec.decoder.modules()
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
        }
        self.tail_weights: dict[int, torch.Tensor] = {
            id(module.conv): tail_weight(module.conv)
            for module in codec.decoder.modules()
            if isinstance(module, CausalConvNet) and is_single_channel_tail(module)
        }

    def __call__(
        self, indices: torch.Tensor, *, keep_from_frame: int = 0
    ) -> torch.Tensor:
        """The audio of the frames from keep_from_frame on: the transformer sees
        every frame, the conv stack only the kept frames and its left context."""
        # note (Wenyao Gao): quantizer.decode(indices) with its transformer run by
        # the fused forward.
        quantizer = self.codec.quantizer
        indices[:, 0] = torch.clamp(
            indices[:, 0], max=quantizer.semantic_quantizer.codebook_size - 1
        )
        indices[:, 1:] = torch.clamp(
            indices[:, 1:], max=quantizer.quantizer.codebook_size - 1
        )
        latent = quantizer.semantic_quantizer.from_codes(indices[:, :1])[0]
        latent = latent + quantizer.quantizer.from_codes(indices[:, 1:])[0]
        latent = quantizer.upsample(self.transformer(latent))
        hidden_states = latent.transpose(1, 2).contiguous()
        start_hop = max(
            0, keep_from_frame * self.hops_per_frame - self.conv_margin_hops
        )
        hidden_states, pending_bias = self.apply_sequence(
            self.codec.decoder.model, hidden_states[:, start_hop:].contiguous()
        )
        audio = add_bias(hidden_states, pending_bias).transpose(1, 2)
        kept_from_sample = (
            keep_from_frame * self.hops_per_frame - start_hop
        ) * self.samples_per_hop
        return audio[..., kept_from_sample:]

    def apply_sequence(
        self,
        sequence: torch.nn.Sequential,
        hidden_states: torch.Tensor,
        pending_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The sequence applied to hidden_states plus pending_bias; returns the
        activation and the bias still to add to it."""
        modules = list(sequence)
        index = 0
        while index < len(modules):
            module = modules[index]
            following = modules[index + 1] if index + 1 < len(modules) else None
            if isinstance(module, Snake1d) and isinstance(following, CausalConvNet):
                assert following.stride == 1
                after = modules[index + 2] if index + 2 < len(modules) else None
                if (
                    isinstance(after, torch.nn.Tanh)
                    and id(following.conv) in self.tail_weights
                    and hidden_states.dtype == torch.float16
                    and hidden_states.is_cuda
                ):
                    hidden_states = self.snake_conv_tanh(
                        module, following, hidden_states, pending_bias
                    )
                    pending_bias = None
                    index += 3
                else:
                    frame_count = hidden_states.shape[1]
                    alignment = (
                        -(following.padding + frame_count)
                    ) % following.conv.dilation[0]
                    padded = self.snake(
                        module,
                        hidden_states,
                        following.padding,
                        alignment,
                        bias=pending_bias,
                    )
                    hidden_states = self.causal_conv(
                        following, padded, original_frame_count=frame_count
                    )
                    pending_bias = self.conv_biases[id(following.conv)]
                    index += 2
            else:
                hidden_states, pending_bias = self.apply(
                    module, hidden_states, pending_bias
                )
                index += 1
        return hidden_states, pending_bias

    def snake_conv_tanh(
        self,
        snake_module: Snake1d,
        conv_module: CausalConvNet,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """tanh(conv_module(snake_module(x + bias))) of a (B, L, C) half activation, for
        the final conv onto one channel: cuDNN has only a slow channels-last engine for
        it, so the taps run as one GEMM and each frame sums its taps along a diagonal.
        """
        conv = conv_module.conv
        tap_count = conv.kernel_size[0]
        batch_size, frame_count, channel_count = hidden_states.shape
        padded = self.snake(
            snake_module, hidden_states, conv_module.padding, 0, bias=bias
        )
        tap_products = torch.mm(
            padded.view(-1, channel_count),
            self.tail_weights[id(conv)],
            out_dtype=torch.float32,
        ).view(batch_size, -1, tap_count + 1)
        # note (Wenyao Gao): frame t's conv is tap_products[b, t + k, k] summed over
        # the taps k.
        framed = tap_products.as_strided(
            (batch_size, frame_count, tap_count),
            (tap_products.stride(0), tap_count + 1, tap_count + 2),
        )
        conv_output = framed.sum(-1, keepdim=True).to(hidden_states.dtype)
        return torch.tanh(conv_output + conv.bias)

    def apply(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        pending_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(module, CausalConvNet):
            return (
                self.causal_conv(module, add_bias(hidden_states, pending_bias)),
                self.conv_biases[id(module.conv)],
            )
        elif isinstance(module, CausalTransConvNet):
            return (
                self.causal_transposed_conv(
                    module, add_bias(hidden_states, pending_bias)
                ),
                self.conv_biases[id(module.conv)],
            )
        elif isinstance(module, Snake1d):
            return self.snake(module, hidden_states, bias=pending_bias), None
        elif isinstance(module, ResidualUnit):
            residual, residual_bias = self.apply_sequence(
                module.block, hidden_states, pending_bias
            )
            return (
                self.residual_add(
                    hidden_states[:, : residual.shape[1]],
                    pending_bias,
                    residual,
                    residual_bias,
                ),
                None,
            )
        elif isinstance(module, DecoderBlock):
            return self.apply_sequence(module.block, hidden_states, pending_bias)
        elif isinstance(module, torch.nn.Tanh):
            return torch.tanh(add_bias(hidden_states, pending_bias)), None
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
        """The conv without its bias, which the caller adds later."""
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
            None,
            frame_count,
        )

    def causal_transposed_conv(
        self, module: CausalTransConvNet, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """The transposed conv without its bias, which the caller adds later."""
        output = channels_last_conv_transpose1d(
            hidden_states,
            module.conv,
            self.conv_weights[id(module.conv)],
            None,
        )
        return output[:, : output.shape[1] - (module.kernel_size - module.stride)]

    def snake(
        self,
        module: Snake1d,
        hidden_states: torch.Tensor,
        pad_left: int = 0,
        pad_right: int = 0,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """snake(hidden_states + bias), padded by zeros on either side."""
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
            alpha if bias is None else bias,
            output,
            rows,
            frame_count,
            channel_count,
            hidden_states.stride(0) // channel_count,
            BLOCK_ROWS=SNAKE_BLOCK_ROWS,
            BLOCK_CHANNELS=SNAKE_BLOCK_CHANNELS,
            PAD_LEFT=pad_left,
            pad_right=pad_right,
            HAS_BIAS=bias is not None,
        )
        return output

    def residual_add(
        self,
        hidden_states: torch.Tensor,
        hidden_bias: torch.Tensor | None,
        residual: torch.Tensor,
        residual_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """(hidden_states + hidden_bias) + (residual + residual_bias) in one pass."""
        batch_size, frame_count, channel_count = residual.shape
        assert hidden_states.shape == residual.shape
        for activation in (hidden_states, residual):
            assert activation.stride(2) == 1
            assert activation.stride(1) == channel_count
        output = torch.empty_like(residual)
        rows = batch_size * frame_count
        residual_bias_add_kernel[
            (
                triton.cdiv(rows, SNAKE_BLOCK_ROWS),
                triton.cdiv(channel_count, SNAKE_BLOCK_CHANNELS),
            )
        ](
            hidden_states,
            residual if hidden_bias is None else hidden_bias,
            residual,
            residual if residual_bias is None else residual_bias,
            output,
            rows,
            channel_count,
            frame_count,
            hidden_states.stride(0),
            residual.stride(0),
            BLOCK_ROWS=SNAKE_BLOCK_ROWS,
            BLOCK_CHANNELS=SNAKE_BLOCK_CHANNELS,
            HAS_X_BIAS=hidden_bias is not None,
            HAS_RESIDUAL_BIAS=residual_bias is not None,
        )
        return output


def add_bias(hidden_states: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """hidden_states plus a per-channel bias, as the conv would have added it."""
    if bias is None:
        return hidden_states
    else:
        return hidden_states + bias
