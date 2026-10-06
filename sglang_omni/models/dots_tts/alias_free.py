# SPDX-License-Identifier: Apache-2.0
"""Inference fusion for the causal AudioVAE alias-free activation."""

from __future__ import annotations

import logging

import torch
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

logger = logging.getLogger(__name__)

FILTER_TAPS = 12
RESAMPLE_RATIO = 2
BLOCK_SAMPLES = 256

if triton is not None:

    @triton.jit
    def upsample_snake_kernel(
        input_pointer,
        bias_pointer,
        filter_pointer,
        alpha_pointer,
        inverse_beta_pointer,
        output_pointer,
        channels: tl.constexpr,
        frames,
        batch_stride,
        channel_stride,
        frame_stride,
        shared_filter: tl.constexpr,
        has_bias: tl.constexpr,
        block_size: tl.constexpr,
    ) -> None:
        row = tl.program_id(1)
        batch = row // channels
        channel = row % channels
        samples = tl.program_id(0) * block_size + tl.arange(0, block_size)
        phase = samples % 2
        first_frame = samples // 2
        filter_offset = tl.full((), 0, tl.int32)
        if not shared_filter:
            filter_offset = channel * 12
        else:
            pass
        accumulator = tl.full((block_size,), 0, tl.float32)
        bias = tl.full((), 0, tl.float32)
        if has_bias:
            bias = tl.load(bias_pointer + channel)
        else:
            pass
        for tap in tl.static_range(6):
            input_frame = first_frame - tap
            valid = (samples < 2 * frames) & (input_frame >= 0) & (input_frame < frames)
            value = tl.load(
                input_pointer
                + batch * batch_stride
                + channel * channel_stride
                + input_frame * frame_stride,
                valid,
                other=0,
            )
            if has_bias:
                # note (0xtoward): the producing conv's bias, added only to real samples.
                value = tl.where(valid, value + bias, 0.0)
            else:
                pass
            coefficient = tl.load(filter_pointer + filter_offset + phase + 2 * tap)
            accumulator = tl.fma(value, coefficient, accumulator)
        upsampled = 2.0 * accumulator
        alpha = tl.load(alpha_pointer + channel)
        inverse_beta = tl.load(inverse_beta_pointer + channel)
        periodic = libdevice.sin(upsampled * alpha)
        activated = upsampled + inverse_beta * (periodic * periodic)
        tl.store(
            output_pointer + row * (2 * frames) + samples,
            activated,
            samples < 2 * frames,
        )

    @triton.jit
    def downsample_kernel(
        input_pointer,
        filter_pointer,
        output_pointer,
        channels: tl.constexpr,
        frames,
        shared_filter: tl.constexpr,
        block_size: tl.constexpr,
    ) -> None:
        row = tl.program_id(1)
        channel = row % channels
        output_frame = tl.program_id(0) * block_size + tl.arange(0, block_size)
        filter_offset = tl.full((), 0, tl.int32)
        if not shared_filter:
            filter_offset = channel * 12
        else:
            pass
        accumulator = tl.full((block_size,), 0, tl.float32)
        for tap in tl.static_range(12):
            input_sample = tl.maximum(2 * output_frame + tap - 11, 0)
            value = tl.load(
                input_pointer + row * (2 * frames) + input_sample,
                output_frame < frames,
                other=0,
            )
            coefficient = tl.load(filter_pointer + filter_offset + tap)
            accumulator = tl.fma(value, coefficient, accumulator)
        tl.store(
            output_pointer + row * frames + output_frame,
            accumulator,
            output_frame < frames,
        )

    @triton.jit
    def residual_bias_kernel(
        conv_pointer,
        bias_pointer,
        residual_pointer,
        output_pointer,
        channels: tl.constexpr,
        frames,
        conv_batch_stride,
        conv_channel_stride,
        residual_batch_stride,
        residual_channel_stride,
        block_size: tl.constexpr,
    ) -> None:
        row = tl.program_id(1)
        batch = row // channels
        channel = row % channels
        frame = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = frame < frames
        convolved = tl.load(
            conv_pointer
            + batch * conv_batch_stride
            + channel * conv_channel_stride
            + frame,
            mask,
            other=0,
        )
        residual = tl.load(
            residual_pointer
            + batch * residual_batch_stride
            + channel * residual_channel_stride
            + frame,
            mask,
            other=0,
        )
        bias = tl.load(bias_pointer + channel)
        tl.store(
            output_pointer + row * frames + frame, (convolved + bias) + residual, mask
        )

else:
    pass


def residual_bias_add(
    convolved: torch.Tensor, bias: torch.Tensor | None, residual: torch.Tensor
) -> torch.Tensor:
    """Add bias and residual in order, without an intermediate tensor."""
    if (
        bias is None
        or triton is None
        or not convolved.is_cuda
        or torch.version.hip is not None
        or convolved.ndim != 3
        or convolved.numel() == 0
        or convolved.dtype != torch.float32
        or residual.dtype != torch.float32
        or residual.device != convolved.device
        or bias.device != convolved.device
        or bias.dtype != torch.float32
        or bias.ndim != 1
        or bias.numel() != convolved.shape[1]
        or bias.stride(0) != 1
        or convolved.shape != residual.shape
        or convolved.stride(-1) != 1
        or residual.stride(-1) != 1
        or torch.is_grad_enabled()
    ):
        return (
            convolved if bias is None else convolved + bias.view(1, -1, 1)
        ) + residual
    else:
        batch_size, channels, frames = convolved.shape
        output = torch.empty(
            convolved.shape, device=convolved.device, dtype=convolved.dtype
        )
        residual_bias_kernel[
            (triton.cdiv(frames, BLOCK_SAMPLES), batch_size * channels)
        ](
            convolved,
            bias,
            residual,
            output,
            channels,
            frames,
            convolved.stride(0),
            convolved.stride(1),
            residual.stride(0),
            residual.stride(1),
            BLOCK_SAMPLES,
            num_warps=4,
        )
        return output


class FusedAliasFree(torch.nn.Module):
    """Keep checkpoint children and use the native chain outside FP32 inference."""

    frozen_alpha: torch.Tensor
    frozen_inverse_beta: torch.Tensor

    def __init__(self, activation: torch.nn.Module) -> None:
        super().__init__()
        self.upsample = activation.upsample
        self.downsample = activation.downsample
        self.act = activation.act
        self.up_ratio = activation.up_ratio
        self.down_ratio = activation.down_ratio
        self.register_buffer(
            "frozen_alpha", torch.empty_like(self.act.alpha), persistent=False
        )
        self.register_buffer(
            "frozen_inverse_beta", torch.empty_like(self.act.beta), persistent=False
        )
        self.refresh_activation_parameters()
        self.register_load_state_dict_post_hook(self.refresh_after_load)
        self.training = activation.training

    @torch.no_grad()
    def refresh_activation_parameters(self) -> None:
        alpha = self.act.alpha.detach()
        beta = self.act.beta.detach()
        if self.act.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        else:
            pass
        self.frozen_alpha.copy_(alpha)
        self.frozen_inverse_beta.copy_(1.0 / (beta + self.act.no_div_by_zero))

    def refresh_after_load(
        self, module: torch.nn.Module, incompatible_keys: IncompatibleKeys
    ) -> None:
        """Refresh after child loads; the hook's mismatch argument is unused."""
        assert module is self
        self.refresh_activation_parameters()

    def train(self, mode: bool = True) -> FusedAliasFree:
        super().train(mode)
        if not mode:
            self.refresh_activation_parameters()
        else:
            pass
        return self

    def forward(
        self, inputs: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Add the convolution bias inside the fused activation."""
        if (
            triton is None
            or not inputs.is_cuda
            or torch.version.hip is not None
            or inputs.dtype != torch.float32
            or torch.is_autocast_enabled("cuda")
            or self.training
            or torch.is_grad_enabled()
            or inputs.ndim != 3
            or inputs.shape[0] == 0
            or inputs.shape[-1] == 0
            or inputs.shape[1] != self.frozen_alpha.numel()
            or self.frozen_alpha.device != inputs.device
            or self.frozen_alpha.dtype != torch.float32
            or (
                bias is not None
                and (
                    bias.device != inputs.device
                    or bias.dtype != torch.float32
                    or bias.ndim != 1
                    or bias.numel() != inputs.shape[1]
                    or bias.stride(0) != 1
                )
            )
        ):
            if bias is not None:
                inputs = inputs + bias.view(1, -1, 1)
            else:
                pass
            return self.downsample(self.act(self.upsample(inputs)))
        else:
            batch_size, channels, frames = inputs.shape
            upsampled = torch.empty(
                (batch_size, channels, RESAMPLE_RATIO * frames),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            output = torch.empty_like(inputs, memory_format=torch.contiguous_format)
            upsample_snake_kernel[
                (
                    triton.cdiv(RESAMPLE_RATIO * frames, BLOCK_SAMPLES),
                    batch_size * channels,
                )
            ](
                inputs,
                inputs if bias is None else bias,
                self.upsample.filter,
                self.frozen_alpha,
                self.frozen_inverse_beta,
                upsampled,
                channels,
                frames,
                *inputs.stride(),
                self.upsample.filter.numel() == FILTER_TAPS,
                bias is not None,
                BLOCK_SAMPLES,
                num_warps=4,
                enable_fp_fusion=False,
            )
            downsample_kernel[
                (triton.cdiv(frames, BLOCK_SAMPLES), batch_size * channels)
            ](
                upsampled,
                self.downsample.lowpass.filter,
                output,
                channels,
                frames,
                self.downsample.lowpass.filter.numel() == FILTER_TAPS,
                BLOCK_SAMPLES,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return output


def install_alias_free_fusion(decoder: torch.nn.Module) -> int:
    """Validate the whole decoder before replacing any supported activation."""
    if triton is None or torch.version.hip is not None:
        logger.warning("Alias-free fusion unavailable; using native decoder")
        return 0
    else:
        replacements: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
        for parent in decoder.modules():
            for name, activation in parent.named_children():
                if type(activation).__name__ == "Activation1d":
                    upsample = activation.upsample
                    downsample = activation.downsample
                    lowpass = downsample.lowpass
                    if type(activation.act).__name__ != "SnakeBeta":
                        logger.warning(
                            "Unsupported alias-free activation; using native decoder"
                        )
                        return 0
                    else:
                        channels = activation.act.in_features
                    supported = (
                        activation.up_ratio == activation.down_ratio == RESAMPLE_RATIO
                        and upsample.ratio == upsample.stride == RESAMPLE_RATIO
                        and downsample.ratio == lowpass.stride == RESAMPLE_RATIO
                        and upsample.causal
                        and upsample.pad == 0
                        and upsample.kernel_size
                        == downsample.kernel_size
                        == lowpass.kernel_size
                        == FILTER_TAPS
                        and lowpass.padding
                        and lowpass.pad_left == FILTER_TAPS - 1
                        and lowpass.pad_right == 0
                        and lowpass.padding_mode == "replicate"
                        and activation.act.alpha.shape
                        == activation.act.beta.shape
                        == (channels,)
                    )
                    for filter_tensor in (upsample.filter, lowpass.filter):
                        supported = supported and (
                            filter_tensor.shape
                            in ((1, 1, FILTER_TAPS), (channels, 1, FILTER_TAPS))
                            and filter_tensor.is_contiguous()
                        )
                    for parameter in (
                        upsample.filter,
                        lowpass.filter,
                        activation.act.alpha,
                        activation.act.beta,
                    ):
                        supported = supported and (
                            parameter.is_cuda
                            and parameter.dtype == torch.float32
                            and parameter.device == activation.act.alpha.device
                        )
                    if not supported:
                        logger.warning(
                            "Unsupported alias-free activation; using native decoder"
                        )
                        return 0
                    else:
                        replacements.append((parent, name, activation))
                else:
                    pass
        candidates = [FusedAliasFree(activation) for _, _, activation in replacements]
        for (parent, name, _), candidate in zip(replacements, candidates):
            setattr(parent, name, candidate)
        logger.info(f"Enabled alias-free fusion for {len(replacements)} activations")
        return len(replacements)
