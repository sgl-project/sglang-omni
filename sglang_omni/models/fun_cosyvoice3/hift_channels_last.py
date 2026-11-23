# SPDX-License-Identifier: Apache-2.0
"""HiFT's decode over channels-last activations: cuDNN's NHWC convolution engines
without their layout transposes, each conv's bias, residual, activation and causal
padding in the kernel that writes the next conv's input, as the object the vocoder
installs in place of the module's decode."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.hift_kernels import (
    add_bias_and_residual,
    leaky_relu_padded,
    overlap_add_inverse,
    short_time_spectrum,
    snake_padded,
)
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_folded_weight,
    channels_last_strides,
    channels_last_weight,
)

# note (ratish): cuDNN's TF32 tensor-core engines take channel counts in multiples of
# four; the spectrum's n_fft + 2 channels are zero padded to a multiple of eight.
CHANNEL_ALIGNMENT = 8

# note (ratish): CosyVoice's decode calls leaky_relu with no slope before conv_post,
# so torch's default applies there, not the stages' lrelu_slope.
CONV_POST_LEAKY_RELU_SLOPE = 0.01


class ChannelsLastHiftDecode:
    """CosyVoice's decode written over (rows, samples, channels) activations: the
    finalized call, and the non-final one that keeps the look-right frames as context.
    """

    def __init__(self, hift: torch.nn.Module, device: torch.device) -> None:
        self.hift = hift
        self.spectrum_frames_per_mel_frame = math.prod(hift.upsample_rates)
        self.n_fft = hift.istft_params["n_fft"]
        self.hop_length = hift.istft_params["hop_len"]
        self.bins = self.n_fft // 2 + 1
        self.spectrum_channel_padding = (-(self.n_fft + 2)) % CHANNEL_ALIGNMENT
        blocks = [*hift.source_resblocks, *hift.resblocks]
        convs = [hift.conv_pre, *hift.ups]
        for block in blocks:
            convs.extend((*block.convs1, *block.convs2))
        # note (ratish): restrided in place, so no second copy is kept; the module's
        # own convs read the same values at these strides.
        for conv in convs:
            conv.weight.data = channels_last_weight(conv)
        self.source_down_weights: list[torch.Tensor] = []
        for source_down in hift.source_downs:
            weight = F.pad(
                source_down.weight.data, (0, 0, 0, self.spectrum_channel_padding)
            )
            stride = source_down.stride[0]
            if stride > 1:
                # note (ratish): a strided conv runs stride one over its phases laid
                # along the channels, the layout cuDNN has fast engines for.
                weight = channels_last_folded_weight(weight, stride)
            else:
                weight = channels_last_strides(weight)
            self.source_down_weights.append(weight)
        self.conv_post_weight = channels_last_strides(
            F.pad(
                hift.conv_post.weight.data,
                (0, 0, 0, 0, 0, self.spectrum_channel_padding),
            )
        )
        self.conv_post_bias = F.pad(
            hift.conv_post.bias.data, (0, self.spectrum_channel_padding)
        )
        self.snake_alphas: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for block in blocks:
            for activation in (*block.activations1, *block.activations2):
                assert not activation.alpha_logscale, "HiFT builds linear scale alphas"
                alpha = activation.alpha.detach().contiguous()
                self.snake_alphas[id(activation)] = (
                    alpha,
                    1.0 / (alpha + activation.no_div_by_zero),
                )
        window = hift.stft_window.to(device=device, dtype=torch.float64)
        angle = (2 * math.pi / self.n_fft) * torch.outer(
            torch.arange(self.bins, device=device, dtype=torch.float64),
            torch.arange(self.n_fft, device=device, dtype=torch.float64),
        )
        self.forward_basis = (
            torch.cat([torch.cos(angle), -torch.sin(angle)]) * window
        ).float()
        # note (ratish): a one sided inverse counts the bins between DC and Nyquist
        # twice.
        one_sided = torch.full((self.bins, 1), 2.0, device=device, dtype=torch.float64)
        one_sided[0] = 1.0
        one_sided[-1] = 1.0
        self.inverse_basis = (
            torch.cat([torch.cos(angle) * one_sided, -torch.sin(angle) * one_sided])
            * window
            / self.n_fft
        ).float()
        self.squared_window = hift.stft_window.to(device).pow(2)

    def spectrum(self, signal: torch.Tensor) -> torch.Tensor:
        """The centered short time spectrum of (rows, samples): (rows, 2 * bins,
        frames), the real parts then the imaginary parts."""
        half = self.n_fft // 2
        padded_signal = F.pad(signal[:, None], (half, half), mode="reflect")[
            :, 0
        ].contiguous()
        return short_time_spectrum(padded_signal, self.forward_basis, self.hop_length)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """The inverse transform of (rows, bins, frames) magnitudes and phases."""
        magnitude = torch.clip(magnitude, max=1e2)
        return overlap_add_inverse(
            (magnitude * torch.cos(phase)).contiguous(),
            (magnitude * torch.sin(phase)).contiguous(),
            self.inverse_basis,
            self.squared_window,
            self.hop_length,
        )

    def source_down(self, stage: int, source_spectrum: torch.Tensor) -> torch.Tensor:
        """The stage's downsampling conv of the source spectrum, (rows, frames,
        channels)."""
        module = self.hift.source_downs[stage]
        frames = source_spectrum.shape[1]
        left_padding = module.causal_padding
        stride = module.stride[0]
        outputs = (left_padding + frames - module.kernel_size[0]) // stride + 1
        padded = F.pad(
            source_spectrum,
            (0, 0, left_padding, (-(left_padding + frames)) % stride),
        )
        return channels_last_conv1d(
            padded, module, self.source_down_weights[stage], module.bias, outputs
        )

    def snake(
        self,
        activation: torch.nn.Module,
        hidden_states: torch.Tensor,
        next_conv: torch.nn.Conv1d,
        *,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        sum_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The Snake of hidden_states + bias + residual as next_conv's input: its causal
        context, then the end rows that make the input divide by its dilation."""
        alpha, inverse_alpha = self.snake_alphas[id(activation)]
        left_padding = next_conv.causal_padding
        dilation = next_conv.dilation[0]
        right_padding = (-(left_padding + hidden_states.shape[1])) % dilation
        return snake_padded(
            hidden_states,
            alpha,
            inverse_alpha,
            left_padding,
            right_padding,
            bias=bias,
            residual=residual,
            sum_out=sum_out,
        )

    def resblock(
        self, block: torch.nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        length = hidden_states.shape[1]
        count = len(block.convs1)
        padded = self.snake(block.activations1[0], hidden_states, block.convs1[0])
        for index in range(count):
            conv1 = block.convs1[index]
            conv2 = block.convs2[index]
            output = channels_last_conv1d(padded, conv1, conv1.weight, None, length)
            padded = self.snake(
                block.activations2[index], output, conv2, bias=conv1.bias
            )
            output = channels_last_conv1d(padded, conv2, conv2.weight, None, length)
            if index + 1 < count:
                next_hidden_states = torch.empty_like(output)
                padded = self.snake(
                    block.activations1[index + 1],
                    output,
                    block.convs1[index + 1],
                    bias=conv2.bias,
                    residual=hidden_states,
                    sum_out=next_hidden_states,
                )
                hidden_states = next_hidden_states
            else:
                hidden_states = add_bias_and_residual(output, conv2.bias, hidden_states)
        return hidden_states

    def __call__(
        self, x: torch.Tensor, s: torch.Tensor, finalize: bool = True
    ) -> torch.Tensor:
        hift = self.hift
        conv_pre = hift.conv_pre
        # note (ratish): three small transposes, the mel, the source spectrum and the
        # output spectrum, in place of one per convolution inside cuDNN.
        source_spectrum = F.pad(
            self.spectrum(s.squeeze(1)).transpose(1, 2),
            (0, self.spectrum_channel_padding),
        ).contiguous()
        if finalize:
            mel = F.pad(x.transpose(1, 2), (0, 0, 0, conv_pre.causal_padding))
        else:
            mel = x.transpose(1, 2).contiguous()
            held_frames = self.spectrum_frames_per_mel_frame * hift.conv_pre_look_right
            source_spectrum = source_spectrum[:, :-held_frames].contiguous()
        # note (ratish): the kernel that reads a conv's output adds its bias; the source
        # and post convs keep torch's.
        hidden_states = channels_last_conv1d(
            mel, conv_pre, conv_pre.weight, None, mel.shape[1] - conv_pre.causal_padding
        )
        last_stage = hift.num_upsamples - 1
        for stage in range(hift.num_upsamples):
            upsample = hift.ups[stage]
            factor = int(upsample.upsample.scale_factor)
            upsampled = leaky_relu_padded(
                hidden_states,
                hift.lrelu_slope,
                factor,
                upsample.causal_padding,
                bias=conv_pre.bias if stage == 0 else None,
            )
            hidden_states = channels_last_conv1d(
                upsampled,
                upsample,
                upsample.weight,
                None,
                hidden_states.shape[1] * factor,
            )
            if stage == last_stage:
                # note (ratish): the reflection pad of one sample is the second sample.
                hidden_states = torch.cat((hidden_states[:, 1:2], hidden_states), dim=1)
            else:
                pass
            source = self.resblock(
                hift.source_resblocks[stage], self.source_down(stage, source_spectrum)
            )
            hidden_states = add_bias_and_residual(hidden_states, upsample.bias, source)
            blocks = hift.resblocks[
                stage * hift.num_kernels : (stage + 1) * hift.num_kernels
            ]
            summed = self.resblock(blocks[0], hidden_states)
            for block in blocks[1:]:
                summed = summed + self.resblock(block, hidden_states)
            hidden_states = summed / hift.num_kernels
        padded = leaky_relu_padded(
            hidden_states,
            CONV_POST_LEAKY_RELU_SLOPE,
            1,
            hift.conv_post.causal_padding,
        )
        hidden_states = channels_last_conv1d(
            padded,
            hift.conv_post,
            self.conv_post_weight,
            self.conv_post_bias,
            hidden_states.shape[1],
        )
        magnitude = torch.exp(hidden_states[..., : self.bins]).transpose(1, 2)
        phase = torch.sin(hidden_states[..., self.bins : 2 * self.bins]).transpose(1, 2)
        audio = self.inverse(magnitude, phase)
        if finalize:
            pass
        else:
            audio = audio[:, : -(self.spectrum_frames_per_mel_frame * self.hop_length)]
        return torch.clamp(audio, -hift.audio_limit, hift.audio_limit)
