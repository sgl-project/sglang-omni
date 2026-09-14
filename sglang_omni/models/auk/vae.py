# SPDX-License-Identifier: MIT AND Apache-2.0
# Copyright (C) 2026 Tencent. All rights reserved.
# Copyright (c) 2022 NVIDIA CORPORATION.
# Derived from Tencent-Hunyuan/AuK; see LICENSE for the MIT permission notice.
# Alias-free activation code adapts junjun3518/alias-free-torch (Apache-2.0).
"""AuK audio VAE: BigVGAN decoder, convolutional encoder, and normalizing flow."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import torch
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


# vendored: alias-free-torch (anti-aliased periodic activations)


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int):
    """Kaiser-windowed sinc low-pass kernel."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter_ = filter_ / filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
        causal: bool = False,
    ):
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        if causal:
            self.pad_left = kernel_size - 1
            self.pad_right = 0
        else:
            self.even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(self.even)
            self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        if self.padding:
            x = torch.nn.functional.pad(
                x, (self.pad_left, self.pad_right), mode=self.padding_mode
            )
        return torch.nn.functional.conv1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )


class UpSample1d(nn.Module):
    def __init__(
        self, ratio: int = 2, kernel_size: int | None = None, causal: bool = False
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.causal = causal
        if causal:
            self.pad = 0
        else:
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = (
                self.pad * self.stride + (self.kernel_size - self.stride) // 2
            )
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        x = torch.nn.functional.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * torch.nn.functional.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        if self.causal:
            return x[..., : -(self.kernel_size - self.stride)]
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(
        self, ratio: int = 2, kernel_size: int | None = None, causal: bool = False
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
            causal=causal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Activation wrapped in anti-aliased upsample and downsample."""

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        causal: bool = False,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size, causal=causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class SnakeBeta(nn.Module):
    """Periodic snake activation."""

    def __init__(self, in_features: int, alpha_logscale: bool = True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        init = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        self.alpha = nn.Parameter(init.clone())
        self.beta = nn.Parameter(init.clone())
        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )


# vendored: commons (causal convs, weight init)


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class Conv1d(nn.Conv1d):
    """Conv1d with causal left padding and optional transpose handling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        **kwargs,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )
        self.in_channels = in_channels
        self.transpose = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True
        if self.causal:
            x = torch.nn.functional.pad(
                x.unsqueeze(2), (self.left_padding, 0, 0, 0)
            ).squeeze(2)
        out = super().forward(x)
        return out.transpose(1, 2) if self.transpose else out


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
        **kwargs,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert (
                kernel_size == 2 * stride
            ), "kernel_size must equal 2*stride when causal."
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.causal = causal
        self.stride = stride
        self.transpose = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True
        x = super().forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x.transpose(1, 2) if self.transpose else x


class Conv1dS(nn.Module):
    """Conv1d with orthogonal/normal init and weight or spectral normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        norm_type: str = "weight_norm",
        init_type: str | None = None,
    ):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
        )
        if init_type == "orthogonal":
            nn.init.orthogonal_(self.layer.weight)
        elif init_type == "normal":
            nn.init.normal_(self.layer.weight, mean=0.0, std=0.01)

        if norm_type == "weight_norm":
            self.layer = weight_norm(self.layer)
        elif norm_type == "spectral_norm":
            self.layer = torch.nn.utils.spectral_norm(self.layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


# vendored: VITS normalizing flow (posterior regularization only)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


class WN(nn.Module):
    """WaveNet-style dilated residual stack used inside each coupling layer."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
        causal: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = weight_norm(
                Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1, causal=causal),
                name="weight",
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            self.in_layers.append(
                weight_norm(
                    Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        causal=causal,
                    ),
                    name="weight",
                )
            )
            res_skip_channels = (
                2 * hidden_channels if i < n_layers - 1 else hidden_channels
            )
            self.res_skip_layers.append(
                weight_norm(
                    Conv1d(hidden_channels, res_skip_channels, 1, causal=causal),
                    name="weight",
                )
            )

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                offset = i * 2 * self.hidden_channels
                g_l = g[:, offset : offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = self.drop(
                fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            )
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm(layer)


class Flip(nn.Module):
    def forward(self, x, *args, reverse: bool = False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: int = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
        causal: bool = True,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = Conv1d(self.half_channels, hidden_channels, 1, causal=causal)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
            causal=causal,
        )
        self.post = Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1, causal=causal
        )
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse: bool = False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x, torch.sum(logs, [1, 2])
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        causal: bool = True,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                    causal=causal,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse: bool = False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResStack(nn.Module):
    def __init__(
        self, channel: int, kernel_size: int = 3, base: int = 3, nums: int = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.LeakyReLU(),
                weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=base**i,
                        padding=base**i,
                    )
                ),
                nn.LeakyReLU(),
                weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=1,
                    )
                ),
            )
            for i in range(nums)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        base_channels: int = 12,
        proj_kernel_size: int = 3,
        stack_kernel_size: int = 3,
        stack_dilation_base: int = 2,
        stacks: int = 6,
        channels=None,
        down_sample_factors=None,
        use_vae: bool = False,
    ):
        super().__init__()
        if channels is None:
            channels = [12, 24, 48, 96, 192, 384, 768]
        if down_sample_factors is None:
            down_sample_factors = [2, 2, 2, 3, 4, 5]

        act_slope = 0.2
        if use_vae:
            out_channels = out_channels * 2

        layers = [
            Conv1dS(in_channels, base_channels, kernel_size=proj_kernel_size, stride=1),
            nn.LeakyReLU(act_slope, True),
        ]
        for (in_c, out_c), down_f in zip(
            zip(channels[:-1], channels[1:]), down_sample_factors
        ):
            layers += [
                Conv1dS(in_c, out_c, kernel_size=down_f * 2, stride=down_f),
                ResStack(out_c, stack_kernel_size, stack_dilation_base, stacks),
                nn.LeakyReLU(act_slope, True),
            ]
        layers += [Conv1dS(channels[-1], out_channels, proj_kernel_size, stride=1)]
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions: torch.Tensor, z_inputs=None) -> torch.Tensor:
        return self.generator(conditions)


class AMPBlock1(nn.Module):
    """Anti-aliased multi-periodicity composition block (BigVGAN)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation=(1, 3, 5),
        causal: bool = True,
        act_causal: bool = False,
        snake_logscale: bool = True,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(
                Conv1d(channels, channels, kernel_size, 1, dilation=d, causal=causal)
            )
            for d in dilation
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            weight_norm(
                Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)
            )
            for _ in dilation
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            Activation1d(
                activation=SnakeBeta(channels, alpha_logscale=snake_logscale),
                causal=act_causal,
            )
            for _ in range(self.num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


@dataclass
class AuKVAEConfig:
    upsample_rates: list = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    upsample_kernel_sizes: list = field(default_factory=lambda: [10, 8, 6, 4, 4, 4])
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: list = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    downsample_rates: list = field(default_factory=lambda: [2, 2, 2, 3, 4, 5])
    downsample_channels: list = field(
        default_factory=lambda: [12, 24, 48, 96, 192, 384, 768]
    )
    snake_logscale: bool = True
    latent_dim: int = 64
    use_vae: bool = True
    causal: bool = True
    flow_hidden_channels: int = 256
    act_causal: bool = True

    @property
    def hop_size(self) -> int:
        return math.prod(self.downsample_rates)

    @classmethod
    def from_dict(cls, config_dict: dict | None) -> AuKVAEConfig:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (config_dict or {}).items() if k in valid})


class BigVGANFlowVAE(nn.Module):
    """Continuous latent VAE used by AuK for both conditioning and synthesis."""

    def __init__(self, h: AuKVAEConfig):
        super().__init__()
        self.h = h
        causal = h.causal
        act_causal = h.act_causal
        self.hop_size = math.prod(h.downsample_rates)

        self.register_buffer(
            "global_mean", torch.zeros(h.latent_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "global_log_std", torch.ones(h.latent_dim, dtype=torch.float32)
        )

        self.audio_encoder = Encoder(
            out_channels=h.latent_dim,
            use_vae=h.use_vae,
            down_sample_factors=h.downsample_rates,
            channels=h.downsample_channels,
            base_channels=h.downsample_channels[0],
        )
        self.flow = ResidualCouplingBlock(
            h.latent_dim, h.flow_hidden_channels, 5, 1, 4, gin_channels=0, causal=causal
        )

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(h.latent_dim, h.upsample_initial_channel, 7, 1, causal=False)
        )
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(
                    AMPBlock1(
                        ch,
                        k,
                        d,
                        causal=causal,
                        act_causal=act_causal,
                        snake_logscale=h.snake_logscale,
                    )
                )

        self.activation_post = Activation1d(
            activation=SnakeBeta(
                h.upsample_initial_channel // (2 ** len(h.upsample_rates)),
                alpha_logscale=h.snake_logscale,
            ),
            causal=act_causal,
        )
        self.conv_post = weight_norm(
            Conv1d(
                h.upsample_initial_channel // (2 ** len(h.upsample_rates)),
                1,
                7,
                1,
                causal=causal,
                bias=False,
            )
        )

        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def encoding_and_normalization(
        self,
        sample: torch.Tensor,
        sample_lengths: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a waveform to a normalized latent."""
        with torch.autocast(device_type=sample.device.type, enabled=False):
            latent_stats = self.audio_encoder(sample.float())
            if sample_lengths is None:
                sample_lengths = torch.LongTensor(
                    [sample.size(-1)] * sample.size(0)
                ).to(sample.device)
            latent_lens = sample_lengths // self.hop_size
            mean, log_std = latent_stats.chunk(2, 1)
            noise = torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
                generator=generator,
            )
            latents = mean + noise * torch.exp(log_std)
            latents = latents.transpose(1, 2).float()
            latents = (latents - self.global_mean.float()) / torch.sqrt(
                self.global_log_std.float()
            )
            latent_lens = torch.clamp(latent_lens, max=latents.size(1))
        return latents, latent_lens

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.float()
        return (
            latents * torch.sqrt(self.global_log_std.float()) + self.global_mean.float()
        )

    def inference_from_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a channel-first latent to a waveform."""
        assert (
            x.size(1) == self.h.latent_dim
        ), f"Input must be [B, D, T], got {tuple(x.shape)}"
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j]
                xs = block(x) if xs is None else xs + block(x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        return torch.clamp(x, min=-1.0, max=1.0)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a normalized latent to a waveform."""
        latents = self.denormalize(latents).permute(0, 2, 1)
        return self.inference_from_latents(latents).squeeze(1)

    def forward(self, data: dict):
        """Training forward: returns the reconstructed sample and the KL term."""
        import torch.distributions as D

        x = data["sample"]
        x = self.audio_encoder(x)
        assert self.h.use_vae

        m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q)

        mask = torch.ones([z.size(0), 1, z.size(-1)]).to(z.device)
        z_p = self.flow(z, mask)

        p_z = D.Normal(torch.zeros_like(m_q), torch.ones_like(logs_q))
        q_z = D.Normal(z_p, torch.exp(logs_q))

        x = self.conv_pre(z)
        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j]
                xs = block(x) if xs is None else xs + block(x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        return {
            "sample": torch.clamp(x, min=-1.0, max=1.0),
            "kl_div": D.kl_divergence(q_z, p_z).mean(),
        }

    def remove_weight_norm(self) -> None:
        for module in self.ups:
            for layer in module:
                remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
