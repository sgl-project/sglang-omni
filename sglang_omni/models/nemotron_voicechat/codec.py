from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

MAX_MAGNITUDE = 100.0
LAYER_NORM_EPS = 1e-6
EXPANSION = 4


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden_BCT):
        centred_BCT = hidden_BCT - hidden_BCT.mean(1, keepdim=True)
        scale_B1T = torch.rsqrt(
            centred_BCT.pow(2).mean(1, keepdim=True) + LAYER_NORM_EPS
        )
        return centred_BCT * scale_B1T * self.weight[:, None] + self.bias[:, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.dwconv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, groups=channels
        )
        self.norm = ChannelLayerNorm(channels)
        self.pwconv1 = nn.Conv1d(channels, channels * EXPANSION, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(channels * EXPANSION, channels, kernel_size=1)

    def forward(self, hidden_BCT):
        residual_BCT = hidden_BCT
        hidden_BCT = functional.pad(hidden_BCT, (self.left_padding, 0))
        hidden_BCT = self.dwconv(hidden_BCT)
        hidden_BCT = self.norm(hidden_BCT)
        hidden_BCT = self.pwconv1(hidden_BCT)
        hidden_BCT = self.act(hidden_BCT)
        hidden_BCT = self.pwconv2(hidden_BCT)
        return residual_BCT + hidden_BCT


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_quantizers = int(config["num_quantizers"])
        codebook_size = int(config["codebook_size"])
        latent_size = int(config["latent_size"])
        self.mus_list = nn.ParameterList(
            nn.Parameter(torch.empty(codebook_size, latent_size))
            for _ in range(num_quantizers)
        )

    def forward(self, codes_TQ):
        levels = [codebook[codes_TQ[:, q]] for q, codebook in enumerate(self.mus_list)]
        return torch.stack(levels, dim=0).sum(0)


class Latent2Wav(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.n_fft = int(config["n_fft"])
        self.hop_length = int(config["hop_length"])
        base = int(config["base_hidden_size"])
        mults = list(config["channel_mult"])
        rates = list(config["rates"])
        blocks = int(config["num_blocks"])
        kernel_size = int(config["kernel_size"])
        groups = int(config["groups"])
        layers: list[nn.Module] = []
        in_channels = int(config["latent_size"])
        for mult, rate in zip(reversed(mults), reversed(rates)):
            channels = base * mult
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    channels,
                    kernel_size=rate,
                    stride=rate,
                    bias=False,
                    groups=groups,
                )
            )
            layers.extend(ConvNeXtBlock(channels, kernel_size) for _ in range(blocks))
            in_channels = channels
        layers.append(nn.Conv1d(in_channels, self.n_fft + 2, kernel_size=1, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, latent_TD):
        hidden_BCT = rearrange(latent_TD, "t d -> 1 d t")
        for layer in self.layers:
            hidden_BCT = layer(hidden_BCT)
        return self._spectrogram_to_wave(hidden_BCT.float())

    def _spectrogram_to_wave(self, spectrum_BCT):
        magnitude_BFT, phase_BFT = spectrum_BCT.chunk(2, dim=1)
        magnitude_BFT = MAX_MAGNITUDE * torch.exp(
            -functional.softplus(-magnitude_BFT + math.log(MAX_MAGNITUDE))
        )
        real_BFT = magnitude_BFT * torch.cos(phase_BFT)
        imaginary_BFT = magnitude_BFT * torch.sin(phase_BFT)
        imaginary_BFT[:, 0] = 0.0
        imaginary_BFT[:, -1] = 0.0
        return self._inverse_stft(torch.complex(real_BFT, imaginary_BFT))

    def _inverse_stft(self, spectrum_BFT):
        window_length = self.n_fft
        frames = spectrum_BFT.shape[-1]
        window_W = torch.hann_window(window_length, device=spectrum_BFT.device)
        frames_BWT = torch.fft.irfft(
            spectrum_BFT, n=self.n_fft, dim=-2, norm="backward"
        )
        frames_BWT = frames_BWT * window_W[:, None]

        length = (frames - 1) * self.hop_length + window_length
        fold = functional.fold(
            frames_BWT,
            (1, length),
            kernel_size=(1, window_length),
            stride=(1, self.hop_length),
        )
        envelope = functional.fold(
            window_W.square().expand(frames, -1).transpose(0, 1)[None],
            (1, length),
            kernel_size=(1, window_length),
            stride=(1, self.hop_length),
        )
        pad = (window_length - self.hop_length) // 2
        return (fold / envelope.clamp_min(1e-11))[..., 0, 0, pad:-pad]


class RVQVAEDecoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.prvq = ResidualVectorQuantizer(config)
        self.decoder = Latent2Wav(config)
        self.samples_per_frame = int(config["wav_to_token_ratio"])
        # The codes marking an utterance's edges sit above the codebook and
        # index nothing, so they decode as silence.
        self.register_buffer(
            "control_codes", torch.empty(3, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "silence_codes",
            torch.empty(int(config["num_quantizers"]), dtype=torch.long),
            persistent=False,
        )

    def forward(self, codes_TQ):
        codes_TQ = torch.where(
            torch.isin(codes_TQ, self.control_codes),
            self.silence_codes.to(codes_TQ.device).expand_as(codes_TQ),
            codes_TQ,
        )
        wave_BS = self.decoder(self.prvq(codes_TQ))
        return rearrange(wave_BS, "1 s -> s")
