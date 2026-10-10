from __future__ import annotations

import torch
from einops import einsum, rearrange
from torch import nn

# NeMo defaults omitted from the exported checkpoint config.
DEFAULT_PREEMPHASIS = 0.97
DEFAULT_LOG_ZERO_GUARD = 2**-24


class Featurizer(nn.Module):
    def __init__(self, num_mels: int, num_freqs: int, win_length: int):
        super().__init__()
        self.register_buffer("fb", torch.empty(1, num_mels, num_freqs))
        self.register_buffer("window", torch.empty(win_length))


class LogMelFeatures(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.sample_rate = int(config["sample_rate"])
        self.n_fft = int(config["n_fft"])
        self.hop_length = round(self.sample_rate * float(config["window_stride"]))
        self.win_length = round(self.sample_rate * float(config["window_size"]))
        num_mels = int(config["features"])
        num_freqs = self.n_fft // 2 + 1

        self.left_padding = self.n_fft - self.hop_length
        self.featurizer = Featurizer(num_mels, num_freqs, self.win_length)

    # Wavform -> spectrogram -> mel -> log mel
    def forward(self, waveform_BL: torch.Tensor) -> torch.Tensor:
        waveform_BL = waveform_BL.to(dtype=self.featurizer.window.dtype)
        preemphasized_BL = torch.cat(
            (
                waveform_BL[:, :1],
                waveform_BL[:, 1:] - DEFAULT_PREEMPHASIS * waveform_BL[:, :-1],
            ),
            dim=1,
        )
        padded_BL = nn.functional.pad(preemphasized_BL, (self.left_padding, 0))
        spectrum_BFT = torch.stft(
            padded_BL,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            center=False,
            window=self.featurizer.window,
        )
        power_BFT = spectrum_BFT.abs().square()
        mel_BMT = einsum(self.featurizer.fb[0], power_BFT, "m f, b f t -> b m t")
        log_mel_BMT = torch.log(mel_BMT + DEFAULT_LOG_ZERO_GUARD)
        return rearrange(log_mel_BMT, "b m t -> b t m")
