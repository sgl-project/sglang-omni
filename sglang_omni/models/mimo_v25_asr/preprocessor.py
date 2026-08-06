# SPDX-License-Identifier: Apache-2.0
"""Waveform-to-RVQ preprocessing for MiMo-V2.5-ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from sglang_omni.models.mimo_v25_asr.prompt import AUDIO_CHANNELS, pad_audio_codes
from sglang_omni.utils.audio import load_audio


@dataclass(frozen=True)
class MiMoAudioTokenizerSettings:
    sampling_rate: int = 24_000
    n_fft: int = 960
    hop_length: int = 240
    window_size: int = 960
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    chunk_seconds: float = 30.0

    @classmethod
    def from_config(cls, config: Any) -> MiMoAudioTokenizerSettings:
        def value(name: str, default: Any) -> Any:
            if isinstance(config, dict):
                return config.get(name, default)
            return getattr(config, name, default)

        return cls(
            sampling_rate=int(value("sampling_rate", cls.sampling_rate)),
            n_fft=int(value("nfft", cls.n_fft)),
            hop_length=int(value("hop_length", cls.hop_length)),
            window_size=int(value("window_size", cls.window_size)),
            n_mels=int(value("n_mels", cls.n_mels)),
            f_min=float(value("fmin", cls.f_min)),
            f_max=(
                None
                if value("fmax", cls.f_max) is None
                else float(value("fmax", cls.f_max))
            ),
        )


class MiMoAudioEncoder(Protocol):
    def encode(
        self,
        *,
        input_features: torch.Tensor,
        input_lens: torch.Tensor,
        return_codes_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class MiMoAudioPreprocessor:
    """Official V2.5-ASR log-mel preparation with 30s waveform chunking."""

    def __init__(
        self,
        settings: MiMoAudioTokenizerSettings | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.settings = settings or MiMoAudioTokenizerSettings()
        self.device = torch.device(device)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.settings.sampling_rate,
            n_fft=self.settings.n_fft,
            hop_length=self.settings.hop_length,
            win_length=self.settings.window_size,
            f_min=self.settings.f_min,
            f_max=self.settings.f_max,
            n_mels=self.settings.n_mels,
            power=1.0,
            center=True,
        ).to(self.device)

    def load_waveform(self, source: Any) -> torch.Tensor:
        waveform = load_audio(
            source,
            source_name="MiMo-V2.5-ASR",
            target_sample_rate=self.settings.sampling_rate,
        )
        if waveform.size == 0:
            raise ValueError("MiMo-V2.5-ASR input waveform is empty")
        return torch.from_numpy(np.asarray(waveform, dtype=np.float32)).to(self.device)

    def waveform_to_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = torch.as_tensor(
            waveform,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        if waveform.numel() == 0:
            raise ValueError("MiMo-V2.5-ASR input waveform is empty")
        spectrogram = self.mel_transform(waveform[None, :])
        return (
            torch.log(torch.clamp(spectrogram, min=1.0e-7)).squeeze(0).transpose(0, 1)
        )

    @torch.no_grad()
    def encode_waveform(
        self,
        waveform: torch.Tensor,
        encoder: MiMoAudioEncoder,
    ) -> torch.Tensor:
        """Encode with the official 30-second waveform chunk loop."""

        wav = torch.as_tensor(
            waveform, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if wav.numel() == 0:
            raise ValueError("MiMo-V2.5-ASR input waveform is empty")

        chunk_samples = int(self.settings.chunk_seconds * self.settings.sampling_rate)
        n_fft = self.settings.n_fft
        total_samples = int(wav.shape[-1])
        code_parts: list[torch.Tensor] = []
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            # Merge a too-short trailing chunk (would break mel reflect padding).
            if 0 < total_samples - end < n_fft:
                end = total_samples
            chunk = wav[start:end]
            if chunk.shape[-1] < n_fft:
                chunk = F.pad(chunk, (0, n_fft - chunk.shape[-1]))
            mel = self.waveform_to_log_mel(chunk)
            packed_codes, _ = encoder.encode(
                input_features=mel.to(self.device),
                input_lens=torch.tensor([mel.size(0)], device=self.device),
                return_codes_only=True,
            )
            if packed_codes.ndim != 2 or packed_codes.shape[0] < AUDIO_CHANNELS:
                raise ValueError(
                    "MiMo tokenizer returned invalid codes with shape "
                    f"{tuple(packed_codes.shape)}"
                )
            code_parts.append(packed_codes[:AUDIO_CHANNELS].detach())
            start = end

        codes = torch.cat(code_parts, dim=-1).transpose(0, 1).cpu().long()
        return pad_audio_codes(codes)

    def encode_source(
        self,
        source: Any,
        encoder: MiMoAudioEncoder,
    ) -> tuple[torch.Tensor, float]:
        waveform = self.load_waveform(source)
        duration_s = float(waveform.numel() / self.settings.sampling_rate)
        return self.encode_waveform(waveform, encoder), duration_s


def default_tokenizer_config_path(tokenizer_snapshot: str | Path) -> Path:
    return Path(tokenizer_snapshot) / "config.json"


__all__ = [
    "MiMoAudioEncoder",
    "MiMoAudioPreprocessor",
    "MiMoAudioTokenizerSettings",
    "default_tokenizer_config_path",
]
