# SPDX-License-Identifier: Apache-2.0
"""FireRed's Kaldi fbank/CMVN frontend and exact audio token lengths."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

try:
    import kaldi_native_fbank as knf
except ImportError:
    knf = None


def num_audio_tokens(num_frames: int, downsample_rate: int = 2) -> int:
    """Conformer pads six frames before two stride-two convolutions."""
    if num_frames < 1 or downsample_rate < 1:
        raise ValueError("num_frames and downsample_rate must be positive")
    else:
        pass
    return ((num_frames + 3) // 4) // downsample_rate


class FireRedFeatureExtractor:
    def __init__(
        self,
        means: Sequence[float] | NDArray[np.float64],
        inverse_std_variences: Sequence[float] | NDArray[np.float64],
        *,
        sample_rate: int = 16000,
    ) -> None:
        self.means = np.asarray(means, dtype=np.float64)
        self.inverse_standard_deviations = np.asarray(
            inverse_std_variences, dtype=np.float64
        )
        if self.means.shape != (80,) or self.inverse_standard_deviations.shape != (80,):
            raise ValueError("FireRedASR2 requires 80-dimensional CMVN statistics")
        else:
            pass
        if (
            not np.isfinite(self.means).all()
            or not np.isfinite(self.inverse_standard_deviations).all()
        ):
            raise ValueError("CMVN statistics must be finite")
        else:
            pass
        if sample_rate != 16000:
            raise ValueError("FireRedASR2 requires a 16 kHz frontend")
        else:
            pass
        self.sample_rate = sample_rate

    @classmethod
    def from_pretrained(cls, path: str) -> FireRedFeatureExtractor:
        root = Path(path)
        config = json.loads((root / "preprocessor_config.json").read_text())
        return cls(
            config["means"],
            config["inverse_std_variences"],
            sample_rate=int(config.get("sampling_rate", 16000)),
        )

    def __call__(self, waveform: NDArray[np.float32]) -> torch.Tensor:
        if knf is None:
            raise ImportError(
                "FireRedASR2 requires kaldi-native-fbank; install the firered-asr2 extra"
            )
        else:
            pass

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError("Expected a finite, mono audio waveform")
        else:
            pass
        # note (cocoa00): Request builders run concurrently; each call needs its own extractor.
        options = knf.FbankOptions()
        options.frame_opts.dither = 0.0
        options.frame_opts.snip_edges = True
        options.frame_opts.samp_freq = self.sample_rate
        options.frame_opts.frame_length_ms = 25.0
        options.frame_opts.frame_shift_ms = 10.0
        options.mel_opts.num_bins = 80
        options.mel_opts.debug_mel = False
        fbank = knf.OnlineFbank(options)
        # note (cocoa00): Training uses int16-scale Kaldi features.
        fbank.accept_waveform(self.sample_rate, (waveform * 32768.0).tolist())
        if fbank.num_frames_ready < 5:
            raise ValueError("Audio is too short to produce a FireRedASR2 audio token")
        else:
            pass
        features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        features = (features - self.means) * self.inverse_standard_deviations
        return torch.from_numpy(features.astype(np.float32))


class FireRedAdapter(nn.Module):
    """Frame concatenation and two-layer projection, matching official weights."""

    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int) -> None:
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        usable_frames = x.shape[1] - x.shape[1] % self.ds
        x = (
            x[:, :usable_frames]
            .contiguous()
            .view(x.shape[0], usable_frames // self.ds, -1)
        )
        return (
            self.linear2(self.relu(self.linear1(x))),
            lengths.clamp(max=usable_frames) // self.ds,
        )
