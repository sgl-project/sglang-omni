"""Audio helpers used by the retained train/infer pipeline."""

from __future__ import annotations

import torch
import torchaudio.compliance.kaldi as Kaldi
import torchaudio.functional as AF


def high_quality_resample(x, orig_sr, target_sr):
    return AF.resample(
        x,
        orig_freq=orig_sr,
        new_freq=target_sr,
        lowpass_filter_width=64,
        rolloff=0.95,
        resampling_method="sinc_interp_kaiser",
    )


def extract_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int,
    dither: float = 0.0,
    mean_norm: bool = False,
) -> torch.Tensor:
    if waveform.ndim == 1:
        feature_input = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        feature_input = waveform if waveform.size(0) == 1 else waveform[0:1, :]
    else:
        raise ValueError(
            f"FBank expects a 1D or 2D waveform, got shape {tuple(waveform.shape)}."
        )
    features = Kaldi.fbank(
        feature_input,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        dither=dither,
    )
    if mean_norm:
        features = features - features.mean(dim=0, keepdim=True)
    return features
