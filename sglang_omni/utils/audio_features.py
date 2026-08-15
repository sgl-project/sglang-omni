# SPDX-License-Identifier: Apache-2.0
"""Kaldi-compliant fbank with the constant tables cached across requests."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=32)
def _mel_banks(
    num_mel_bins: int,
    padded_window_size: int,
    sample_frequency: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mel filterbank padded to the rfft bin count.

    Note (Jiaxin Deng): torchaudio rebuilds this table inside every
    ``kaldi.fbank`` call; under load it is about a fifth of an ASR process.
    """
    import torchaudio.compliance.kaldi as kaldi

    banks, _ = kaldi.get_mel_banks(
        num_mel_bins,
        padded_window_size,
        sample_frequency,
        20.0,
        0.0,
        100.0,
        -500.0,
        1.0,
    )
    banks = banks.to(device=device, dtype=dtype)
    return torch.nn.functional.pad(banks, (0, 1), mode="constant", value=0)


def cached_fbank(
    waveform: torch.Tensor,
    *,
    num_mel_bins: int = 23,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    window_type: str = "povey",
    sample_frequency: float = 16000.0,
) -> torch.Tensor:
    """``kaldi.fbank`` with a cached mel table, for callers that keep the
    remaining options at their Kaldi defaults (no dither, no energy column,
    snip_edges, log fbank over the power spectrum).

    Note (Jiaxin Deng): the defaults every caller shares are inlined so the
    table can be keyed by the few that vary; bit-identity is asserted in tests.
    """
    import torchaudio.compliance.kaldi as kaldi

    device, dtype = waveform.device, waveform.dtype
    (
        wav,
        window_shift,
        window_size,
        padded_window_size,
    ) = kaldi._get_waveform_and_window_properties(
        waveform, 0, sample_frequency, frame_shift, frame_length, True, 0.97
    )
    strided_input, _ = kaldi._get_window(
        wav,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        0.42,
        True,
        True,
        0.0,
        0.0,
        True,
        0.97,
    )
    spectrum = torch.fft.rfft(strided_input).abs().pow(2.0)
    banks = _mel_banks(
        num_mel_bins, padded_window_size, sample_frequency, device, dtype
    )
    return torch.max(
        torch.mm(spectrum, banks.T), kaldi._get_epsilon(device, dtype)
    ).log()
