# SPDX-License-Identifier: Apache-2.0
"""The log-mel filterbank the ELLSA 2 audio encoder was trained on.

The checkpoint reads lhotse 1.33's `Fbank(FbankConfig(num_mel_bins=128))`,
computed on CPU in float32. This is that computation (lhotse
`Wav2LogFilterBank` with its defaults, Apache-2.0), operation for operation,
so the features match without depending on lhotse.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE_HZ = 16000
NUM_MEL_BINS = 128
# lhotse FbankConfig defaults: 25 ms frames every 10 ms, padded to the next power of two.
FRAME_SAMPLES = 400
HOP_SAMPLES = 160
FFT_SIZE = 512
PREEMPHASIS = 0.97
POVEY_EXPONENT = 0.85
LOW_FREQ_HZ = 20.0
# A negative high frequency counts down from the Nyquist frequency.
HIGH_FREQ_HZ = -400.0


def mel_scale(hz: float) -> float:
    return 1127.0 * np.log(1 + hz / 700)


def mel_banks() -> torch.Tensor:
    """[FFT_SIZE // 2 + 1, NUM_MEL_BINS] triangular mel filters, torchaudio's Kaldi layout as lhotse builds it."""
    num_fft_bins = FFT_SIZE / 2
    nyquist = 0.5 * SAMPLE_RATE_HZ
    high_freq = HIGH_FREQ_HZ + nyquist
    fft_bin_width = SAMPLE_RATE_HZ / FFT_SIZE
    mel_low_freq = mel_scale(LOW_FREQ_HZ)
    mel_high_freq = mel_scale(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_MEL_BINS + 1)
    bin_index = torch.arange(NUM_MEL_BINS).unsqueeze(1)
    left_mel = mel_low_freq + bin_index * mel_freq_delta
    center_mel = mel_low_freq + (bin_index + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_index + 2.0) * mel_freq_delta
    # Note (Junnan Li): lhotse takes this log with numpy in float32 on the torch values; kept so the banks match it bit for bit.
    mel = 1127.0 * torch.from_numpy(
        np.log((1 + fft_bin_width * torch.arange(num_fft_bins) / 700).numpy())
    )
    mel = mel.unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
    return F.pad(bins, (0, 1), mode="constant", value=0).T


def tail_start(first_frame: int) -> int:
    """The first sample frame first_frame reads, past the start of the waveform."""
    return first_frame * HOP_SAMPLES - (FRAME_SAMPLES - HOP_SAMPLES) // 2


class LogMelFilterbank:
    """Waveform [samples] at 16 kHz → features [frames, NUM_MEL_BINS], frames = round(samples / HOP_SAMPLES)."""

    def __init__(self) -> None:
        self.window = torch.hann_window(FRAME_SAMPLES, periodic=False).pow(
            POVEY_EXPONENT
        )
        self.banks = mel_banks()
        self.eps = torch.tensor(torch.finfo(torch.float).eps)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.many(waveform.unsqueeze(0))[0]

    def many(self, samples: torch.Tensor, first_frame: int = 0) -> torch.Tensor:
        """Waveforms of one length [batch, samples] on any device → features [batch, frames, NUM_MEL_BINS].

        With a first_frame, samples are the waveforms from
        tail_start(first_frame) on, and the features are those of the whole
        waveforms from that frame on.
        """
        batch = samples.shape[0]
        skipped = 0 if first_frame == 0 else tail_start(first_frame)
        num_samples = samples.size(-1) + skipped
        num_frames = (num_samples + HOP_SAMPLES // 2) // HOP_SAMPLES - first_frame
        pad = (num_frames + first_frame - 1) * HOP_SAMPLES + FRAME_SAMPLES - num_samples
        pad_left = (FRAME_SAMPLES - HOP_SAMPLES) // 2
        samples = torch.cat(
            (
                (
                    torch.flip(samples[:, :pad_left], (1,))
                    if first_frame == 0
                    else samples[:, :0]
                ),
                samples,
                torch.flip(samples[:, -(pad - pad_left) :], (1,)),
            ),
            dim=1,
        )
        frames = samples.as_strided(
            [batch, num_frames, FRAME_SAMPLES],
            (samples.stride(0), HOP_SAMPLES * samples.stride(1), samples.stride(1)),
        )
        frames = frames - torch.mean(frames, dim=2, keepdim=True)
        shifted = F.pad(frames, (1, 0), mode="replicate")
        frames = frames - PREEMPHASIS * shifted[:, :, :-1]
        frames = frames * self.window.to(frames.device)
        frames = F.pad(
            frames.unsqueeze(1),
            [0, FFT_SIZE - FRAME_SAMPLES],
            mode="constant",
            value=0.0,
        ).squeeze(1)
        power = torch.fft.rfft(frames, dim=-1).abs() ** 2
        return torch.max(
            torch.matmul(power, self.banks.to(power.device)), self.eps.to(power.device)
        ).log()


__all__ = ["NUM_MEL_BINS", "LogMelFilterbank", "tail_start"]
