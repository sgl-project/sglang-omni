# SPDX-License-Identifier: Apache-2.0
"""Report-only diagnostics for decoded audio samples and uncompressed WAV files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
from numpy.typing import NDArray

PCM_BITS = {"PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
WAV_FORMATS = {"WAV", "WAVEX", "RF64"}
DEFAULT_SILENCE_WINDOW_MS = 20.0
DEFAULT_SILENCE_THRESHOLD_DBFS = -60.0


@dataclass(frozen=True, kw_only=True)
class ChannelSignalMetrics:
    sample_peak: float
    dc_offset: float
    full_scale_sample_count: int


@dataclass(frozen=True, kw_only=True)
class SilenceRegion:
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float


@dataclass(frozen=True, kw_only=True)
class AudioSignalMetrics:
    sample_rate_hz: int
    frame_count: int
    channel_count: int
    duration_s: float
    channels: list[ChannelSignalMetrics]
    positive_full_scale: float
    silence_window_frames: int
    window_count: int
    silent_window_count: int
    silent_window_ratio: float
    silent_duration_s: float
    silence_regions: list[SilenceRegion]


@dataclass(frozen=True, kw_only=True)
class AudioSignalReport:
    path: str
    format: str
    subtype: str
    metrics: AudioSignalMetrics


def validate_silence_settings(
    silence_window_ms: float, silence_threshold_dbfs: float
) -> None:
    """Reject settings that cannot describe finite RMS windows."""
    if not math.isfinite(silence_window_ms) or silence_window_ms <= 0:
        raise ValueError("silence_window_ms must be finite and positive")
    elif not math.isfinite(silence_threshold_dbfs) or silence_threshold_dbfs > 0:
        raise ValueError("silence_threshold_dbfs must be finite and at most 0")
    else:
        pass


def compute_signal_metrics(
    samples: NDArray[np.float64],
    sample_rate_hz: int,
    *,
    positive_full_scale: float = 1.0,
    silence_window_ms: float = DEFAULT_SILENCE_WINDOW_MS,
    silence_threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS,
) -> AudioSignalMetrics:
    """Measure native-rate samples shaped as frames by channels, without downmixing."""
    validate_silence_settings(silence_window_ms, silence_threshold_dbfs)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    elif samples.ndim != 2 or not all(samples.shape):
        raise ValueError("samples must contain at least one frame and one channel")
    elif not np.isfinite(samples).all():
        raise ValueError("audio samples must be finite")
    elif not math.isfinite(positive_full_scale) or not 0 < positive_full_scale <= 1:
        raise ValueError("positive_full_scale must be finite and in (0, 1]")
    else:
        pass

    window_frames_float = sample_rate_hz * (silence_window_ms / 1000)
    if not math.isfinite(window_frames_float):
        raise ValueError("silence window exceeds the supported frame range")
    else:
        window_frames = max(1, round(window_frames_float))

    frame_count, channel_count = samples.shape
    channels = []
    for channel_index in range(channel_count):
        channel = samples[:, channel_index]
        sample_peak = float(np.max(np.abs(channel)))
        if sample_peak == 0:
            dc_offset = 0.0
        else:
            dc_offset = float(np.mean(channel / sample_peak) * sample_peak)
        channels.append(
            ChannelSignalMetrics(
                sample_peak=sample_peak,
                dc_offset=dc_offset,
                full_scale_sample_count=int(
                    np.count_nonzero((channel <= -1) | (channel >= positive_full_scale))
                ),
            )
        )

    silence_amplitude = 10 ** (silence_threshold_dbfs / 20)
    silent_window_count = 0
    silent_frames = 0
    silent_intervals: list[tuple[int, int]] = []
    for start_frame in range(0, frame_count, window_frames):
        end_frame = min(start_frame + window_frames, frame_count)
        window = samples[start_frame:end_frame]
        channel_peaks = np.max(np.abs(window), axis=0)
        # note (PansaLegrand): Scale before squaring to retain finite FLOAT/DOUBLE excursions.
        scaled = np.divide(
            window,
            channel_peaks,
            out=np.zeros_like(window),
            where=channel_peaks != 0,
        )
        channel_rms = np.sqrt(np.mean(scaled * scaled, axis=0)) * channel_peaks
        if np.all(channel_rms <= silence_amplitude):
            silent_window_count += 1
            silent_frames += end_frame - start_frame
            if silent_intervals and silent_intervals[-1][1] == start_frame:
                silent_intervals[-1] = (silent_intervals[-1][0], end_frame)
            else:
                silent_intervals.append((start_frame, end_frame))
        else:
            pass

    window_count = math.ceil(frame_count / window_frames)
    return AudioSignalMetrics(
        sample_rate_hz=sample_rate_hz,
        frame_count=frame_count,
        channel_count=channel_count,
        duration_s=frame_count / sample_rate_hz,
        channels=channels,
        positive_full_scale=positive_full_scale,
        silence_window_frames=window_frames,
        window_count=window_count,
        silent_window_count=silent_window_count,
        silent_window_ratio=silent_window_count / window_count,
        silent_duration_s=silent_frames / sample_rate_hz,
        silence_regions=[
            SilenceRegion(
                start_frame=start_frame,
                end_frame=end_frame,
                start_s=start_frame / sample_rate_hz,
                end_s=end_frame / sample_rate_hz,
            )
            for start_frame, end_frame in silent_intervals
        ],
    )


def analyze_audio_file(
    path: Path,
    *,
    silence_window_ms: float = DEFAULT_SILENCE_WINDOW_MS,
    silence_threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS,
) -> AudioSignalReport:
    """Decode an uncompressed WAV at its original rate and amplitude."""
    with soundfile.SoundFile(path) as audio_file:
        if audio_file.format not in WAV_FORMATS:
            raise ValueError(f"expected a WAV file, found {audio_file.format}")
        elif audio_file.subtype in PCM_BITS:
            positive_full_scale = 1.0 - 2.0 ** (1 - PCM_BITS[audio_file.subtype])
        elif audio_file.subtype in {"FLOAT", "DOUBLE"}:
            positive_full_scale = 1.0
        else:
            raise ValueError(f"unsupported WAV subtype: {audio_file.subtype}")
        samples = audio_file.read(dtype="float64", always_2d=True)
        return AudioSignalReport(
            path=str(path),
            format=audio_file.format,
            subtype=audio_file.subtype,
            metrics=compute_signal_metrics(
                samples,
                audio_file.samplerate,
                positive_full_scale=positive_full_scale,
                silence_window_ms=silence_window_ms,
                silence_threshold_dbfs=silence_threshold_dbfs,
            ),
        )
