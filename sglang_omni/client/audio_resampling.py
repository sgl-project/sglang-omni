# SPDX-License-Identifier: Apache-2.0
"""High-quality complete and streaming audio resampling."""

from __future__ import annotations

import numpy as np

FILTER_TAPS_PER_PHASE = 20
FILTER_CUTOFF_RATIO = 0.95
KAISER_BETA = 5.0
# note (Codex): Filter shape is guarded by the alias-attenuation test.


class StreamingAudioResampler:
    """Stateful integer-ratio downsampler for streaming mono audio."""

    def __init__(
        self,
        source_sample_rate_hz: int,
        target_sample_rate_hz: int,
    ) -> None:
        if source_sample_rate_hz <= 0 or target_sample_rate_hz <= 0:
            raise ValueError("Audio sample rates must be positive")
        if (
            source_sample_rate_hz < target_sample_rate_hz
            or source_sample_rate_hz % target_sample_rate_hz != 0
        ):
            raise ValueError(
                "Streaming audio resampling requires an integer downsampling "
                f"ratio, got {source_sample_rate_hz} Hz to "
                f"{target_sample_rate_hz} Hz"
            )
        self.source_sample_rate_hz: int = source_sample_rate_hz
        self.target_sample_rate_hz: int = target_sample_rate_hz
        self.downsampling_ratio: int = source_sample_rate_hz // target_sample_rate_hz
        self.audio_buffer: np.ndarray = np.empty((0,), dtype=np.float32)
        self.audio_buffer_start_index: int = 0
        self.total_sample_count: int = 0
        self.next_center_index: int = 0
        self.filter_taps: np.ndarray

        if self.downsampling_ratio == 1:
            self.filter_taps = np.ones((1,), dtype=np.float32)
        else:
            tap_count = FILTER_TAPS_PER_PHASE * self.downsampling_ratio + 1
            half_width = tap_count // 2
            offsets = np.arange(tap_count, dtype=np.float64) - half_width
            cutoff_ratio = FILTER_CUTOFF_RATIO / self.downsampling_ratio
            filter_taps = (
                cutoff_ratio
                * np.sinc(cutoff_ratio * offsets)
                * np.kaiser(tap_count, KAISER_BETA)
            )
            self.filter_taps = (filter_taps / filter_taps.sum()).astype(np.float32)

    def process(self, audio: np.ndarray, *, is_final: bool = False) -> np.ndarray:
        audio_chunk = np.asarray(audio, dtype=np.float32)
        if audio_chunk.ndim != 1:
            raise ValueError(
                "Streaming audio resampling only supports mono audio, "
                f"got shape {audio_chunk.shape}"
            )
        if self.downsampling_ratio == 1:
            return audio_chunk

        if audio_chunk.size:
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk))
            self.total_sample_count += int(audio_chunk.size)

        half_width = self.filter_taps.size // 2
        max_center_index = (
            self.total_sample_count - 1
            if is_final
            else self.total_sample_count - 1 - half_width
        )
        if self.next_center_index > max_center_index:
            return np.empty((0,), dtype=np.float32)

        center_indexes = np.arange(
            self.next_center_index,
            max_center_index + 1,
            self.downsampling_ratio,
            dtype=np.int64,
        )
        first_center_index = int(center_indexes[0])
        last_center_index = int(center_indexes[-1])
        window_start_index = first_center_index - half_width
        window_end_index = last_center_index + half_width
        filter_segment = np.zeros(
            (window_end_index - window_start_index + 1,), dtype=np.float32
        )

        copy_start_index = max(window_start_index, self.audio_buffer_start_index, 0)
        copy_end_index = min(
            window_end_index + 1,
            self.audio_buffer_start_index + self.audio_buffer.size,
        )
        if copy_end_index > copy_start_index:
            destination_start_index = copy_start_index - window_start_index
            source_start_index = copy_start_index - self.audio_buffer_start_index
            copy_sample_count = copy_end_index - copy_start_index
            filter_segment[
                destination_start_index : destination_start_index + copy_sample_count
            ] = self.audio_buffer[
                source_start_index : source_start_index + copy_sample_count
            ]

        sample_windows = np.lib.stride_tricks.sliding_window_view(
            filter_segment, self.filter_taps.size
        )[:: self.downsampling_ratio]
        output_audio = sample_windows @ self.filter_taps[::-1]
        self.next_center_index = last_center_index + self.downsampling_ratio

        keep_from_index = max(0, self.next_center_index - half_width)
        drop_sample_count = min(
            max(keep_from_index - self.audio_buffer_start_index, 0),
            self.audio_buffer.size,
        )
        if drop_sample_count:
            self.audio_buffer = self.audio_buffer[drop_sample_count:]
            self.audio_buffer_start_index += int(drop_sample_count)
        return np.asarray(output_audio, dtype=np.float32)


def resample_audio(
    audio: np.ndarray,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> np.ndarray:
    """Resample complete channel-first audio with the streaming filter."""

    if source_sample_rate_hz == target_sample_rate_hz:
        return audio.astype(np.float32, copy=False)
    elif audio.ndim == 1:
        return StreamingAudioResampler(
            source_sample_rate_hz, target_sample_rate_hz
        ).process(audio, is_final=True)
    else:
        audio_channels = audio.reshape(-1, audio.shape[-1])
        resampled_channels = np.stack(
            [
                StreamingAudioResampler(
                    source_sample_rate_hz, target_sample_rate_hz
                ).process(channel, is_final=True)
                for channel in audio_channels
            ]
        )
        return resampled_channels.reshape(
            audio.shape[:-1] + (resampled_channels.shape[-1],)
        )
