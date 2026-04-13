# SPDX-License-Identifier: Apache-2.0
"""WebRTC VAD wrapper for the realtime prototype."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil

import numpy as np

try:
    import webrtcvad
except ImportError:  # pragma: no cover - surfaced at runtime
    webrtcvad = None


@dataclass
class VadConfig:
    sample_rate: int = 16000
    aggressiveness: int = 3
    frame_duration_ms: int = 20
    min_speech_s: float = 0.25
    min_silence_s: float = 0.60
    preroll_s: float = 0.18
    # Legacy fields kept for request compatibility; WebRTC VAD does not use them.
    start_threshold: float = 0.020
    stop_threshold: float = 0.012
    start_margin: float = 0.020
    stop_margin: float = 0.005
    bootstrap_s: float = 0.50
    noise_floor_alpha: float = 0.20


@dataclass
class VadEvent:
    speech_started: bool = False
    speech_stopped: bool = False


class EnergyVad:
    """Stateful wrapper around WebRTC VAD with start/stop hysteresis."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        if webrtcvad is None:
            raise RuntimeError(
                "Realtime VAD now depends on webrtcvad. "
                "Install the project with the realtime extra."
            )

        if self.config.sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError(
                f"Unsupported VAD sample rate {self.config.sample_rate}; "
                "expected one of 8000, 16000, 32000, 48000."
            )
        if self.config.frame_duration_ms not in {10, 20, 30}:
            raise ValueError(
                f"Unsupported VAD frame duration {self.config.frame_duration_ms} ms; "
                "expected one of 10, 20, 30."
            )

        self._vad = webrtcvad.Vad(int(np.clip(self.config.aggressiveness, 0, 3)))
        self._frame_samples = (
            self.config.sample_rate * self.config.frame_duration_ms
        ) // 1000
        self._frame_duration_s = self.config.frame_duration_ms / 1000.0
        self._start_window_frames = max(
            1,
            int(round(self.config.min_speech_s / self._frame_duration_s)),
        )
        self._stop_window_frames = max(
            1,
            int(round(self.config.min_silence_s / self._frame_duration_s)),
        )
        self._start_required_frames = max(
            1,
            ceil(self._start_window_frames * 0.6),
        )
        self._stop_required_unvoiced_frames = max(
            1,
            ceil(self._stop_window_frames * 0.5),
        )

        self.speaking = False
        self._frame_tail = np.zeros(0, dtype=np.int16)
        self._recent_votes: deque[bool] = deque()
        self._last_frame_count = 0
        self._last_voiced_frame_count = 0

    @staticmethod
    def measure_level(audio: np.ndarray) -> float:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return 0.0
        centered = audio - float(np.mean(audio))
        return float(np.sqrt(np.mean(np.square(centered))))

    @property
    def noise_floor(self) -> float:
        return 0.0

    def effective_start_threshold(self) -> float:
        return 0.0

    def effective_stop_threshold(self) -> float:
        return 0.0

    @property
    def last_frame_count(self) -> int:
        return self._last_frame_count

    @property
    def last_voiced_frame_count(self) -> int:
        return self._last_voiced_frame_count

    @property
    def last_speech_ratio(self) -> float:
        if self._last_frame_count <= 0:
            return 0.0
        return float(self._last_voiced_frame_count / self._last_frame_count)

    def _detect_frame(self, pcm_frame: np.ndarray) -> bool:
        return bool(self._vad.is_speech(pcm_frame.tobytes(), self.config.sample_rate))

    def _append_vote(self, vote: bool, *, speaking: bool) -> None:
        window = self._stop_window_frames if speaking else self._start_window_frames
        self._recent_votes.append(vote)
        while len(self._recent_votes) > window:
            self._recent_votes.popleft()

    def reset(self) -> None:
        self.speaking = False
        self._frame_tail = np.zeros(0, dtype=np.int16)
        self._recent_votes.clear()
        self._last_frame_count = 0
        self._last_voiced_frame_count = 0

    def process(self, audio: np.ndarray) -> VadEvent:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            self._last_frame_count = 0
            self._last_voiced_frame_count = 0
            return VadEvent()

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16, copy=False)
        if self._frame_tail.size:
            pcm = np.concatenate([self._frame_tail, pcm])

        total_frames = int(pcm.size // self._frame_samples)
        if total_frames <= 0:
            self._frame_tail = pcm
            self._last_frame_count = 0
            self._last_voiced_frame_count = 0
            return VadEvent()

        event = VadEvent()
        voiced_frames = 0

        for index in range(total_frames):
            start = index * self._frame_samples
            frame = pcm[start : start + self._frame_samples]
            is_voiced = self._detect_frame(frame)
            voiced_frames += int(is_voiced)
            self._append_vote(is_voiced, speaking=self.speaking)

            if not self.speaking:
                if (
                    len(self._recent_votes) >= self._start_window_frames
                    and sum(self._recent_votes) >= self._start_required_frames
                ):
                    self.speaking = True
                    self._recent_votes.clear()
                    event.speech_started = True
                    continue

            if self.speaking:
                unvoiced_frames = len(self._recent_votes) - sum(self._recent_votes)
                if (
                    len(self._recent_votes) >= self._stop_window_frames
                    and unvoiced_frames >= self._stop_required_unvoiced_frames
                ):
                    self.speaking = False
                    self._recent_votes.clear()
                    event.speech_stopped = True

        consumed = total_frames * self._frame_samples
        self._frame_tail = pcm[consumed:]
        self._last_frame_count = total_frames
        self._last_voiced_frame_count = voiced_frames
        return event
