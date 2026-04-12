# SPDX-License-Identifier: Apache-2.0
"""Lightweight energy-based VAD for the realtime prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VadConfig:
    sample_rate: int = 16000
    start_threshold: float = 0.020
    stop_threshold: float = 0.012
    min_speech_s: float = 0.25
    min_silence_s: float = 0.60
    preroll_s: float = 0.18


@dataclass
class VadEvent:
    speech_started: bool = False
    speech_stopped: bool = False


class EnergyVad:
    """Simple RMS-threshold VAD with start/stop hysteresis."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        self.speaking = False
        self._speech_duration_s = 0.0
        self._silence_duration_s = 0.0

    def process(self, audio: np.ndarray) -> VadEvent:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration_s = audio.shape[0] / float(self.config.sample_rate)
        if duration_s <= 0.0:
            return VadEvent()

        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        event = VadEvent()

        if not self.speaking:
            if rms >= self.config.start_threshold:
                self.speaking = True
                self._speech_duration_s = duration_s
                self._silence_duration_s = 0.0
                event.speech_started = True
            return event

        self._speech_duration_s += duration_s
        if rms < self.config.stop_threshold:
            self._silence_duration_s += duration_s
        else:
            self._silence_duration_s = 0.0

        if (
            self._speech_duration_s >= self.config.min_speech_s
            and self._silence_duration_s >= self.config.min_silence_s
        ):
            self.speaking = False
            self._speech_duration_s = 0.0
            self._silence_duration_s = 0.0
            event.speech_stopped = True

        return event
