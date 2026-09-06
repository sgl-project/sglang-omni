# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from silero_vad import load_silero_vad

from .vad import VAD_FRAME_SAMPLES, VAD_SAMPLE_RATE, Emit, VADEvent

logger = logging.getLogger(__name__)


class SemanticEOUModel(Protocol):
    def predict(self, audio: np.ndarray, sample_rate: int) -> float: ...


class SpeechProbabilityModel(Protocol):
    def predict(self, frame: np.ndarray, sample_rate: int) -> float: ...

    def reset(self) -> None: ...


class SileroSpeechModel:
    def __init__(self) -> None:
        self.model = load_silero_vad(onnx=True)

    def predict(self, frame: np.ndarray, sample_rate: int) -> float:
        with torch.inference_mode():
            tensor = torch.from_numpy(frame).unsqueeze(0)
            return float(self.model(tensor, sample_rate).item())

    def reset(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()  # type: ignore[union-attr]


@dataclass(frozen=True)
class SemanticVADConfig:
    eagerness: str = "medium"
    speech_threshold: float = 0.5
    prefix_padding_ms: int = 300
    candidate_pause_ms: int = 160
    max_utterance_seconds: int = 8
    confidence_threshold: float = 0.88
    immediate_confidence_threshold: float = 0.97
    confidence_silence_ms: int = 640
    immediate_silence_ms: int = 250
    max_pause_ms: int = 2000
    fallback_silence_ms: int = 640

    @classmethod
    def from_eagerness(cls, eagerness: str) -> SemanticVADConfig:
        presets = {
            "low": {
                "confidence_threshold": 0.92,
                "immediate_confidence_threshold": 0.99,
                "confidence_silence_ms": 1200,
                "immediate_silence_ms": 500,
                "max_pause_ms": 3000,
                "fallback_silence_ms": 800,
            },
            "medium": {},
            "high": {
                "confidence_threshold": 0.80,
                "immediate_confidence_threshold": 0.93,
                "confidence_silence_ms": 320,
                "immediate_silence_ms": 200,
                "max_pause_ms": 1200,
                "fallback_silence_ms": 512,
            },
        }
        if eagerness not in presets:
            raise ValueError(f"Unsupported semantic VAD eagerness: {eagerness}")
        return cls(eagerness=eagerness, **presets[eagerness])


class SemanticTurnDetector:
    """Silero speech detection with Smart Turn end-of-utterance scoring."""

    def __init__(
        self,
        eou_model: SemanticEOUModel,
        config: SemanticVADConfig | None = None,
        *,
        speech_model: SpeechProbabilityModel | None = None,
    ) -> None:
        self.config = config or SemanticVADConfig()
        self.eou_model = eou_model
        self.speech_model = speech_model or SileroSpeechModel()
        self.leftover_pcm = bytearray()
        self.samples_consumed = 0
        self.is_speech = False
        self.last_speech_offset = 0
        self.silence_run_samples = 0
        self.candidate_probability: float | None = None
        self.utterance_audio = bytearray()
        self._eou_broken = False

    def process(self, pcm_bytes: bytes) -> list[Emit]:
        if not pcm_bytes:
            return []
        self.leftover_pcm.extend(pcm_bytes)
        emits: list[Emit] = []
        frame_bytes_count = VAD_FRAME_SAMPLES * 2

        while len(self.leftover_pcm) >= frame_bytes_count:
            frame_bytes = bytes(self.leftover_pcm[:frame_bytes_count])
            del self.leftover_pcm[:frame_bytes_count]
            frame_start = self.samples_consumed
            self.samples_consumed += VAD_FRAME_SAMPLES
            frame = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0
            speech = (
                self.speech_model.predict(frame, VAD_SAMPLE_RATE)
                >= self.config.speech_threshold
            )

            if speech:
                self.last_speech_offset = self.samples_consumed
                self.silence_run_samples = 0
                self.candidate_probability = None
                if not self.is_speech:
                    self.is_speech = True
                    self.utterance_audio.clear()
                    padding = self.config.prefix_padding_ms * VAD_SAMPLE_RATE // 1000
                    emits.append(
                        Emit(
                            VADEvent.SPEECH_STARTED,
                            max(0, frame_start - padding),
                        )
                    )
                self.utterance_audio.extend(frame_bytes)
                continue

            if not self.is_speech:
                continue

            self.utterance_audio.extend(frame_bytes)
            self.silence_run_samples += VAD_FRAME_SAMPLES
            silence_ms = self.silence_run_samples * 1000 // VAD_SAMPLE_RATE

            if self._eou_broken:
                if silence_ms >= self.config.fallback_silence_ms:
                    emits.append(self._end_turn())
                continue

            if (
                self.candidate_probability is None
                and silence_ms >= self.config.candidate_pause_ms
            ):
                try:
                    self.candidate_probability = self._predict_eou()
                except Exception:
                    logger.warning(
                        "Smart Turn inference failed; using fixed-silence fallback",
                        exc_info=True,
                    )
                    self._eou_broken = True
                    if silence_ms >= self.config.fallback_silence_ms:
                        emits.append(self._end_turn())
                    continue

            required_silence_ms = self.config.max_pause_ms
            if (
                self.candidate_probability is not None
                and self.candidate_probability
                >= self.config.immediate_confidence_threshold
            ):
                required_silence_ms = max(
                    self.config.candidate_pause_ms,
                    self.config.immediate_silence_ms,
                )
            elif (
                self.candidate_probability is not None
                and self.candidate_probability >= self.config.confidence_threshold
            ):
                required_silence_ms = max(
                    self.config.candidate_pause_ms,
                    self.config.confidence_silence_ms,
                )
            if silence_ms >= required_silence_ms:
                emits.append(self._end_turn())

        return emits

    def _predict_eou(self) -> float:
        audio = np.frombuffer(self.utterance_audio, dtype="<i2").astype(np.float32)
        audio /= 32768.0
        max_samples = self.config.max_utterance_seconds * VAD_SAMPLE_RATE
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        return float(self.eou_model.predict(audio, VAD_SAMPLE_RATE))

    def _end_turn(self) -> Emit:
        self.is_speech = False
        self.silence_run_samples = 0
        self.candidate_probability = None
        self.utterance_audio.clear()
        return Emit(VADEvent.SPEECH_STOPPED, self.last_speech_offset)

    def reset(self) -> None:
        self.leftover_pcm.clear()
        self.samples_consumed = 0
        self.is_speech = False
        self.last_speech_offset = 0
        self.silence_run_samples = 0
        self.candidate_probability = None
        self.utterance_audio.clear()
        self._eou_broken = False
        self.speech_model.reset()
