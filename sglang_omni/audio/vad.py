from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)

# silero-vad operates on 512-sample windows @ 16 kHz (32 ms each).
VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16000


@dataclass
class VADConfig:
    """Mirrors OpenAI Realtime turn_detection (server_vad mode)."""

    # Probs greater than threshold are considered speech
    threshold: float = 0.5
    # Prefix padding in milliseconds
    prefix_padding_ms: int = 300
    # Silence duration in milliseconds
    silence_duration_ms: int = 500
    # Voices this many dB below the last utterance start no speech; None turns this off
    background_margin_db: float | None = None


class VADEvent:
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"


@dataclass
class Emit:
    event_type: str
    sample_offset: int


class SpeechProbabilityModel(Protocol):
    def predict(self, frame: np.ndarray, sample_rate: int) -> float: ...

    def reset(self) -> None: ...


class SileroSpeechModel:
    """Per-frame speech probability. Holds no turn state, only the RNN's."""

    def __init__(self) -> None:
        # note (ratish): importing silero_vad sets torch to one intra-op thread for
        # the whole process, so only the process that loads the model imports it.
        import silero_vad

        self.model = silero_vad.load_silero_vad(onnx=True)

    def predict(self, frame: np.ndarray, sample_rate: int) -> float:
        with torch.inference_mode():
            tensor = torch.from_numpy(frame).unsqueeze(0)
            return float(self.model(tensor, sample_rate).item())

    def reset(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()  # type: ignore[union-attr]
        else:
            pass


@dataclass(frozen=True)
class ScoredFrame:
    start_sample: int
    probability: float


@dataclass(frozen=True)
class Boundary:
    started: bool
    # Onset: the speech frame minus prefix padding, which may reach before the
    # audio the caller still holds. Offset: the end of the last speech frame.
    sample: int


class StatelessVAD:
    """This is VAD that hands over "is a turn open?" to the caller, "Stateless" is about turn state only.

    Note(Jeffro): Only realtime transcription uses this for now. That session already keeps
    its own record of whether a turn is open (active_segment) and also relies
    on the VAD to cut turns, so with a self-contained VAD there are two
    records to keep in sync. They drift as soon as the session fails to act on
    an event the VAD has already counted as delivered.
    """

    def __init__(
        self,
        config: VADConfig | None = None,
        model: SpeechProbabilityModel | None = None,
    ) -> None:
        self.config = config or VADConfig()
        self.model = model or SileroSpeechModel()
        self.prefix_padding_samples = (
            self.config.prefix_padding_ms * VAD_SAMPLE_RATE // 1000
        )
        self.silence_samples = self.config.silence_duration_ms * VAD_SAMPLE_RATE // 1000
        self.leftover_pcm = bytearray()
        self.silence_run_samples = 0
        self.last_speech_sample = 0

    async def process(
        self,
        pcm_bytes: bytes,
        start_sample: int,
        *,
        in_speech: Callable[[], bool],
        on_started: Callable[[int], Awaitable[None]],
        on_stopped: Callable[[int], Awaitable[None]],
    ) -> None:
        frames = await asyncio.to_thread(self.score, pcm_bytes, start_sample)
        for frame in frames:
            boundary = self.step(frame, in_speech=in_speech())
            if boundary is None:
                continue
            else:
                pass
            if boundary.started:
                await on_started(boundary.sample)
            else:
                await on_stopped(boundary.sample)

    def score(self, pcm_bytes: bytes, start_sample: int) -> list[ScoredFrame]:
        frame_start = start_sample - len(self.leftover_pcm) // 2
        self.leftover_pcm.extend(pcm_bytes)
        frames: list[ScoredFrame] = []
        while len(self.leftover_pcm) >= VAD_FRAME_SAMPLES * 2:
            frame_bytes = bytes(self.leftover_pcm[: VAD_FRAME_SAMPLES * 2])
            del self.leftover_pcm[: VAD_FRAME_SAMPLES * 2]
            frame = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0
            frames.append(
                ScoredFrame(frame_start, self.model.predict(frame, VAD_SAMPLE_RATE))
            )
            frame_start += VAD_FRAME_SAMPLES
        return frames

    def step(self, frame: ScoredFrame, *, in_speech: bool) -> Boundary | None:
        speech = frame.probability >= self.config.threshold
        if speech:
            self.silence_run_samples = 0
            self.last_speech_sample = frame.start_sample + VAD_FRAME_SAMPLES
        else:
            self.silence_run_samples += VAD_FRAME_SAMPLES

        if speech and not in_speech:
            return Boundary(
                started=True, sample=frame.start_sample - self.prefix_padding_samples
            )
        else:
            pass
        if (
            not speech
            and in_speech
            and self.silence_run_samples >= self.silence_samples
        ):
            return Boundary(started=False, sample=self.last_speech_sample)
        else:
            pass
        return None

    def reset(self) -> None:
        self.leftover_pcm.clear()
        self.silence_run_samples = 0
        self.last_speech_sample = 0
        self.model.reset()


class StreamingVAD:
    """Per-session frame-by-frame VAD state machine.

    Callers feed raw PCM16 LE mono @ 16 kHz via :meth:`process`. The
    wrapper buffers up to one frame's worth of leftover bytes between
    calls so the caller doesn't have to align to 32 ms.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        # note (ratish): imported here, not at module scope, for the thread side
        # effect described in SileroSpeechModel.
        import silero_vad

        self.vad_model = silero_vad.load_silero_vad(onnx=True)
        self.leftover_pcm = bytearray()
        self.samples_consumed = 0
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = 0
        self.run_start_offset: int | None = None
        self.utterance_power = 0.0
        self.utterance_frames = 0
        self.reference_db: float | None = None

    def is_background(self, power: float) -> bool:
        margin = self.config.background_margin_db
        if margin is None or self.reference_db is None:
            return False
        else:
            return power_db(power) < self.reference_db - margin

    def process(self, pcm_bytes: bytes) -> list[Emit]:
        """Feed PCM16 LE mono @ 16 kHz; return any state transitions."""
        if not pcm_bytes:
            return []
        else:
            pass
        self.leftover_pcm.extend(pcm_bytes)
        emits: list[Emit] = []

        while len(self.leftover_pcm) >= VAD_FRAME_SAMPLES * 2:
            frame_bytes = bytes(self.leftover_pcm[: VAD_FRAME_SAMPLES * 2])
            del self.leftover_pcm[: VAD_FRAME_SAMPLES * 2]
            frame = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0

            prob = self.infer(frame)
            power = float(np.mean(np.square(frame)))
            self.samples_consumed += VAD_FRAME_SAMPLES
            speech = prob >= self.config.threshold

            if speech:
                self.silence_run_samples = 0
                self.last_speech_offset = self.samples_consumed
                if self.run_start_offset is None:
                    self.run_start_offset = self.samples_consumed - VAD_FRAME_SAMPLES
                else:
                    pass
                if self.is_speech:
                    pass
                elif self.is_background(power):
                    pass
                else:
                    self.is_speech = True
                    self.utterance_power = 0.0
                    self.utterance_frames = 0
                    # OpenAI's contract: speech_started reports the start
                    # offset *minus* prefix_padding so the caller includes
                    # a leading prefix in the committed audio.
                    pad = self.config.prefix_padding_ms * VAD_SAMPLE_RATE // 1000
                    # note (wenyao): An interruption can begin below the margin, so speech starts at the run's first frame, not at the frame that crossed it.
                    started_at = max(0, self.run_start_offset - pad)
                    emits.append(
                        Emit(
                            event_type=VADEvent.SPEECH_STARTED, sample_offset=started_at
                        )
                    )
                if self.is_speech:
                    self.utterance_power += power
                    self.utterance_frames += 1
                else:
                    pass
            else:
                self.run_start_offset = None
                self.silence_run_samples += VAD_FRAME_SAMPLES
                if self.is_speech:
                    silence_threshold = (
                        self.config.silence_duration_ms * VAD_SAMPLE_RATE // 1000
                    )
                    if self.silence_run_samples >= silence_threshold:
                        self.is_speech = False
                        self.reference_db = power_db(
                            self.utterance_power / self.utterance_frames
                        )
                        emits.append(
                            Emit(
                                event_type=VADEvent.SPEECH_STOPPED,
                                sample_offset=self.last_speech_offset,
                            )
                        )
                    else:
                        pass
                else:
                    pass

        return emits

    def infer(self, frame: np.ndarray) -> float:
        with torch.inference_mode():
            tensor = torch.from_numpy(frame).unsqueeze(0)
            prob = self.vad_model(tensor, VAD_SAMPLE_RATE).item()
        return float(prob)

    def reset(self) -> None:
        """Keeps the last utterance's level, which describes the speaker rather than the stream."""
        self.leftover_pcm.clear()
        self.samples_consumed = 0
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = 0
        self.run_start_offset = None
        if hasattr(self.vad_model, "reset_states"):
            self.vad_model.reset_states()  # type: ignore[union-attr]
        else:
            pass


def power_db(power: float) -> float:
    return 10 * float(np.log10(power + 1e-12))


def offsets_to_ms(samples: int) -> int:
    return samples * 1000 // VAD_SAMPLE_RATE
