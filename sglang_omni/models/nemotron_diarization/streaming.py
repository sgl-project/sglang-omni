# SPDX-License-Identifier: Apache-2.0
"""Bounded, session-local Sortformer inference from incremental PCM audio."""

from __future__ import annotations

import threading
import time

import numpy as np
import torch

from sglang_omni.client.types import DiarizationResult, DiarizationSegment
from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache

SAMPLE_RATE = 16000
MAX_AUDIO_BYTES = SAMPLE_RATE * 2  # At most one second of PCM16 per message.
SESSION_IDLE_SECONDS = 60


class LiveDiarization:
    """Keep acoustic context and speaker identity without retaining the recording."""

    def __init__(self, model, *, device):
        self.model = model
        self.device = device
        self.audio = np.empty(0, dtype=np.float32)
        self.buffer_start = 0
        self.samples = 0
        self.frame = 0
        self.active = np.zeros(8, dtype=bool)
        # The published low-latency profile: 720 ms chunks, 320 ms lookahead.
        self.state = SpeakerCache(
            torch.empty(0, device=device),
            cache_size=264,
            fifo_size=264,
            update_period=222,
        )
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self.finished = False

    @torch.inference_mode()
    def probabilities(self, audio: np.ndarray, *, final: bool = False):
        if self.finished:
            raise ValueError("Diarization stream has finished")
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono audio")
        self.audio = np.concatenate([self.audio, audio])
        self.samples += len(audio)
        outputs = []
        while self.frame < self.samples // 160:
            available = self.samples // 160 - self.frame
            count = min(72, available)
            right = min(32, available - count)
            # Centered STFT needs future samples as well as encoder lookahead.
            if not final and self.samples < (self.frame + 104) * 160 + 256:
                break
            if self.state.cache.is_cuda:
                # Successive messages can run on different workers. Keep old
                # cache storage alive until this stream finishes reading it.
                stream = torch.cuda.current_stream(self.device)
                for tensor in (
                    self.state.cache,
                    self.state.fifo,
                    self.state.cache_preds,
                ):
                    tensor.record_stream(stream)
            offset = (self.frame * 160 - self.buffer_start) // 160
            end = (offset + count + right) * 160 + 256
            signal = torch.as_tensor(
                self.audio[:end], device=self.device, dtype=torch.float32
            )[None, :]
            features, _ = self.model.preprocessor(signal)
            features = features[:, :, offset : offset + count + right]
            outputs.append(
                self.model.forward_chunk(features, self.state, right)[:, :count]
            )
            self.frame += count
            # Two hop lengths preserve the STFT window and preemphasis history.
            keep_from = max(0, self.frame * 160 - 320)
            self.audio = self.audio[keep_from - self.buffer_start :].copy()
            self.buffer_start = keep_from
        self.finished = final
        if final:
            self.audio = np.empty(0, dtype=np.float32)
        return (
            torch.cat(outputs, dim=1)
            if outputs
            else torch.empty(1, 0, 8, device=self.device)
        )

    def append(self, pcm: bytes, *, final: bool = False) -> DiarizationResult:
        start_frame = self.frame
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        predictions = self.probabilities(audio, final=final)[0].cpu().numpy()
        segments = []
        for speaker in range(8):
            values = predictions[:, speaker]
            positions = np.arange(1, len(values) + 1)
            last_event = np.maximum.accumulate(np.where(values != 0.5, positions, 0))
            active = np.concatenate(([self.active[speaker]], values > 0.5))[last_event]
            if len(active):
                self.active[speaker] = active[-1]
            changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
            for start, end in zip(
                np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            ):
                segments.append(
                    DiarizationSegment(
                        start=float(start_frame + start) / 100,
                        end=float(start_frame + end) / 100,
                        speaker=f"speaker_{speaker}",
                    )
                )
        segments.sort(key=lambda item: (item.start, item.end, item.speaker))
        return DiarizationResult(
            duration=self.samples / SAMPLE_RATE if final else self.frame / 100,
            segments=segments,
        )


class LiveSessions:
    """Worker-side session admission; the WebSocket owns each session's lifetime."""

    def __init__(self, diarizer, max_sessions: int):
        self.diarizer = diarizer
        self.max_sessions = max_sessions
        self.sessions: dict[str, LiveDiarization] = {}
        self.lock = threading.Lock()

    def compute(self, inputs) -> DiarizationResult:
        session_id = inputs["session_id"]
        operation = inputs["operation"]
        pcm = inputs.get("pcm", b"")
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("Expected at most one second of mono PCM16 audio")
        if operation not in {"open", "append", "finish", "close"}:
            raise ValueError("Unknown diarization stream operation")
        if operation != "append" and pcm:
            raise ValueError("Only append accepts audio")
        with self.lock:
            now = time.monotonic()
            for key, state in list(self.sessions.items()):
                if (
                    not state.lock.locked()
                    and now - state.last_used > SESSION_IDLE_SECONDS
                ):
                    del self.sessions[key]
            if operation == "close":
                self.sessions.pop(session_id, None)
                return DiarizationResult(duration=0, segments=[])
            if operation == "open":
                if session_id in self.sessions:
                    raise ValueError("Diarization session already exists")
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError("Live diarization session limit reached")
                self.sessions[session_id] = LiveDiarization(
                    self.diarizer.model, device=self.diarizer.device
                )
                return DiarizationResult(duration=0, segments=[])
            state = self.sessions.get(session_id)
            if state is None:
                raise ValueError("Diarization session is closed or expired")
            if not state.lock.acquire(blocking=False):
                raise ValueError("Diarization session already has a pending operation")
        try:
            return state.append(pcm, final=operation == "finish")
        except Exception:
            with self.lock:
                self.sessions.pop(session_id, None)
            raise
        finally:
            with self.lock:
                state.last_used = time.monotonic()
                state.lock.release()
                if operation == "finish":
                    self.sessions.pop(session_id, None)
