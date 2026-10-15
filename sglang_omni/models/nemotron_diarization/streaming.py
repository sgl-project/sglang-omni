# SPDX-License-Identifier: Apache-2.0
"""Bounded, session-local Sortformer inference from incremental PCM audio."""

from __future__ import annotations

import threading
import time
from typing import get_args

import numpy as np
import torch

from sglang_omni.client.types import DiarizationOperation, DiarizationResult
from sglang_omni.models.nemotron_diarization.backend import (
    PROFILES,
    SAMPLE_RATE,
    probabilities_to_segments,
)
from sglang_omni.models.nemotron_diarization.model import NemotronDiarizationModel
from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache

# note (Ryan Stewart): At most one second of PCM16 per message.
MAX_AUDIO_BYTES = SAMPLE_RATE * 2
SESSION_IDLE_SECONDS = 60
OPERATIONS: tuple[DiarizationOperation, ...] = get_args(DiarizationOperation)


class LiveDiarization:
    """Keep acoustic context and speaker identity without retaining the recording."""

    def __init__(
        self, model: NemotronDiarizationModel, *, device: torch.device
    ) -> None:
        self.model = model
        self.device = device
        self.audio = np.empty(0, dtype=np.float32)
        self.buffer_start = 0
        self.samples = 0
        self.frame = 0
        self.active = np.zeros(8, dtype=bool)
        # note (Richard Wang): Live audio uses the published low-latency profile.
        cache_size, fifo_size, chunk_size, right_context, update_period = PROFILES[
            "low_latency"
        ]
        self.chunk_frames, self.right_frames = chunk_size * 8, right_context * 8
        self.state = SpeakerCache(
            torch.empty(0, device=device),
            cache_size=cache_size,
            fifo_size=fifo_size,
            update_period=update_period,
        )
        self.no_frames = torch.empty(1, 0, 8, device=device)
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self.finished = False

    @torch.inference_mode()
    def probabilities(self, audio: np.ndarray, *, final: bool = False) -> torch.Tensor:
        if self.finished:
            raise ValueError("Diarization stream has finished")
        elif audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Expected finite mono audio")
        else:
            pass
        self.audio = np.concatenate([self.audio, audio])
        self.samples += len(audio)
        outputs = []
        while self.frame < self.samples // 160:
            available_frames = self.samples // 160 - self.frame
            window_chunk_frames = min(self.chunk_frames, available_frames)
            window_right_frames = min(
                self.right_frames, available_frames - window_chunk_frames
            )
            # note (Ryan Stewart): Centered STFT needs future samples beyond the encoder lookahead.
            window_frames = self.chunk_frames + self.right_frames
            if not final and self.samples < (self.frame + window_frames) * 160 + 256:
                break
            elif self.state.cache.is_cuda:
                # note (Ryan Stewart): Keep old cache storage alive until this stream has read it.
                stream = torch.cuda.current_stream(self.device)
                for tensor in (
                    self.state.cache,
                    self.state.fifo,
                    self.state.cache_preds,
                ):
                    tensor.record_stream(stream)
            else:
                pass
            offset_frames = (self.frame * 160 - self.buffer_start) // 160
            end_sample = (
                offset_frames + window_chunk_frames + window_right_frames
            ) * 160 + 256
            signal = torch.as_tensor(
                self.audio[:end_sample], device=self.device, dtype=torch.float32
            )[None, :]
            features, _ = self.model.preprocessor(signal)
            features = features[
                :,
                :,
                offset_frames : offset_frames
                + window_chunk_frames
                + window_right_frames,
            ]
            outputs.append(
                self.model.forward_chunk(features, self.state, window_right_frames)[
                    :, :window_chunk_frames
                ]
            )
            self.frame += window_chunk_frames
            # note (Ryan Stewart): Two hop lengths preserve the STFT window and preemphasis history.
            keep_from_sample = max(0, self.frame * 160 - 320)
            self.audio = self.audio[keep_from_sample - self.buffer_start :].copy()
            self.buffer_start = keep_from_sample
        self.finished = final
        self.audio = np.empty(0, dtype=np.float32) if final else self.audio
        return torch.cat(outputs, dim=1) if outputs else self.no_frames

    def append(self, pcm: bytes, *, final: bool = False) -> DiarizationResult:
        start_frame = self.frame
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        predictions = self.probabilities(audio, final=final)[0]
        duration_s = self.samples / SAMPLE_RATE if final else self.frame / 100
        if predictions.shape[0] == 0:
            # note (Richard Wang): Most messages complete no window, so skip decoding.
            return DiarizationResult(duration=duration_s, segments=[])
        else:
            return probabilities_to_segments(
                predictions.cpu().numpy(),
                duration_s=duration_s,
                start_frame=start_frame,
                active=self.active,
            )


class LiveSessions:
    """Worker-side session admission, while the WebSocket owns each session."""

    def __init__(
        self, model: NemotronDiarizationModel, device: torch.device, max_sessions: int
    ) -> None:
        self.model = model
        self.device = device
        self.max_sessions = max_sessions
        self.sessions: dict[str, LiveDiarization] = {}
        self.pending: dict[str, int] = {}
        self.lock = threading.Lock()

    def expire_idle_sessions(self) -> None:
        now = time.monotonic()
        for session_id, state in list(self.sessions.items()):
            if (
                session_id not in self.pending
                and not state.lock.locked()
                and now - state.last_used > SESSION_IDLE_SECONDS
            ):
                self.sessions.pop(session_id)
            else:
                pass

    def reserve(self, session_id: str) -> bool:
        """Pin accepted activity, returning whether it continues an idle session."""
        with self.lock:
            state = self.sessions.get(session_id)
            if (
                state is not None
                and session_id not in self.pending
                and not state.lock.locked()
                and time.monotonic() - state.last_used > SESSION_IDLE_SECONDS
            ):
                self.sessions.pop(session_id)
            else:
                pass
            continues = session_id in self.sessions and session_id not in self.pending
            self.pending[session_id] = self.pending.get(session_id, 0) + 1
            return continues

    def release(self, session_id: str, *, aborted: bool) -> None:
        with self.lock:
            remaining = self.pending.pop(session_id, 1) - 1
            if remaining:
                self.pending[session_id] = remaining
            else:
                pass
            if aborted:
                self.sessions.pop(session_id, None)
            else:
                pass

    def close(self) -> None:
        with self.lock:
            self.sessions.clear()
            self.pending.clear()

    def compute(self, inputs: dict[str, str | bytes]) -> DiarizationResult:
        session_id, operation = inputs["session_id"], inputs["operation"]
        pcm = inputs.get("pcm", b"")
        # note (Richard Wang): A reset swaps the session in one operation, without a second slot.
        opened = inputs["next_session_id"] if operation == "reset" else session_id
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("Expected at most one second of mono PCM16 audio")
        elif operation not in OPERATIONS:
            raise ValueError("Unknown diarization stream operation")
        elif pcm and operation != "append":
            raise ValueError("Only append accepts audio")
        else:
            pass
        with self.lock:
            self.expire_idle_sessions()
            state = self.sessions.get(session_id)
            if operation in ("close", "reset"):
                self.sessions.pop(session_id, None)
            else:
                pass
            if operation == "close":
                return DiarizationResult(duration=0, segments=[])
            elif operation in ("open", "reset") and opened in self.sessions:
                raise ValueError("Diarization session already exists")
            elif operation in ("open", "reset") and len(self.sessions) >= (
                self.max_sessions
            ):
                raise ValueError("Live diarization session limit reached")
            elif operation in ("open", "reset"):
                self.sessions[opened] = LiveDiarization(self.model, device=self.device)
                return DiarizationResult(duration=0, segments=[])
            elif state is None:
                raise ValueError("Diarization session is closed or expired")
            elif not state.lock.acquire(blocking=False):
                raise ValueError("Diarization session already has a pending operation")
            else:
                pass
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
                else:
                    pass
