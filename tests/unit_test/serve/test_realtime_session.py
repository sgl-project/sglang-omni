# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the realtime session's VAD auto-commit / turn segmentation.

Regression coverage for https://github.com/sgl-project/sglang-omni/issues/1322:
server-side VAD dropped audio at an adjacent turn boundary when a single input
chunk contained the end of one utterance, enough silence to close that turn,
and the beginning of the next utterance.

The real :class:`StreamingVAD` loads a silero ONNX model, which is not
available in offline/unit CI. So this test replaces it with a fake VAD that
mirrors the same frame-by-frame state machine but classifies a frame as speech
purely from its content (non-zero samples). The session under test uses the
*real* :class:`RealtimeAudioBuffer` and :class:`RealtimeSession`, so the
bug-fix behavior (preserving the uncommitted suffix and re-segmenting it) is
exercised end to end.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.vad import Emit, VADConfig, VADEvent

# Mirror the real silero operating parameters.
VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16000


class _FakeWS:
    def __init__(self) -> None:
        self.application_state = WebSocketState.CONNECTED
        self.client_state = WebSocketState.CONNECTED
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class _FakeClient:
    """No-op engine client: responses/transcriptions come back empty."""

    async def abort(self, request_id: str) -> None:
        return

    async def completion_stream(self, request, request_id=None):
        return  # async generator that yields nothing
        yield  # pragma: no cover


class _FakeVAD:
    """Deterministic stand-in for StreamingVAD.

    A frame is speech iff it contains any non-zero sample, so a caller can
    synthesise ``speech / silence / speech`` simply by choosing which frames
    are zeroed. Reproduces the real post-fix behavior: processing stops as soon
    as a turn boundary (SPEECH_STOPPED) is emitted and resets re-segment from a
    clean slate.
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.remaining = bytearray()
        self.samples_consumed = 0
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = 0

    def reset(self) -> None:
        self.remaining.clear()
        self.samples_consumed = 0
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = 0

    def process(self, pcm_bytes: bytes) -> list[Emit]:
        if not pcm_bytes:
            return []
        self.remaining.extend(pcm_bytes)
        emits: list[Emit] = []
        silence_threshold = self.config.silence_duration_ms * VAD_SAMPLE_RATE // 1000
        while len(self.remaining) >= VAD_FRAME_SAMPLES * 2:
            frame_bytes = bytes(self.remaining[: VAD_FRAME_SAMPLES * 2])
            del self.remaining[: VAD_FRAME_SAMPLES * 2]
            self.samples_consumed += VAD_FRAME_SAMPLES
            speech = any(b != 0 for b in frame_bytes)
            if speech:
                self.silence_run_samples = 0
                self.last_speech_offset = self.samples_consumed
                if not self.is_speech:
                    self.is_speech = True
                    pad = self.config.prefix_padding_ms * VAD_SAMPLE_RATE // 1000
                    started_at = max(
                        0, self.samples_consumed - VAD_FRAME_SAMPLES - pad
                    )
                    emits.append(
                        Emit(event_type=VADEvent.SPEECH_STARTED, sample_offset=started_at)
                    )
            else:
                self.silence_run_samples += VAD_FRAME_SAMPLES
                if self.is_speech and self.silence_run_samples >= silence_threshold:
                    self.is_speech = False
                    emits.append(
                        Emit(
                            event_type=VADEvent.SPEECH_STOPPED,
                            sample_offset=self.last_speech_offset,
                        )
                    )
                    break
        return emits


def _frame(volume: int) -> bytes:
    """One 32 ms PCM16 frame (512 samples). ``volume`` 0 => silence, else speech."""
    sample = volume.to_bytes(2, "little", signed=True)
    return sample * VAD_FRAME_SAMPLES


@pytest.mark.asyncio
async def test_adjacent_turn_boundary_preserves_suffix(monkeypatch: pytest.MonkeyPatch):
    """A single chunk spanning two turns must not drop the second utterance.

    Input is ``speech -> 512 ms silence -> speech``. The expected server event
    sequence is ``speech_started -> speech_stopped -> committed -> speech_started``,
    with the second utterance's audio still retained in the buffer and all
    timestamps monotonic across the boundary.
    """
    monkeypatch.setattr(
        "sglang_omni.serve.realtime.session.StreamingVAD", _FakeVAD
    )

    ws = _FakeWS()
    session = RealtimeSession(
        ws, client=_FakeClient(), model_name="qwen3-omni", session_id="sess_test"
    )

    # speech (1 frame) -> silence (16 x 32 ms = 512 ms) -> speech (1 frame)
    pcm = _frame(4000) + _frame(0) * 16 + _frame(4000)
    append = json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode(),
        }
    )
    await session.dispatch(json.loads(append))
    # Let the background queue drainer (and its sends) settle.
    await asyncio.sleep(0.05)
    await session.teardown()

    types = [e["type"] for e in ws.sent]

    # The bug previously produced this exact window, but *without* the trailing
    # speech_started because the second utterance's audio was already dropped.
    # Filter out the background engine-pass events (response.* / transcription,
    # emitted by the queue drainer) so we assert on the VAD/commit sequence only.
    expected = [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "input_audio_buffer.speech_started",
    ]
    vad_commit_types = [t for t in types if t.startswith("input_audio_buffer.")]
    assert vad_commit_types == expected, vad_commit_types

    # Each turn is committed exactly once (only the first utterance completed).
    assert vad_commit_types.count("input_audio_buffer.committed") == 1

    # Timestamps remain monotonic across the boundary.
    started_a = next(
        e["audio_start_ms"] for e in ws.sent if e["type"] == "input_audio_buffer.speech_started"
    )
    stopped = next(
        e["audio_end_ms"] for e in ws.sent if e["type"] == "input_audio_buffer.speech_stopped"
    )
    started_b = [
        e["audio_start_ms"]
        for e in ws.sent
        if e["type"] == "input_audio_buffer.speech_started"
    ][-1]
    assert 0 <= started_a < stopped < started_b, (started_a, stopped, started_b)

    # The second utterance's audio must remain available for segmentation,
    # rather than being discarded along with the committed prefix.
    assert not session.audio_buffer.is_empty(), "uncommitted suffix was dropped"
    remaining = bytes(session.audio_buffer.buf)
    assert remaining.endswith(_frame(4000)), "second utterance missing from buffer"
