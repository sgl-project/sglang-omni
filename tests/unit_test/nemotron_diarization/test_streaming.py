# SPDX-License-Identifier: Apache-2.0
"""Streaming acoustic boundaries, hysteresis, and session ownership."""

from typing import NoReturn

import numpy as np
import pytest
import torch

from sglang_omni.client.types import DiarizationOperation, DiarizationResult
from sglang_omni.models.nemotron_diarization import streaming
from sglang_omni.models.nemotron_diarization.model import Preprocessor
from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache
from sglang_omni.models.nemotron_diarization.streaming import (
    LiveDiarization,
    LiveSessions,
)

CHUNK_FRAMES = 72
WINDOW_FRAMES = 104
# note (Richard Wang): A window spans chunk, lookahead, and STFT tail, plus two hops of history.
WINDOW_SAMPLES = WINDOW_FRAMES * 160 + 256
HISTORY_SAMPLES = 320


class AcousticProbe:
    def __init__(self) -> None:
        self.preprocessor = Preprocessor()
        self.preprocessor.featurizer.window.copy_(torch.hann_window(400))
        self.preprocessor.featurizer.fb.fill_(0.01)
        self.chunks: list[torch.Tensor] = []

    def forward_chunk(
        self, features: torch.Tensor, state: SpeakerCache, right_frames: int
    ) -> torch.Tensor:
        self.chunks.append(features.clone())
        return features.new_zeros(1, features.shape[2] - right_frames, 8)


def call(
    live_sessions: LiveSessions,
    session_id: str,
    operation: DiarizationOperation,
    pcm: bytes = b"",
    next_session_id: str | None = None,
) -> DiarizationResult:
    inputs = {"session_id": session_id, "operation": operation, "pcm": pcm}
    if next_session_id is not None:
        inputs["next_session_id"] = next_session_id
    else:
        pass
    return live_sessions.compute(inputs)


@pytest.mark.parametrize("sample_count", [159, 16640, 50003])
def test_incremental_features_match_complete_audio_and_flush_partial_tail(
    sample_count: int,
) -> None:
    model = AcousticProbe()
    audio = np.random.default_rng(17).normal(size=sample_count).astype(np.float32)
    expected, frame_count = model.preprocessor(torch.from_numpy(audio)[None, :])
    expected = expected[:, :, :frame_count]
    live = LiveDiarization(model, device=torch.device("cpu"))
    outputs = []
    offset = 0
    for message_samples in [1, 159, 161, 800, 3201] * (sample_count // 4000 + 1):
        chunk = audio[offset : offset + message_samples]
        if not len(chunk):
            break
        else:
            pass
        offset += len(chunk)
        outputs.append(live.probabilities(chunk))
        if model.chunks:
            # note (Ryan Stewart): No premature right-context padding.
            assert offset >= WINDOW_SAMPLES
        else:
            pass
        assert live.audio.size < WINDOW_SAMPLES + HISTORY_SAMPLES
    if sample_count > WINDOW_SAMPLES:
        # note (Ryan Stewart): Live, before EOF.
        assert sum(output.shape[1] for output in outputs) > 0
    else:
        pass
    outputs.append(live.probabilities(np.empty(0, np.float32), final=True))
    assert torch.cat(outputs, dim=1).shape == (1, sample_count // 160, 8)
    assert live.audio.size == 0
    for index, chunk in enumerate(model.chunks):
        start_frame = index * CHUNK_FRAMES
        torch.testing.assert_close(
            chunk,
            expected[:, :, start_frame : start_frame + WINDOW_FRAMES],
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="finished"):
        live.probabilities(np.zeros(160, np.float32))


def test_equal_threshold_preserves_activity_across_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = LiveDiarization(AcousticProbe(), device=torch.device("cpu"))
    speaker_probabilities = iter([[0.6, 0.5], [0.5, 0.4, 0.5]])

    def probabilities(audio: np.ndarray, *, final: bool) -> torch.Tensor:
        speaker = torch.tensor(next(speaker_probabilities))
        live.frame += len(speaker)
        result = torch.zeros(1, len(speaker), 8)
        result[0, :, 0] = speaker
        # note (Ryan Stewart): Overlap must survive.
        result[0, :, 7] = speaker
        return result

    monkeypatch.setattr(live, "probabilities", probabilities)
    first, second = live.append(b""), live.append(b"")
    assert [
        (segment.start, segment.end, segment.speaker) for segment in first.segments
    ] == [
        (0, 0.02, "speaker_0"),
        (0, 0.02, "speaker_7"),
    ]
    assert [
        (segment.start, segment.end, segment.speaker) for segment in second.segments
    ] == [
        (0.02, 0.03, "speaker_0"),
        (0.02, 0.03, "speaker_7"),
    ]
    assert second.duration == 0.05


def test_sessions_bound_state_reject_invalid_or_concurrent_work_and_release() -> None:
    live_sessions = LiveSessions(AcousticProbe(), torch.device("cpu"), 2)
    call(live_sessions, "a", "open")
    call(live_sessions, "b", "open")
    assert live_sessions.sessions["a"].state is not live_sessions.sessions["b"].state
    with pytest.raises(ValueError, match="limit"):
        call(live_sessions, "c", "open")
    with pytest.raises(ValueError, match="PCM16"):
        call(live_sessions, "a", "append", b"x")
    with live_sessions.sessions["a"].lock:
        with pytest.raises(ValueError, match="pending"):
            call(live_sessions, "a", "append", b"\0\0")
    call(live_sessions, "a", "append", b"\0\0" * 111)
    assert call(live_sessions, "a", "finish").duration == 111 / 16000
    call(live_sessions, "b", "reset", next_session_id="c")
    assert set(live_sessions.sessions) == {"c"}
    call(live_sessions, "c", "close")
    with pytest.raises(ValueError, match="closed"):
        call(live_sessions, "c", "append", b"\0\0")
    call(live_sessions, "idle", "open")
    live_sessions.sessions["idle"].last_used -= 61
    call(live_sessions, "new", "open")
    assert set(live_sessions.sessions) == {"new"}


def test_inference_failure_discards_partial_state_and_releases_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AcousticProbe()
    live_sessions = LiveSessions(model, torch.device("cpu"), 1)
    call(live_sessions, "failed", "open")
    state = live_sessions.sessions["failed"]
    call(live_sessions, "failed", "append", b"\0\0" * 16000)

    def fail(
        features: torch.Tensor, speaker_cache: SpeakerCache, right_frames: int
    ) -> NoReturn:
        speaker_cache.fifo = features.new_zeros(1, 1, 512)
        raise RuntimeError("inference failed after changing state")

    monkeypatch.setattr(model, "forward_chunk", fail)
    with pytest.raises(RuntimeError, match="after changing state"):
        call(live_sessions, "failed", "append", b"\0\0" * 1600)
    assert live_sessions.sessions == {}
    assert not state.lock.locked()
    call(live_sessions, "replacement", "open")
    assert live_sessions.sessions["replacement"].state.fifo.shape[1] == 0


def test_queued_activity_prevents_expiry_and_abort_releases_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_s = [0.0]
    monkeypatch.setattr(streaming.time, "monotonic", lambda: now_s[0])
    live_sessions = LiveSessions(AcousticProbe(), torch.device("cpu"), 1)
    call(live_sessions, "live", "open")
    now_s[0] = 55.0
    assert live_sessions.reserve("live")
    now_s[0] = 65.0
    assert call(live_sessions, "live", "append", b"\0\0").duration == 0.0
    live_sessions.release("live", aborted=False)
    assert "live" in live_sessions.sessions and not live_sessions.pending
    live_sessions.reserve("live")
    live_sessions.release("live", aborted=True)
    assert not live_sessions.sessions and not live_sessions.pending
    call(live_sessions, "expired", "open")
    now_s[0] += 61.0
    assert not live_sessions.reserve("expired")
    with pytest.raises(ValueError, match="closed or expired"):
        call(live_sessions, "expired", "append", b"\0\0")
