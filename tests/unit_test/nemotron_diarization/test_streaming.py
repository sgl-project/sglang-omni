# SPDX-License-Identifier: Apache-2.0
"""Streaming acoustic boundaries, hysteresis, and session admission regressions."""

from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch

from sglang_omni.models.nemotron_diarization.model import Preprocessor
from sglang_omni.models.nemotron_diarization.streaming import (
    LiveDiarization,
    LiveSessions,
)


class AcousticProbe:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.preprocessor.featurizer.window.copy_(torch.hann_window(400))
        self.preprocessor.featurizer.fb.fill_(0.01)
        self.chunks = []

    def forward_chunk(self, features, state, right):
        self.chunks.append(features.clone())
        return features.new_zeros(1, features.shape[2] - right, 8)


@pytest.mark.parametrize("length", [1, 159, 160, 16640, 50003])
def test_incremental_features_match_complete_audio_and_flush_partial_tail(length):
    model = AcousticProbe()
    audio = np.random.default_rng(17).normal(size=length).astype(np.float32)
    expected, count = model.preprocessor(torch.from_numpy(audio)[None, :])
    expected = expected[:, :, :count]
    live = LiveDiarization(model, device="cpu")
    outputs = []
    offset = 0
    for size in [1, 159, 161, 800, 3201] * (length // 4000 + 1):
        chunk = audio[offset : offset + size]
        if not len(chunk):
            break
        offset += len(chunk)
        outputs.append(live.probabilities(chunk))
        if model.chunks:
            assert offset >= 104 * 160 + 256  # No premature right-context padding.
        assert live.audio.size < 104 * 160 + 256 + 320
    assert offset == length
    if length > 104 * 160 + 256:
        assert sum(output.shape[1] for output in outputs) > 0  # Live, before EOF.
    outputs.append(live.probabilities(np.empty(0, np.float32), final=True))
    assert torch.cat(outputs, dim=1).shape == (1, length // 160, 8)
    assert live.audio.size == 0
    for index, chunk in enumerate(model.chunks):
        start = index * 72
        torch.testing.assert_close(
            chunk, expected[:, :, start : start + 104], rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="finished"):
        live.probabilities(np.zeros(160, np.float32))


def test_equal_threshold_preserves_activity_across_updates(monkeypatch):
    live = LiveDiarization(AcousticProbe(), device="cpu")
    values = iter([[0.6, 0.5], [0.5, 0.4, 0.5]])

    def probabilities(audio, *, final):
        del audio, final
        speaker = torch.tensor(next(values))
        live.frame += len(speaker)
        result = torch.zeros(1, len(speaker), 8)
        result[0, :, 0] = speaker
        result[0, :, 7] = speaker  # Overlap must survive.
        return result

    monkeypatch.setattr(live, "probabilities", probabilities)
    first = live.append(b"")
    second = live.append(b"")
    assert [(s.start, s.end, s.speaker) for s in first.segments] == [
        (0, 0.02, "speaker_0"),
        (0, 0.02, "speaker_7"),
    ]
    assert [(s.start, s.end, s.speaker) for s in second.segments] == [
        (0.02, 0.03, "speaker_0"),
        (0.02, 0.03, "speaker_7"),
    ]
    assert second.duration == 0.05
    # Nonempty segments must cross the same binary boundary as real stage results.
    msgspec.msgpack.encode(second)


def test_sessions_bound_state_reject_concurrent_mutation_and_release_slots(monkeypatch):
    manager = LiveSessions(SimpleNamespace(model=AcousticProbe(), device="cpu"), 2)

    def call(session, operation, pcm=b""):
        return manager.compute(dict(session_id=session, operation=operation, pcm=pcm))

    call("a", "open")
    call("b", "open")
    assert manager.sessions["a"].state is not manager.sessions["b"].state
    with pytest.raises(ValueError, match="limit"):
        call("c", "open")
    with manager.sessions["a"].lock:
        with pytest.raises(ValueError, match="pending"):
            call("a", "append", b"\0\0")
    call("a", "append", b"\0\0" * 111)
    call("b", "append", b"\0\0" * 222)
    assert call("a", "finish").duration == 111 / 16000
    assert "a" not in manager.sessions
    assert call("b", "finish").duration == 222 / 16000
    call("c", "open")
    call("c", "close")
    with pytest.raises(ValueError, match="closed"):
        call("c", "append", b"\0\0")
    call("idle", "open")
    manager.sessions["idle"].last_used -= 61
    call("new", "open")
    assert set(manager.sessions) == {"new"}


@pytest.mark.parametrize("pcm", [b"x", b"\0" * 32002, "wrong type"])
def test_invalid_pcm_cannot_advance_session(pcm):
    manager = LiveSessions(SimpleNamespace(model=AcousticProbe(), device="cpu"), 1)
    manager.compute(dict(session_id="s", operation="open"))
    with pytest.raises(ValueError, match="PCM16"):
        manager.compute(dict(session_id="s", operation="append", pcm=pcm))
    assert manager.sessions["s"].samples == 0


def test_inference_failure_discards_partial_state_and_releases_capacity(monkeypatch):
    model = AcousticProbe()
    manager = LiveSessions(SimpleNamespace(model=model, device="cpu"), 1)
    manager.compute(dict(session_id="failed", operation="open"))
    state = manager.sessions["failed"]
    manager.compute(dict(session_id="failed", operation="append", pcm=b"\0\0" * 16000))

    def fail(features, cache, right):
        cache.fifo = features.new_zeros(1, 1, 512)
        raise RuntimeError("inference failed after changing state")

    monkeypatch.setattr(model, "forward_chunk", fail)
    with pytest.raises(RuntimeError, match="after changing state"):
        manager.compute(
            dict(session_id="failed", operation="append", pcm=b"\0\0" * 1600)
        )
    assert manager.sessions == {}
    assert not state.lock.locked()
    manager.compute(dict(session_id="replacement", operation="open"))
    assert manager.sessions["replacement"].state.fifo.shape[1] == 0
