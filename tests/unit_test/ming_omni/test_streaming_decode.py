# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MingStreamingDetokenizeScheduler and make_text_stream_output_builder.

All tests run without a real Ming-Omni model — a mock tokenizer is used.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.ming_omni.bootstrap import make_text_stream_output_builder
from sglang_omni.models.ming_omni.components.streaming_detokenizer import (
    _STATE_MAX,
    MingStreamingDetokenizeScheduler,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _SimpleTokenizer:
    """Token id → fixed string; no special tokens."""

    def __init__(self, vocab: dict[int, str], eos_token_id: int | None = None):
        self._vocab = vocab
        self.eos_token_id = eos_token_id

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self._vocab.get(i, "") for i in ids)


class _ByteTokenizer:
    """Token id → fixed bytes; UTF-8 decode with errors='replace'."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        special_token_ids: set[int] | None = None,
        eos_token_id: int | None = None,
    ):
        self._vocab = vocab
        self._special = special_token_ids or set()
        self.eos_token_id = eos_token_id

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        chunks: list[bytes] = []
        for tid in ids:
            if skip_special_tokens and tid in self._special:
                continue
            chunks.append(self._vocab[tid])
        return b"".join(chunks).decode("utf-8", errors="replace")


@dataclass
class _StreamItem:
    """Mimics the StreamItem the runtime wraps stream_chunk data in."""

    data: Any


def _make_payload(
    request_id: str,
    *,
    stream: bool,
    output_ids: list[int],
    output_modalities: list[str] | None = None,
) -> StagePayload:
    """Build a minimal StagePayload as the thinker would send to decode."""
    metadata: dict[str, Any] = {}
    if output_modalities is not None:
        metadata["output_modalities"] = output_modalities
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={"text": "hi"},
            params={"stream": stream},
            metadata=metadata,
        ),
        data={
            "thinker_out": {
                "output_ids": output_ids,
                "step": len(output_ids),
                "is_final": True,
                "extra_model_outputs": {},
            },
            "engine_outputs": {},
            "stream_state": {},
            "prompt": {},
        },
    )


def _drain_outbox(scheduler: MingStreamingDetokenizeScheduler) -> list[OutgoingMessage]:
    msgs = []
    while not scheduler.outbox.empty():
        msgs.append(scheduler.outbox.get_nowait())
    return msgs


def _run_scheduler(scheduler: MingStreamingDetokenizeScheduler) -> threading.Thread:
    t = threading.Thread(target=scheduler.start, daemon=True)
    t.start()
    return t


def _send(scheduler, msg: IncomingMessage) -> None:
    scheduler.inbox.put(msg)


def _collect(
    scheduler: MingStreamingDetokenizeScheduler, *, expect: int, timeout: float = 2.0
) -> list[OutgoingMessage]:
    """Blocking-get the expected number of outbox messages (no sleeps)."""
    return [scheduler.outbox.get(timeout=timeout) for _ in range(expect)]


# ---------------------------------------------------------------------------
# Tests: MingStreamingDetokenizeScheduler (threaded, real start() loop)
# ---------------------------------------------------------------------------


def test_streaming_emits_text_deltas():
    """stream=true: every stream_chunk produces its own delta, then the result."""
    vocab = {1: "A", 2: "B", 3: "C"}
    tok = _SimpleTokenizer(vocab, eos_token_id=0)
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=0)
    t = _run_scheduler(sched)

    rid = "req-1"
    try:
        for token_id in [1, 2, 3]:
            _send(
                sched,
                IncomingMessage(
                    request_id=rid,
                    type="stream_chunk",
                    data=SimpleNamespace(
                        data=torch.tensor([token_id], dtype=torch.long)
                    ),
                ),
            )
        _send(sched, IncomingMessage(request_id=rid, type="stream_done", data=None))
        _send(
            sched,
            IncomingMessage(
                request_id=rid,
                type="new_request",
                data=_make_payload(rid, stream=True, output_ids=[1, 2, 3]),
            ),
        )
        msgs = _collect(sched, expect=4)
    finally:
        sched.stop()
        t.join(timeout=2.0)
    assert not t.is_alive()

    # Per-token incrementality: one delta per chunk, result strictly last.
    assert [m.type for m in msgs] == ["stream", "stream", "stream", "result"]
    stream_msgs = msgs[:3]
    assert [m.data["text"] for m in stream_msgs] == ["A", "B", "C"]
    # target=None routes to the Coordinator; data keys are the client contract.
    assert all(m.target is None for m in stream_msgs)
    assert all(set(m.data) == {"text", "modality", "stage_name"} for m in stream_msgs)
    assert all(m.data["modality"] == "text" for m in stream_msgs)
    # Streaming result must NOT contain text (would double-send)
    assert "text" not in msgs[3].data.data


def test_non_streaming_emits_single_result():
    """stream=false: no stream messages, one result with full text."""
    vocab = {1: "H", 2: "i"}
    tok = _SimpleTokenizer(vocab, eos_token_id=0)
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=0)
    t = _run_scheduler(sched)

    rid = "req-2"
    try:
        _send(sched, IncomingMessage(request_id=rid, type="stream_done", data=None))
        _send(
            sched,
            IncomingMessage(
                request_id=rid,
                type="new_request",
                data=_make_payload(rid, stream=False, output_ids=[1, 2]),
            ),
        )
        msgs = _collect(sched, expect=1)
    finally:
        sched.stop()
        t.join(timeout=2.0)
    assert not t.is_alive()

    # First (and only) message is the result — no stream messages preceded it.
    assert msgs[0].type == "result"
    assert msgs[0].data.data.get("text") == "Hi"
    assert _drain_outbox(sched) == []


def test_stream_done_before_new_request():
    """stream_done arrives before new_request — must still finalize correctly."""
    vocab = {5: "X"}
    tok = _SimpleTokenizer(vocab)
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)
    t = _run_scheduler(sched)

    rid = "req-3"
    try:
        _send(
            sched,
            IncomingMessage(
                request_id=rid,
                type="stream_chunk",
                data=SimpleNamespace(data=torch.tensor([5], dtype=torch.long)),
            ),
        )
        # stream_done arrives BEFORE new_request (the normal runtime order)
        _send(sched, IncomingMessage(request_id=rid, type="stream_done", data=None))
        _send(
            sched,
            IncomingMessage(
                request_id=rid,
                type="new_request",
                data=_make_payload(rid, stream=True, output_ids=[5]),
            ),
        )
        msgs = _collect(sched, expect=2)
    finally:
        sched.stop()
        t.join(timeout=2.0)
    assert not t.is_alive()

    assert [m.type for m in msgs] == ["stream", "result"]


def test_stream_done_before_new_request_without_token_chunks_finalizes():
    """decode must finalize when runtime sends only stream_done before payload."""
    vocab = {1: "A", 2: "B"}
    tok = _SimpleTokenizer(vocab)
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)
    t = _run_scheduler(sched)

    rid = "req-speech-done-only"
    try:
        _send(sched, IncomingMessage(request_id=rid, type="stream_done", data=None))
        _send(
            sched,
            IncomingMessage(
                request_id=rid,
                type="new_request",
                data=_make_payload(rid, stream=True, output_ids=[1, 2]),
            ),
        )
        msgs = _collect(sched, expect=1)
    finally:
        sched.stop()
        t.join(timeout=2.0)
    assert not t.is_alive()

    assert msgs[0].type == "result"
    assert "text" not in msgs[0].data.data


def test_failure_isolation():
    """A malformed request must not prevent subsequent valid requests."""

    call_count = 0

    class _ErrorOnFirst:
        def decode(self, ids, skip_special_tokens=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("simulated tokenizer error")
            return "ok"

    sched = MingStreamingDetokenizeScheduler(_ErrorOnFirst(), eos_token_id=None)
    t = _run_scheduler(sched)

    rid_bad = "req-bad"
    rid_good = "req-good"
    try:
        # Bad request: tokenizer raises on first call
        _send(
            sched,
            IncomingMessage(
                request_id=rid_bad,
                type="stream_chunk",
                data=SimpleNamespace(data=torch.tensor([1], dtype=torch.long)),
            ),
        )
        err = sched.outbox.get(timeout=2.0)
        assert err.type == "error"
        assert err.request_id == rid_bad
        assert rid_bad not in sched._state
        assert t.is_alive()

        # Good request: should still be processed after the bad one
        _send(
            sched, IncomingMessage(request_id=rid_good, type="stream_done", data=None)
        )
        _send(
            sched,
            IncomingMessage(
                request_id=rid_good,
                type="new_request",
                data=_make_payload(rid_good, stream=False, output_ids=[2]),
            ),
        )
        ok = sched.outbox.get(timeout=2.0)
        assert ok.type == "result"
        assert ok.request_id == rid_good
    finally:
        sched.stop()
        t.join(timeout=2.0)
    assert not t.is_alive()


# ---------------------------------------------------------------------------
# Tests: detokenization state machine (synchronous, qwen3-test style)
# ---------------------------------------------------------------------------


def test_utf8_multibyte_hold_then_emit():
    """A 3-byte CJK char split across 3 tokens must hold until complete."""
    # "你" is U+4F60 → b'\xe4\xbd\xa0'. Split byte-per-token.
    tok = _ByteTokenizer(vocab={1: b"\xe4", 2: b"\xbd", 3: b"\xa0", 99: b"hello"})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    assert _drain_outbox(sched) == [], "should hold until UTF-8 char completes"

    sched._on_stream_chunk("req-1", _StreamItem(data=3))
    out = _drain_outbox(sched)
    assert len(out) == 1
    assert out[0].type == "stream"
    assert out[0].target is None  # → Coordinator
    assert out[0].data["text"] == "你"

    sched._on_stream_chunk("req-1", _StreamItem(data=99))
    out = _drain_outbox(sched)
    assert [m.data["text"] for m in out] == ["hello"]


def test_special_tokens_emit_no_delta():
    """A token skipped by skip_special_tokens must not produce a stream chunk."""
    tok = _ByteTokenizer(vocab={1: b"hi", 2: b"<eos>"}, special_token_ids={2})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=2)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    out = _drain_outbox(sched)
    assert len(out) == 1
    assert out[0].data["text"] == "hi"


def test_interior_replacement_char_does_not_stall_stream():
    """Only a TRAILING U+FFFD is held; an interior one must flush normally."""
    # 0x80 is a lone continuation byte → decodes to a permanent U+FFFD.
    tok = _ByteTokenizer(vocab={1: b"\x80", 2: b"ok"})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    # Trailing U+FFFD: indistinguishable from an incomplete char — held.
    assert _drain_outbox(sched) == []

    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    out = _drain_outbox(sched)
    assert [m.data["text"] for m in out] == ["�ok"]


def test_finalize_flushes_held_utf8_leftover():
    """Pending tokens held at stream end are flushed exactly once on finalize."""
    tok = _ByteTokenizer(vocab={1: b"\xe4", 2: b"\xbd"})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)

    rid = "req-1"
    sched._on_stream_chunk(rid, _StreamItem(data=1))
    sched._on_stream_chunk(rid, _StreamItem(data=2))
    assert _drain_outbox(sched) == []

    sched._on_stream_done(rid)
    sched._on_new_request(rid, _make_payload(rid, stream=True, output_ids=[1, 2]))
    out = _drain_outbox(sched)

    leftover = b"\xe4\xbd".decode("utf-8", errors="replace")
    stream_msgs = [m for m in out if m.type == "stream"]
    result_msgs = [m for m in out if m.type == "result"]
    assert [m.data["text"] for m in stream_msgs] == [leftover]
    assert len(result_msgs) == 1
    assert rid not in sched._state


def test_interleaved_requests_attribute_deltas_correctly():
    """Two concurrent requests through one loop keep per-request state isolated."""
    tok = _SimpleTokenizer({1: "A", 2: "B"})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)

    sched._on_stream_chunk("r1", _StreamItem(data=1))
    sched._on_stream_chunk("r2", _StreamItem(data=2))
    sched._on_stream_chunk("r1", _StreamItem(data=1))
    out = _drain_outbox(sched)
    assert [(m.request_id, m.data["text"]) for m in out] == [
        ("r1", "A"),
        ("r2", "B"),
        ("r1", "A"),
    ]


def test_late_stream_done_after_finalize_does_not_recreate_state():
    """A late duplicate done must not allocate a new _RequestState row."""
    tok = _SimpleTokenizer({1: "A"})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)

    rid = "req-1"
    sched._on_stream_chunk(rid, _StreamItem(data=1))
    sched._on_stream_done(rid)
    sched._on_new_request(rid, _make_payload(rid, stream=True, output_ids=[1]))
    _drain_outbox(sched)
    assert rid not in sched._state
    assert rid not in sched._done_seen

    sched._on_stream_done(rid)  # duplicate / late
    assert rid not in sched._state, "late done must not re-create state"


def test_abort_clears_state_and_done_seen():
    """abort() removes both the request state and any done latch."""
    tok = _SimpleTokenizer({1: "A"})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)

    sched._on_stream_chunk("r1", _StreamItem(data=1))
    assert "r1" in sched._state
    sched.abort("r1")
    assert "r1" not in sched._state

    sched._on_stream_done("r2")
    assert "r2" in sched._done_seen
    sched.abort("r2")
    assert "r2" not in sched._done_seen


def test_eviction_spares_live_and_done_entries():
    """Over-cap eviction drops only idle orphans — never live or done entries."""
    sched = MingStreamingDetokenizeScheduler(_SimpleTokenizer({}), eos_token_id=None)

    for i in range(_STATE_MAX):
        sched._ensure_state(f"r{i}")
    now = time.monotonic()
    # r0..r4999: idle orphans (evictable). r5000..r5099: idle but done (kept).
    for i in range(5000):
        sched._state[f"r{i}"].last_seen = now - 1000.0
    for i in range(5000, 5100):
        sched._state[f"r{i}"].last_seen = now - 1000.0
        sched._state[f"r{i}"].done = True

    sched._ensure_state("trigger")  # crosses _STATE_MAX → eviction

    assert all(f"r{i}" not in sched._state for i in range(0, 5000, 499))
    assert all(f"r{i}" in sched._state for i in range(5000, 5100))
    assert "r9999" in sched._state  # fresh entries are never evicted
    assert "trigger" in sched._state
    assert len(sched._state) == _STATE_MAX + 1 - 5000


def test_streaming_audio_only_request_keeps_text_in_final():
    """No deltas are emitted for audio-only requests, so the final keeps text."""
    tok = _SimpleTokenizer({1: "H", 2: "i"})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)

    rid = "req-1"
    sched._on_stream_done(rid)
    sched._on_new_request(
        rid,
        _make_payload(rid, stream=True, output_ids=[1, 2], output_modalities=["audio"]),
    )
    out = _drain_outbox(sched)
    result_msgs = [m for m in out if m.type == "result"]
    assert len(result_msgs) == 1
    assert result_msgs[0].data.data.get("text") == "Hi"


# ---------------------------------------------------------------------------
# Tests: factory and builder wiring
# ---------------------------------------------------------------------------


def test_create_decode_executor_returns_streaming_scheduler(monkeypatch):
    import sglang_omni.models.ming_omni.components.common as common

    monkeypatch.setattr(
        common,
        "load_ming_tokenizer",
        lambda path: _SimpleTokenizer({1: "A"}, eos_token_id=0),
    )
    from sglang_omni.models.ming_omni.stages import create_decode_executor

    sched = create_decode_executor("dummy-path")
    assert isinstance(sched, MingStreamingDetokenizeScheduler)
    for attr in ("inbox", "outbox", "start", "stop", "abort"):
        assert hasattr(sched, attr)
    assert sched._eos_token_id == 0


def test_select_stream_output_builder_branches():
    from sglang_omni.models.ming_omni.bootstrap import _select_stream_output_builder

    text_only = _select_stream_output_builder(
        False, tokenizer=_SimpleTokenizer({}), eos_token_id=None
    )
    msgs = text_only("req-1", _make_req_data(stream=True), _make_req_output(7))
    assert [m.target for m in msgs] == ["decode"]

    combined = _select_stream_output_builder(
        True, tokenizer=_SimpleTokenizer({1: "A"}), eos_token_id=None
    )
    msgs = combined("req-1", _make_req_data(stream=True), _make_req_output(1))
    assert "decode" in [m.target for m in msgs]


# ---------------------------------------------------------------------------
# Tests: make_text_stream_output_builder
# ---------------------------------------------------------------------------


def _make_req_data(*, stream: bool) -> Any:
    """Minimal req_data as OmniScheduler would pass to stream_output_builder."""
    payload = OmniRequest(inputs={"text": "hi"}, params={"stream": stream})
    stage_payload = StagePayload(request_id="r", request=payload, data={})
    req = SimpleNamespace(is_chunked=0)
    rd = SimpleNamespace(req=req, stage_payload=stage_payload)
    return rd


def _make_req_output(token_id: int) -> Any:
    return SimpleNamespace(data=token_id)


def test_text_stream_builder_emits_when_streaming():
    builder = make_text_stream_output_builder()
    msgs = builder("req-1", _make_req_data(stream=True), _make_req_output(42))
    assert len(msgs) == 1
    assert msgs[0].type == "stream"
    assert msgs[0].target == "decode"
    assert int(msgs[0].data.item()) == 42


def test_text_stream_builder_silent_when_not_streaming():
    builder = make_text_stream_output_builder()
    msgs = builder("req-1", _make_req_data(stream=False), _make_req_output(42))
    assert msgs == []


def test_text_stream_builder_silent_during_chunked_prefill():
    builder = make_text_stream_output_builder()
    payload = OmniRequest(inputs={"text": "hi"}, params={"stream": True})
    stage_payload = StagePayload(request_id="r", request=payload, data={})
    req = SimpleNamespace(is_chunked=1)  # chunked prefill in progress
    rd = SimpleNamespace(req=req, stage_payload=stage_payload)
    msgs = builder("req-1", rd, _make_req_output(42))
    assert msgs == []


def test_text_stream_builder_silent_when_audio_only_modality():
    """No text chunks when output_modalities=["audio"] (e.g. TTS-only request)."""
    builder = make_text_stream_output_builder()
    payload = OmniRequest(
        inputs={"text": "hi"},
        params={"stream": True},
        metadata={"output_modalities": ["audio"]},
    )
    stage_payload = StagePayload(request_id="r", request=payload, data={})
    req = SimpleNamespace(is_chunked=0)
    rd = SimpleNamespace(req=req, stage_payload=stage_payload)
    msgs = builder("req-1", rd, _make_req_output(42))
    assert msgs == [], "Should not emit text chunks when only audio is requested"


def test_text_stream_builder_emits_when_text_in_modalities():
    """Text chunks emitted when output_modalities includes text."""
    builder = make_text_stream_output_builder()
    payload = OmniRequest(
        inputs={"text": "hi"},
        params={"stream": True},
        metadata={"output_modalities": ["text", "audio"]},
    )
    stage_payload = StagePayload(request_id="r", request=payload, data={})
    req = SimpleNamespace(is_chunked=0)
    rd = SimpleNamespace(req=req, stage_payload=stage_payload)
    msgs = builder("req-1", rd, _make_req_output(42))
    assert len(msgs) == 1
    assert int(msgs[0].data.item()) == 42
