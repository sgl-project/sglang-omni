# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MingStreamingDetokenizeScheduler and make_text_stream_output_builder.

All tests run without a real Ming-Omni model — a mock tokenizer is used.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.ming_omni.components.streaming_detokenizer import (
    MingStreamingDetokenizeScheduler,
)
from sglang_omni.models.ming_omni.bootstrap import make_text_stream_output_builder
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


def _make_payload(request_id: str, *, stream: bool, output_ids: list[int]) -> StagePayload:
    """Build a minimal StagePayload as the thinker would send to decode."""
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={"text": "hi"},
            params={"stream": stream},
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


# ---------------------------------------------------------------------------
# Tests: MingStreamingDetokenizeScheduler
# ---------------------------------------------------------------------------


def test_streaming_emits_text_deltas():
    """stream=true: each stream_chunk produces a stream OutgoingMessage."""
    vocab = {1: "A", 2: "B", 3: "C"}
    tok = _SimpleTokenizer(vocab, eos_token_id=0)
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=0)
    t = _run_scheduler(sched)

    rid = "req-1"
    # Send 3 token chunks
    for token_id in [1, 2, 3]:
        _send(sched, IncomingMessage(
            request_id=rid,
            type="stream_chunk",
            data=SimpleNamespace(data=torch.tensor([token_id], dtype=torch.long)),
        ))
    _send(sched, IncomingMessage(request_id=rid, type="stream_done", data=None))
    _send(sched, IncomingMessage(
        request_id=rid,
        type="new_request",
        data=_make_payload(rid, stream=True, output_ids=[1, 2, 3]),
    ))

    import time
    time.sleep(0.3)
    sched.stop()
    t.join(timeout=1)

    msgs = _drain_outbox(sched)
    stream_msgs = [m for m in msgs if m.type == "stream"]
    result_msgs = [m for m in msgs if m.type == "result"]

    assert len(stream_msgs) >= 1, "Should emit at least one stream message"
    full_text = "".join(m.data["text"] for m in stream_msgs)
    assert full_text == "ABC"

    assert len(result_msgs) == 1
    # Streaming result must NOT contain text (would double-send)
    assert "text" not in result_msgs[0].data.data


def test_non_streaming_emits_single_result():
    """stream=false: no stream messages, one result with full text."""
    vocab = {1: "H", 2: "i"}
    tok = _SimpleTokenizer(vocab, eos_token_id=0)
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=0)
    t = _run_scheduler(sched)

    rid = "req-2"
    _send(sched, IncomingMessage(request_id=rid, type="stream_done", data=None))
    _send(sched, IncomingMessage(
        request_id=rid,
        type="new_request",
        data=_make_payload(rid, stream=False, output_ids=[1, 2]),
    ))

    import time
    time.sleep(0.3)
    sched.stop()
    t.join(timeout=1)

    msgs = _drain_outbox(sched)
    stream_msgs = [m for m in msgs if m.type == "stream"]
    result_msgs = [m for m in msgs if m.type == "result"]

    assert stream_msgs == [], "Non-streaming should produce no stream messages"
    assert len(result_msgs) == 1
    assert result_msgs[0].data.data.get("text") == "Hi"


def test_stream_done_before_new_request():
    """stream_done arrives before new_request — must still finalize correctly."""
    vocab = {5: "X"}
    tok = _SimpleTokenizer(vocab)
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)
    t = _run_scheduler(sched)

    rid = "req-3"
    _send(sched, IncomingMessage(
        request_id=rid, type="stream_chunk",
        data=SimpleNamespace(data=torch.tensor([5], dtype=torch.long)),
    ))
    # stream_done arrives BEFORE new_request (common race condition)
    _send(sched, IncomingMessage(request_id=rid, type="stream_done", data=None))

    import time
    time.sleep(0.05)

    _send(sched, IncomingMessage(
        request_id=rid,
        type="new_request",
        data=_make_payload(rid, stream=True, output_ids=[5]),
    ))

    time.sleep(0.3)
    sched.stop()
    t.join(timeout=1)

    msgs = _drain_outbox(sched)
    result_msgs = [m for m in msgs if m.type == "result"]
    assert len(result_msgs) == 1, "Must finalize even when done arrives before new_request"


def test_abort_clears_state():
    """abort() removes the request from internal state."""
    tok = _SimpleTokenizer({1: "A"})
    sched = MingStreamingDetokenizeScheduler(tok, eos_token_id=None)

    rid = "req-4"
    sched._ensure_state(rid)
    assert rid in sched._state

    sched.abort(rid)
    assert rid not in sched._state


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


def test_failure_isolation():
    """A malformed request must not prevent subsequent valid requests."""
    import time

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

    # Bad request: tokenizer raises on first call
    rid_bad = "req-bad"
    _send(sched, IncomingMessage(
        request_id=rid_bad,
        type="stream_chunk",
        data=SimpleNamespace(data=torch.tensor([1], dtype=torch.long)),
    ))

    time.sleep(0.15)

    # Good request: should still be processed after the bad one
    rid_good = "req-good"
    _send(sched, IncomingMessage(request_id=rid_good, type="stream_done", data=None))
    _send(sched, IncomingMessage(
        request_id=rid_good,
        type="new_request",
        data=_make_payload(rid_good, stream=False, output_ids=[2]),
    ))

    time.sleep(0.3)
    sched.stop()
    t.join(timeout=1)

    msgs = _drain_outbox(sched)
    error_msgs = [m for m in msgs if m.request_id == rid_bad and m.type == "error"]
    result_msgs = [m for m in msgs if m.request_id == rid_good and m.type == "result"]

    assert len(error_msgs) == 1, "Malformed request should emit error message"
    assert len(result_msgs) == 1, "Valid request after failure should still succeed"
