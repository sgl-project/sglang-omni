# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Qwen3-Omni real-streaming path."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    StreamingDetokenizeScheduler,
)
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


class _ByteTokenizer:
    """Token id → fixed bytes mapping; UTF-8 decode with errors='replace'."""

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
    """Mimics StreamItem.data shape passed to the scheduler inbox."""

    data: object
    metadata: dict | None = None


def _make_payload(stream: bool) -> StagePayload:
    """Build a StagePayload with the streaming flag plumbed through params."""
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": stream}),
        data={
            # Minimal PipelineState dict shape (decode_events will produce []).
            "engine_outputs": {
                "thinker": {
                    "output_ids": [],
                    "step": 0,
                    "is_final": True,
                    "extra_model_outputs": {},
                    "finish_reason": "stop",
                }
            },
            "thinker_out": None,
            "prompt": {"input_ids": []},
        },
    )


def _drain_outbox(scheduler: StreamingDetokenizeScheduler) -> list[OutgoingMessage]:
    out: list[OutgoingMessage] = []
    while not scheduler.outbox.empty():
        out.append(scheduler.outbox.get_nowait())
    return out


def test_utf8_multibyte_hold_then_emit():
    """A 3-byte CJK char split across 3 tokens must hold until complete."""
    # "你" is U+4F60 → b'\xe4\xbd\xa0'. Split byte-per-token.
    tok = _ByteTokenizer(
        vocab={1: b"\xe4", 2: b"\xbd", 3: b"\xa0", 99: b"hello"},
    )
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    out = _drain_outbox(sched)
    assert out == [], "should hold until UTF-8 char completes"

    sched._on_stream_chunk("req-1", _StreamItem(data=3))
    out = _drain_outbox(sched)
    assert len(out) == 1
    assert out[0].type == "stream"
    assert out[0].target is None  # → Coordinator
    assert out[0].data["text"] == "你"

    sched._on_stream_chunk("req-1", _StreamItem(data=99))
    out = _drain_outbox(sched)
    assert len(out) == 1
    assert out[0].data["text"] == "hello"


def test_special_tokens_emit_no_delta():
    """A token in the special set must not produce a stream chunk."""
    tok = _ByteTokenizer(
        vocab={1: b"hi", 2: b"<eos>"},
        special_token_ids={2},
    )
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=2)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    out = _drain_outbox(sched)
    assert len(out) == 1
    assert out[0].data["text"] == "hi"


def test_zero_token_stream_done_does_not_deadlock():
    """``stream_done`` arriving before any chunk and before ``new_request``
    must still let ``new_request`` finalize the streaming request."""
    tok = _ByteTokenizer(vocab={})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_done("req-1")
    sched._on_new_request("req-1", _make_payload(stream=True))
    out = _drain_outbox(sched)
    result_msgs = [m for m in out if m.type == "result"]
    assert len(result_msgs) == 1, "finalize must run even with zero-token output"


def test_non_streaming_finalizes_on_new_request():
    """``stream=False`` must finalize immediately on ``new_request``."""
    tok = _ByteTokenizer(vocab={})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_new_request("req-1", _make_payload(stream=False))
    out = _drain_outbox(sched)
    result_msgs = [m for m in out if m.type == "result"]
    assert len(result_msgs) == 1


def test_streaming_finalize_after_chunks_then_done_then_new_request():
    """Normal streaming order: chunks → done → new_request → finalize."""
    tok = _ByteTokenizer(vocab={1: b"hi"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_done("req-1")
    sched._on_new_request("req-1", _make_payload(stream=True))
    out = _drain_outbox(sched)
    types = [m.type for m in out]
    assert types.count("stream") >= 1
    assert types.count("result") == 1


def _payload_with_output_ids(stream: bool, output_ids: list[int]) -> StagePayload:
    """Variant of _make_payload that injects a non-empty output_ids list so
    decode_events produces a text_final event with the full reconstructed
    text in its payload — the case the slim-final invariant guards against.
    """
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": stream}),
        data={
            "engine_outputs": {
                "thinker": {
                    "output_ids": list(output_ids),
                    "step": len(output_ids),
                    "is_final": True,
                    "extra_model_outputs": {},
                    "finish_reason": "stop",
                }
            },
            "thinker_out": None,
            "prompt": {"input_ids": []},
            "stream_state": {},
        },
    )


def test_streaming_final_result_drops_full_text_to_avoid_duplication():
    """When stream=True, the terminal result must NOT carry the full
    reconstructed text — text deltas were already streamed via
    OutgoingMessage(type='stream'). A direct client that appends every
    chunk's text would otherwise emit the whole response twice.
    """
    tok = _ByteTokenizer(vocab={1: b"hi", 2: b" there"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_chunk("req-1", _StreamItem(data=2))
    sched._on_stream_done("req-1")
    sched._on_new_request(
        "req-1", _payload_with_output_ids(stream=True, output_ids=[1, 2])
    )

    out = _drain_outbox(sched)
    stream_msgs = [m for m in out if m.type == "stream"]
    result_msgs = [m for m in out if m.type == "result"]
    assert stream_msgs, "deltas must reach the client before the final result"
    assert len(result_msgs) == 1

    final_data = result_msgs[0].data.data
    assert (
        "text" not in final_data
    ), "streaming final must not duplicate text already emitted as deltas"
    assert "events" in final_data
    assert "usage" in final_data
    assert final_data.get("finish_reason") == "stop"


def test_non_streaming_final_result_keeps_full_text():
    """Non-streaming clients receive a single terminal result and must
    still see the full reconstructed text (regression guard for the
    slim-final branch in _build_result).
    """
    tok = _ByteTokenizer(vocab={1: b"hi", 2: b" there"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_new_request(
        "req-1", _payload_with_output_ids(stream=False, output_ids=[1, 2])
    )
    result_msgs = [m for m in _drain_outbox(sched) if m.type == "result"]
    assert len(result_msgs) == 1
    final_data = result_msgs[0].data.data
    assert final_data.get("text") == "hi there"
    assert final_data.get("finish_reason") == "stop"


def test_abort_clears_state():
    tok = _ByteTokenizer(vocab={1: b"hi"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    assert "req-1" in sched._state
    sched.abort("req-1")
    assert "req-1" not in sched._state


class _FakeCode2Wav:
    """Stand-in for the real vocoder; produces 4 audio samples per code frame."""

    total_upsample = 4

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (1, codebooks, num_frames). Output shape (1, frames * upsample).
        n_frames = codes.shape[-1]
        return torch.zeros(1, n_frames * self.total_upsample)


def _make_code_chunk(metadata: dict | None) -> _StreamItem:
    """One frame per chunk, single codebook, non-EOS code id."""
    return _StreamItem(data=torch.tensor([7], dtype=torch.long), metadata=metadata)


def test_code2wav_chunk_without_stream_metadata_emits_error():
    """Missing metadata['stream'] surfaces via outbox 'error' instead of raising."""
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=10,
        left_context_size=0,
    )
    sched._on_chunk("req-1", _make_code_chunk(metadata=None))

    out = sched.outbox.get_nowait()
    assert out.type == "error"
    assert out.request_id == "req-1"
    assert isinstance(out.data, RuntimeError)
    assert "metadata['stream']" in str(out.data)
    assert "req-1" not in sched._code_chunks
    assert "req-1" not in sched._stream_enabled


def test_code2wav_streaming_emits_per_window_and_slim_final():
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=0,
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": True}),
        data={},
    )
    sched._payloads["req-1"] = payload

    # Two chunks → triggers _decode_and_emit (stream_chunk_size=2).
    sched._on_chunk("req-1", _make_code_chunk(metadata={"stream": True}))
    sched._on_chunk("req-1", _make_code_chunk(metadata={"stream": True}))

    out: list[OutgoingMessage] = []
    while not sched.outbox.empty():
        out.append(sched.outbox.get_nowait())
    assert any(
        m.type == "stream" and m.target is None for m in out
    ), "streaming clients should receive per-window audio"

    # Done → slim final.
    sched._on_done("req-1")
    final = [
        m
        for m in (sched.outbox.get_nowait() for _ in range(sched.outbox.qsize()))
        if m.type == "result"
    ]
    assert len(final) == 1
    fdata = final[0].data.data
    assert fdata.get("modality") == "audio"
    assert "audio_waveform" not in fdata, "streaming final must be slim"


def test_code2wav_non_streaming_returns_full_pcm():
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=10,  # never trips during chunk feed
        left_context_size=0,
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": False}),
        data={},
    )
    sched._payloads["req-1"] = payload

    sched._on_chunk("req-1", _make_code_chunk(metadata={"stream": False}))
    sched._on_done("req-1")

    msgs: list[OutgoingMessage] = []
    while not sched.outbox.empty():
        msgs.append(sched.outbox.get_nowait())
    final = [m for m in msgs if m.type == "result"]
    assert len(final) == 1
    fdata = final[0].data.data
    assert "audio_waveform" in fdata, "non-streaming final must carry full PCM"
    assert fdata["modality"] == "audio"
    # No per-window stream emit on non-streaming.
    assert not any(m.type == "stream" for m in msgs)


def _bare_stage(*, is_terminal: bool, owns_io: bool = True) -> Stage:
    """Construct a Stage shell that bypasses __init__ for unit-level checks."""
    s = Stage.__new__(Stage)
    s.name = "decode" if is_terminal else "thinker"
    s._is_terminal = is_terminal
    s._owns_external_io = owns_io
    s._aborted = set()
    s.control_plane = None  # only touched on the success path
    return s


def test_send_stream_to_coordinator_raises_on_non_terminal():
    s = _bare_stage(is_terminal=False)
    with pytest.raises(RuntimeError, match="terminal"):
        asyncio.run(
            s._send_stream_to_coordinator(
                request_id="req-1",
                data={"text": "hi"},
                metadata={"modality": "text"},
            )
        )


def test_send_stream_to_coordinator_short_circuits_for_followers():
    """TP follower (owns_external_io=False) must drop silently, not raise."""
    s = _bare_stage(is_terminal=True, owns_io=False)
    asyncio.run(
        s._send_stream_to_coordinator(
            request_id="req-1",
            data={"text": "hi"},
            metadata={"modality": "text"},
        )
    )


def test_queue_stream_error_fast_fails_when_no_queue():
    """When _stream_queue is None, _queue_stream_error must surface a
    coordinator failure rather than silently dropping the error."""
    s = _bare_stage(is_terminal=True)
    s._stream_queue = None
    s._send_failure_calls: list[tuple[str, str]] = []

    async def _fake_send_failure(rid, err):
        s._send_failure_calls.append((rid, err))

    s._send_failure = _fake_send_failure
    asyncio.run(
        s._queue_stream_error("req-1", from_stage="thinker", error=RuntimeError("boom"))
    )
    assert s._send_failure_calls == [("req-1", "boom")]


def test_queue_stream_error_aborted_request_no_op():
    """An aborted request must not surface another failure to the coordinator."""
    s = _bare_stage(is_terminal=True)
    s._stream_queue = None
    s._aborted.add("req-1")
    s._send_failure_calls: list[tuple[str, str]] = []

    async def _fake_send_failure(rid, err):
        s._send_failure_calls.append((rid, err))

    s._send_failure = _fake_send_failure
    asyncio.run(
        s._queue_stream_error("req-1", from_stage="thinker", error=RuntimeError("late"))
    )
    assert s._send_failure_calls == []


def test_queue_stream_error_repeated_calls_are_idempotent_at_handler():
    """Each call invokes _send_failure; dedup happens at the coordinator."""
    s = _bare_stage(is_terminal=True)
    s._stream_queue = None
    s._send_failure_calls: list[tuple[str, str]] = []

    async def _fake_send_failure(rid, err):
        s._send_failure_calls.append((rid, err))

    s._send_failure = _fake_send_failure

    async def _drive():
        await s._queue_stream_error("req-1", "thinker", RuntimeError("first"))
        await s._queue_stream_error("req-1", "thinker", RuntimeError("second"))

    asyncio.run(_drive())
    assert len(s._send_failure_calls) == 2
    assert all(rid == "req-1" for rid, _ in s._send_failure_calls)


def test_late_stream_done_after_finalize_does_not_re_create_state():
    """Invariant: a late duplicate done does not allocate a new _RequestState row."""
    tok = _ByteTokenizer(vocab={1: b"hi"})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_chunk("req-1", _StreamItem(data=1))
    sched._on_stream_done("req-1")
    sched._on_new_request("req-1", _make_payload(stream=True))
    _drain_outbox(sched)
    assert "req-1" not in sched._state
    assert "req-1" not in sched._done_seen

    sched._on_stream_done("req-1")  # duplicate / late
    assert "req-1" not in sched._state, "late done must not re-create state"


def test_done_seen_cleared_on_abort():
    """_done_seen latches must be cleared on abort to bound memory."""
    tok = _ByteTokenizer(vocab={})
    sched = StreamingDetokenizeScheduler(tokenizer=tok, eos_token_id=None)

    sched._on_stream_done("req-1")
    assert "req-1" in sched._done_seen
    sched.abort("req-1")
    assert "req-1" not in sched._done_seen


class _RaisingTokenizer:
    """Decode raises on a specific marker token; succeeds otherwise.

    Used to force ``_on_stream_chunk`` to raise from inside the scheduler
    loop without monkey-patching private methods.
    """

    def __init__(self, *, eos_token_id: int | None = None) -> None:
        self.eos_token_id = eos_token_id

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if any(tid == 999 for tid in ids):
            raise RuntimeError("tokenizer-decode-boom")
        # Map every other id to a single ASCII letter; keeps deltas non-empty.
        return "".join(chr(ord("a") + (int(tid) % 26)) for tid in ids)


def test_scheduler_isolates_per_request_chunk_failure():
    """An exception inside ``_on_stream_chunk`` must surface as an
    ``OutgoingMessage(type="error")`` for that request only, and the
    scheduler thread must stay alive to serve later requests.
    """
    sched = StreamingDetokenizeScheduler(
        tokenizer=_RaisingTokenizer(),
        eos_token_id=None,
    )
    thread = threading.Thread(target=sched.start, daemon=True)
    thread.start()
    try:
        # req-bad: chunk carries the poison token id (999) → decode raises.
        sched.inbox.put(
            IncomingMessage(
                request_id="req-bad",
                type="stream_chunk",
                data=_StreamItem(data=999),
            )
        )
        err = sched.outbox.get(timeout=2.0)
        assert err.type == "error"
        assert err.request_id == "req-bad"
        assert isinstance(err.data, RuntimeError)
        assert "tokenizer-decode-boom" in str(err.data)
        # State for the failed request must be cleared.
        assert "req-bad" not in sched._state
        assert "req-bad" not in sched._done_seen

        # Scheduler thread must still be alive and processing.
        assert thread.is_alive()

        # req-good: a healthy non-streaming request finalizes normally.
        sched.inbox.put(
            IncomingMessage(
                request_id="req-good",
                type="new_request",
                data=_make_payload(stream=False),
            )
        )
        ok = sched.outbox.get(timeout=2.0)
        assert ok.type == "result"
        assert ok.request_id == "req-good"
    finally:
        sched.stop()
        thread.join(timeout=2.0)


def test_scheduler_isolates_per_request_finalize_failure():
    """An exception inside ``_finalize`` (e.g., via PipelineState.from_dict
    on a malformed payload) must isolate to that request without taking
    down the scheduler thread.
    """
    sched = StreamingDetokenizeScheduler(
        tokenizer=_RaisingTokenizer(),
        eos_token_id=None,
    )
    thread = threading.Thread(target=sched.start, daemon=True)
    thread.start()
    try:
        # Force _finalize to raise: poison token 999 in output_ids makes
        # _build_result call tokenizer.decode([999], ...) → RuntimeError.
        bad_payload = StagePayload(
            request_id="req-bad",
            request=OmniRequest(inputs=[], params={"stream": False}),
            data={
                "engine_outputs": {
                    "thinker": {
                        "output_ids": [999],
                        "step": 1,
                        "is_final": True,
                        "extra_model_outputs": {},
                        "finish_reason": "stop",
                    }
                },
                "thinker_out": None,
                "prompt": {"input_ids": []},
            },
        )
        sched.inbox.put(
            IncomingMessage(
                request_id="req-bad",
                type="new_request",
                data=bad_payload,
            )
        )
        err = sched.outbox.get(timeout=2.0)
        assert err.type == "error"
        assert err.request_id == "req-bad"
        assert isinstance(err.data, Exception)
        assert "req-bad" not in sched._state
        assert thread.is_alive()

        # Scheduler is still healthy.
        sched.inbox.put(
            IncomingMessage(
                request_id="req-good",
                type="new_request",
                data=_make_payload(stream=False),
            )
        )
        ok = sched.outbox.get(timeout=2.0)
        assert ok.type == "result"
        assert ok.request_id == "req-good"
    finally:
        sched.stop()
        thread.join(timeout=2.0)


def test_code2wav_abort_clears_all_per_request_state():
    sched = Code2WavScheduler(
        model=_FakeCode2Wav(),
        device="cpu",
        stream_chunk_size=10,
        left_context_size=0,
    )
    sched._on_chunk("req-1", _make_code_chunk(metadata={"stream": True}))
    assert "req-1" in sched._code_chunks
    assert "req-1" in sched._stream_enabled

    sched.abort("req-1")
    assert "req-1" not in sched._code_chunks
    assert "req-1" not in sched._emitted
    assert "req-1" not in sched._audio_chunks
    assert "req-1" not in sched._payloads
    assert "req-1" not in sched._stream_enabled
    assert "req-1" not in sched._pending_done


class _FakeCoordinatorForClient:
    """Async-iterates a pre-seeded message list as a Coordinator.stream() stand-in."""

    def __init__(self, messages):
        self._messages = list(messages)

    async def stream(self, request_id, omni_request):
        del request_id, omni_request
        for m in self._messages:
            yield m


def test_client_completion_stream_does_not_duplicate_full_text():
    """Two text deltas + a slim completion must yield a chunk sequence whose
    concatenated text equals the response once, not twice. Covers the
    `_default_stream_builder` / `_default_result_builder` translation path
    that scheduler-level tests can't reach.
    """
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest
    from sglang_omni.proto import CompleteMessage, StreamMessage

    messages = [
        StreamMessage(
            request_id="req-1",
            from_stage="decode",
            chunk={"text": "hi", "modality": "text", "stage_name": "decode"},
            stage_name="decode",
            modality="text",
        ),
        StreamMessage(
            request_id="req-1",
            from_stage="decode",
            chunk={"text": " there", "modality": "text", "stage_name": "decode"},
            stage_name="decode",
            modality="text",
        ),
        CompleteMessage(
            request_id="req-1",
            from_stage="decode",
            success=True,
            # Slim shape produced by StreamingDetokenizeScheduler when
            # stream=True: no top-level "text".
            result={
                "events": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 2,
                    "total_tokens": 2,
                },
                "finish_reason": "stop",
                "modality": "text",
            },
        ),
    ]
    client = Client(coordinator=_FakeCoordinatorForClient(messages))

    request = GenerateRequest(prompt="ignored-in-fake", stream=True)

    async def _collect():
        out = []
        async for chunk in client.completion_stream(request, request_id="req-1"):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())

    text_parts = [c.text for c in chunks if c.text]
    assert text_parts == ["hi", " there"], (
        f"streaming consumer must see each delta exactly once and no "
        f"reconstructed full text, got {text_parts!r}"
    )

    final = chunks[-1]
    assert final.finish_reason == "stop"
    # The terminal chunk must not re-emit the full response text.
    assert final.text in (None, "", "hi", " there"), (
        f"final chunk text must not be the full reconstructed response, "
        f"got {final.text!r}"
    )

    full = "".join(c.text or "" for c in chunks)
    assert (
        full == "hi there"
    ), f"concatenated stream must equal the response once, got {full!r}"


def test_client_completion_stream_non_streaming_keeps_full_text():
    """Regression guard: when the scheduler does NOT slim (non-streaming
    path), `Client.completion_stream()` must surface the full text on
    the terminal chunk so callers using the unified API still receive it.
    """
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest
    from sglang_omni.proto import CompleteMessage

    messages = [
        CompleteMessage(
            request_id="req-1",
            from_stage="decode",
            success=True,
            result={
                "events": [],
                "text": "hi there",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 2,
                    "total_tokens": 2,
                },
                "finish_reason": "stop",
                "modality": "text",
            },
        ),
    ]
    client = Client(coordinator=_FakeCoordinatorForClient(messages))

    # stream=True at the client surface still drives the streaming path;
    # what matters is that the coordinator hands us a non-slim result.
    request = GenerateRequest(prompt="ignored", stream=True)

    async def _collect():
        out = []
        async for chunk in client.completion_stream(request, request_id="req-1"):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    assert len(chunks) == 1
    assert chunks[0].text == "hi there"
    assert chunks[0].finish_reason == "stop"
