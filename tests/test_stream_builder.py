# SPDX-License-Identifier: Apache-2.0
"""Tests for the streaming data path."""

from __future__ import annotations

import json
from typing import Any

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk
from sglang_omni.models.qwen3_omni.io import OmniEvent, PipelineState
from sglang_omni.models.qwen3_omni.pipeline.merge import decode_events
from sglang_omni.models.qwen3_omni.pipeline.stages import _event_to_dict
from sglang_omni.proto import StreamMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer that maps token ids to single characters."""

    eos_token_id = 99

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(ord("a") + (t % 26)) for t in token_ids)


class _MultiByteFakeTokenizer:
    """Tokenizer that simulates multi-byte character behavior.

    Token 50 alone decodes to the replacement character (incomplete sequence).
    Tokens [50, 51] together decode to a complete emoji.
    All other tokens map to single ASCII characters.
    """

    eos_token_id = 99

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        parts: list[str] = []
        i = 0
        while i < len(token_ids):
            tid = token_ids[i]
            if tid == 50 and i + 1 < len(token_ids) and token_ids[i + 1] == 51:
                parts.append("\U0001f60a")  # 😊
                i += 2
            elif tid == 50:
                parts.append("\ufffd")  # incomplete
                i += 1
            elif tid == 51:
                parts.append("\ufffd")
                i += 1
            else:
                parts.append(chr(ord("a") + (tid % 26)))
                i += 1
        return "".join(parts)


def _build_stream_dict(
    events: list[OmniEvent], token_id: int, step: int
) -> dict[str, Any]:
    """Reproduce the _stream_builder text-extraction logic from stages.py."""
    text_delta = ""
    for event in events:
        if event.is_final:
            continue
        t = event.payload.get("text")
        if event.modality == "text" and t:
            text_delta += t

    result: dict[str, Any] = {
        "events": [_event_to_dict(event) for event in events],
        "token_id": token_id,
        "step": step,
        "stage": "thinker_stage",
    }
    if text_delta:
        result["text"] = text_delta
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_text_delta_reaches_client() -> None:
    """Token-by-token text deltas must appear in GenerateChunk.text."""
    tokenizer = _FakeTokenizer()
    state = PipelineState()

    collected_texts: list[str] = []
    for step, token_id in enumerate([0, 1, 2], start=1):
        events = list(
            decode_events(
                thinker_out={"output_ids": [token_id], "step": step, "is_final": False},
                state=state,
                tokenizer=tokenizer,
                eos_token_id=tokenizer.eos_token_id,
                step=step,
            )
        )

        assert len(events) == 1
        assert events[0].type == "text_delta"
        assert not events[0].is_final

        stream_dict = _build_stream_dict(events, token_id, step)

        # Key assertion: top-level "text" must exist
        assert (
            "text" in stream_dict
        ), "stream_builder must surface text at top level for the client"

        # Feed through _default_stream_builder (the real consumer)
        msg = StreamMessage(
            request_id="req-1",
            from_stage="thinker",
            chunk=stream_dict,
        )
        chunk = Client._default_stream_builder("req-1", msg)

        assert isinstance(chunk, GenerateChunk)
        assert chunk.text, f"chunk.text must be non-empty at step {step}"
        collected_texts.append(chunk.text)

    # All deltas together should reconstruct the full decoded text
    assert "".join(collected_texts) == tokenizer.decode([0, 1, 2])


def test_final_event_excluded_from_text_delta() -> None:
    """text_final events must NOT produce a top-level 'text' to avoid duplicates."""
    tokenizer = _FakeTokenizer()
    state = PipelineState()

    # First, accumulate a couple of tokens so stream_state has content
    for token_id in [0, 1]:
        list(
            decode_events(
                thinker_out={"output_ids": [token_id], "step": 1, "is_final": False},
                state=state,
                tokenizer=tokenizer,
                eos_token_id=tokenizer.eos_token_id,
                step=1,
            )
        )

    # Now emit the EOS token -> should produce a text_final event
    events = list(
        decode_events(
            thinker_out={
                "output_ids": [tokenizer.eos_token_id],
                "step": 3,
                "is_final": False,
            },
            state=state,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            step=3,
        )
    )

    assert any(e.is_final for e in events)

    stream_dict = _build_stream_dict(events, tokenizer.eos_token_id, 3)

    # Final events should NOT produce a top-level "text" (that would duplicate)
    assert (
        "text" not in stream_dict
    ), "text_final events must be excluded to prevent duplicate full text"


def test_finish_reason_in_sse_chunks() -> None:
    """SSE chunks must always include 'finish_reason' per OpenAI spec."""
    from fastapi.testclient import TestClient

    from sglang_omni.serve import create_app

    class _DummyClient:
        async def completion_stream(self, request, *, request_id, audio_format="wav"):
            from sglang_omni.client.types import CompletionStreamChunk

            yield CompletionStreamChunk(request_id=request_id, text="hi")
            yield CompletionStreamChunk(request_id=request_id, finish_reason="stop")

        def health(self):
            return {"running": True}

    client = TestClient(create_app(_DummyClient()))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            for choice in data["choices"]:
                assert (
                    "finish_reason" in choice
                ), "Every SSE chunk must include finish_reason (null or string)"


def test_finish_chunk_is_separate() -> None:
    """finish_reason='stop' must be in a dedicated chunk with empty delta."""
    from fastapi.testclient import TestClient

    from sglang_omni.serve import create_app

    class _DummyClient:
        async def completion_stream(self, request, *, request_id, audio_format="wav"):
            from sglang_omni.client.types import CompletionStreamChunk

            yield CompletionStreamChunk(request_id=request_id, text="hello")
            yield CompletionStreamChunk(request_id=request_id, text=" world")
            # Pipeline finish chunk: full text + finish_reason.
            yield CompletionStreamChunk(
                request_id=request_id, text="hello world", finish_reason="stop"
            )

        def health(self):
            return {"running": True}

    client = TestClient(create_app(_DummyClient()))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        events = []
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

    # Content chunks must all have finish_reason: null
    content_chunks = [e for e in events if e["choices"][0].get("finish_reason") is None]
    for chunk in content_chunks:
        delta = chunk["choices"][0]["delta"]
        assert (
            delta.get("content") is not None or delta.get("role") is not None
        ), "Content chunks must carry role or text"

    # Exactly one finish chunk with empty delta
    finish_chunks = [
        e for e in events if e["choices"][0].get("finish_reason") == "stop"
    ]
    assert (
        len(finish_chunks) == 1
    ), f"Expected exactly 1 finish chunk, got {len(finish_chunks)}"
    final_delta = finish_chunks[0]["choices"][0]["delta"]
    assert (
        final_delta.get("content") is None
    ), f"Finish chunk should have empty delta, got content={final_delta.get('content')!r}"


def test_multibyte_char_no_resend() -> None:
    """Multi-byte characters (emoji) must not trigger full text re-sends."""
    tokenizer = _MultiByteFakeTokenizer()
    state = PipelineState()

    all_events: list[OmniEvent] = []

    # Token 0 -> 'a'
    events = list(
        decode_events(
            thinker_out={"output_ids": [0], "step": 1, "is_final": False},
            state=state,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            step=1,
        )
    )
    all_events.extend(events)

    # Token 50 -> incomplete multi-byte (should be buffered, no event)
    events = list(
        decode_events(
            thinker_out={"output_ids": [50], "step": 2, "is_final": False},
            state=state,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            step=2,
        )
    )
    all_events.extend(events)

    # Token 51 -> completes the emoji
    events = list(
        decode_events(
            thinker_out={"output_ids": [51], "step": 3, "is_final": False},
            state=state,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            step=3,
        )
    )
    all_events.extend(events)

    # Token 2 -> 'c'
    events = list(
        decode_events(
            thinker_out={"output_ids": [2], "step": 4, "is_final": False},
            state=state,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            step=4,
        )
    )
    all_events.extend(events)

    # Collect all emitted text
    emitted_texts = []
    for event in all_events:
        if not event.is_final:
            t = event.payload.get("text", "")
            emitted_texts.append(t)

    full_text = "".join(emitted_texts)
    expected = tokenizer.decode([0, 50, 51, 2])  # "a😊c"

    # No replacement characters should reach the client
    assert (
        "\ufffd" not in full_text
    ), f"Replacement character leaked to client: {full_text!r}"
    # Concatenated deltas should reconstruct the correct text
    assert full_text == expected, f"Expected {expected!r}, got {full_text!r}"
    # Should NOT have a single event containing the full text (no re-sends)
    for event in all_events:
        if not event.is_final:
            assert (
                event.payload.get("text") != expected
            ), "Full text appeared in a single event — indicates a re-send bug"
