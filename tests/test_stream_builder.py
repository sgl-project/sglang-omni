# SPDX-License-Identifier: Apache-2.0
"""Tests for the streaming data path: decode_events -> _stream_builder -> _default_stream_builder.

These tests verify that incremental text produced by the thinker stage
reaches the client as ``GenerateChunk.text`` -- the integration seam that
was broken before fix/serving.
"""

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


def _build_stream_dict(events: list[OmniEvent], token_id: int, step: int) -> dict[str, Any]:
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
        assert "text" in stream_dict, (
            "stream_builder must surface text at top level for the client"
        )

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
            thinker_out={"output_ids": [tokenizer.eos_token_id], "step": 3, "is_final": False},
            state=state,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            step=3,
        )
    )

    assert any(e.is_final for e in events)

    stream_dict = _build_stream_dict(events, tokenizer.eos_token_id, 3)

    # Final events should NOT produce a top-level "text" (that would duplicate)
    assert "text" not in stream_dict, (
        "text_final events must be excluded to prevent duplicate full text"
    )


def test_finish_reason_in_sse_chunks() -> None:
    """SSE chunks must always include 'finish_reason' per OpenAI spec."""
    from fastapi.testclient import TestClient

    from sglang_omni.serve import create_app

    class _DummyClient:
        async def completion_stream(self, request, *, request_id, audio_format="wav"):
            from sglang_omni.client.types import CompletionStreamChunk

            yield CompletionStreamChunk(request_id=request_id, text="hi")
            yield CompletionStreamChunk(
                request_id=request_id, finish_reason="stop"
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
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            for choice in data["choices"]:
                assert "finish_reason" in choice, (
                    "Every SSE chunk must include finish_reason (null or string)"
                )
