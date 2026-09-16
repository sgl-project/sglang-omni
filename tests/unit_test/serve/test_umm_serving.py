# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client
from sglang_omni.client.types import CompletionResult, GenerateChunk, GenerateRequest
from sglang_omni.proto import CompleteMessage, StreamMessage
from sglang_omni.proto.continuation import UMMSegment
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _generate_stream

SEGMENTS = [
    UMMSegment("session", 0, "text", "Drawing."),
    UMMSegment(
        "session",
        1,
        "image",
        {
            "kind": "image",
            "url": "data:image/png;base64,eA==",
            "mime_type": "image/png",
            "size_bytes": 1,
        },
    ),
    UMMSegment("session", 2, "text", "Finished."),
]


class Coordinator:
    def __init__(self):
        self.closed = False

    def health(self):
        return {"running": True}

    async def submit(self, request_id, request):
        return {
            "session_id": "session",
            "segments": [s.to_dict() for s in SEGMENTS],
            "finish_reason": "stop",
        }

    async def stream(self, request_id, request):
        try:
            for segment in SEGMENTS:
                yield StreamMessage(
                    request_id,
                    "orchestrator",
                    segment.to_dict(),
                    modality=segment.kind,
                    chunk_id=segment.segment_index,
                )
            yield CompleteMessage(
                request_id,
                "orchestrator",
                True,
                result=await self.submit(request_id, request),
            )
        finally:
            self.closed = True


def test_generate_and_chat_keep_ordered_terminal_manifest():
    client = TestClient(create_app(Client(Coordinator()), model_name="cosmos3"))
    generated = client.post(
        "/generate", json={"prompt": "Draw then describe", "return_logprob": False}
    )
    assert generated.status_code == 200
    assert generated.json()["segments"] == [s.to_dict() for s in SEGMENTS]
    assert generated.json()["text"] == "Drawing.Finished."
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "cosmos3",
            "messages": [{"role": "user", "content": "Draw then describe"}],
        },
    )
    assert chat.status_code == 200
    message = chat.json()["choices"][0]["message"]
    assert message["segments"] == [s.to_dict() for s in SEGMENTS]
    assert message["content"] == "Drawing.Finished."


def test_generate_sse_preserves_segment_order_and_explicit_final_snapshot():
    client = TestClient(create_app(Client(Coordinator()), model_name="cosmos3"))
    response = client.post(
        "/generate",
        json={"prompt": "Draw then describe", "stream": True, "return_logprob": False},
    )
    assert response.status_code == 200
    events = response.text.strip().split("\n\n")
    assert [item.splitlines()[0] for item in events[:-1]] == [
        "event: segment",
        "event: segment",
        "event: segment",
        "event: complete",
    ]
    assert [json.loads(item.splitlines()[1][6:]) for item in events[:3]] == [
        s.to_dict() for s in SEGMENTS
    ]
    final = json.loads(events[3].splitlines()[1][6:])
    assert final["segments"] == [s.to_dict() for s in SEGMENTS]
    assert final["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]"


def test_chat_sse_exposes_media_segment_without_duplicate_text():
    client = TestClient(create_app(Client(Coordinator()), model_name="cosmos3"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "cosmos3",
            "messages": [{"role": "user", "content": "Draw then describe"}],
            "stream": True,
        },
    )
    messages = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [msg["choices"][0]["delta"] for msg in messages]
    assert [d["segment"] for d in deltas if "segment" in d] == [
        s.to_dict() for s in SEGMENTS
    ]
    assert "".join(d.get("content", "") for d in deltas) == "Drawing.Finished."
    assert messages[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_generate_stream_error_closes_owned_iterator_and_has_no_success_sentinel():
    class FailingClient:
        closed = False

        async def generate(self, request, request_id):
            try:
                yield GenerateChunk(request_id, segment=SEGMENTS[0])
                raise RuntimeError("native worker failed")
            finally:
                self.closed = True

    client = FailingClient()
    output = [
        event
        async for event in _generate_stream(
            client, GenerateRequest(prompt="hello"), "r"
        )
    ]
    assert client.closed
    assert output[-1].startswith("event: error")
    assert "native worker failed" in output[-1]
    assert not any("[DONE]" in event for event in output)


def test_streaming_generate_rejects_unsupported_rollout_artifacts_before_dispatch():
    client = TestClient(create_app(Client(Coordinator()), model_name="cosmos3"))
    response = client.post("/generate", json={"prompt": "hello", "stream": True})
    assert response.status_code == 400
    assert "rollout artifacts" in response.text


@pytest.mark.parametrize("route", ["/generate", "/v1/chat/completions"])
@pytest.mark.parametrize("disconnect", [True, False])
@pytest.mark.asyncio
async def test_registered_non_streaming_route_aborts_and_reaps_on_disconnect(
    route, disconnect
):
    started, finished, disconnected = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    class BlockingClient:
        def __init__(self):
            self.aborted = []
            self.dispatched = []

        async def completion(self, *args, **kwargs):
            self.dispatched.append(kwargs["request_id"])
            started.set()
            try:
                await asyncio.Future()
            finally:
                finished.set()

        async def abort(self, request_id):
            self.aborted.append(request_id)

    client = BlockingClient()
    app = create_app(client, model_name="cosmos3")
    body = {"request_id": "owned-request"}
    if route == "/generate":
        body.update(prompt="Draw then describe", return_logprob=False)
    else:
        body.update(
            model="cosmos3",
            messages=[{"role": "user", "content": "Draw then describe"}],
        )
    encoded = json.dumps(body).encode()
    body_sent = False
    sent = []

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": route,
        "raw_path": route.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode()),
        ],
        "client": ("127.0.0.1", 12000),
        "server": ("127.0.0.1", 8000),
    }
    pending = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(started.wait(), 1)
        if disconnect:
            disconnected.set()
        else:
            pending.cancel()
        # Waiting without cancelling on timeout distinguishes a real abort
        # from the test deadline cancelling an otherwise orphaned request.
        done, _ = await asyncio.wait({pending}, timeout=1)
        assert pending in done
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert len(client.dispatched) == 1
        assert client.aborted == client.dispatched
        if route == "/v1/chat/completions":
            assert client.dispatched == ["owned-request"]
        assert finished.is_set()
        assert not sent
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("language", ["zh", None])
@pytest.mark.asyncio
async def test_completion_preserves_language_with_ordered_segments(stream, language):
    class LanguageCoordinator(Coordinator):
        async def submit(self, request_id, request):
            result = await super().submit(request_id, request)
            if language is not None:
                result["language"] = language
            return result

    result = await Client(LanguageCoordinator()).completion(
        GenerateRequest(prompt="hello", stream=stream), request_id="r"
    )
    assert result.language == language
    assert result.segments == SEGMENTS
    assert result.text == "Drawing.Finished."
    assert result.media == [SEGMENTS[1].data]


@pytest.mark.asyncio
async def test_generate_stream_close_releases_coordinator_iterator():
    coordinator = Coordinator()
    stream = Client(coordinator).generate(GenerateRequest(prompt="hello"))
    chunk = await anext(stream)
    assert chunk.to_dict()["segment"] == SEGMENTS[0].to_dict()
    await stream.aclose()
    assert coordinator.closed


def test_legacy_chunk_serialization_does_not_add_segment_keys():
    assert "segment" not in GenerateChunk("r", text="hello").to_dict()
    assert "segments" not in GenerateChunk("r", text="hello").to_dict()


@pytest.mark.parametrize("invalid", ["not_list", "wrong_session", "reordered"])
def test_corrupt_terminal_manifest_is_rejected(invalid):
    manifest = {"session_id": "session", "segments": [s.to_dict() for s in SEGMENTS]}
    if invalid == "not_list":
        manifest["segments"] = {"unexpected": 1}
    elif invalid == "wrong_session":
        manifest["session_id"] = "other"
    else:
        manifest["segments"].reverse()
    with pytest.raises(ValueError, match="UMM terminal"):
        Client._default_result_builder("r", manifest)


def test_completion_result_keeps_legacy_positional_finish_reason():
    result = CompletionResult("r", "hello", None, None, "length")
    assert result.finish_reason == "length"
    assert result.segments is None
