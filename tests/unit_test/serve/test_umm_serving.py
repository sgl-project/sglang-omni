# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client
from sglang_omni.client.types import (
    GenerateChunk,
    GenerateRequest,
    UMMGenerateChunk,
    UMMSegment,
)
from sglang_omni.native_generation.stage import data_url
from sglang_omni.pipeline.umm import UMMRequestError
from sglang_omni.proto import CompleteMessage, StreamMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import generate_stream

SEGMENTS = [
    UMMSegment("execution", 0, "text", "Drawing."),
    UMMSegment(
        "execution",
        1,
        "image",
        {
            "kind": "image",
            "url": data_url("image/png", b"x"),
            "mime_type": "image/png",
            "size_bytes": 1,
        },
    ),
    UMMSegment("execution", 2, "text", "Finished."),
]


def final_result(streamed: bool) -> dict[str, object]:
    # A streamed request already received each segment with its data.
    return {
        "type": "umm_result",
        "execution_id": "execution",
        "segments": [
            {
                key: value
                for key, value in segment.to_dict().items()
                if not (streamed and key == "data")
            }
            for segment in SEGMENTS
        ],
        "finish_reason": "stop",
    }


class Coordinator:
    def __init__(self):
        self.closed = False
        self.submitted = []

    def health(self):
        return {"running": True}

    async def submit(self, request_id, request):
        self.submitted.append(request_id)
        return final_result(streamed=False)

    async def stream(self, request_id, request):
        self.submitted.append(request_id)
        try:
            for segment in SEGMENTS:
                yield StreamMessage(
                    request_id,
                    "output",
                    segment.to_dict(),
                    modality=segment.kind,
                    chunk_id=segment.segment_index,
                )
            yield CompleteMessage(
                request_id, "output", True, result=final_result(streamed=True)
            )
        finally:
            self.closed = True


def umm_app(coordinator=None, *, interleaved=True):
    app = create_app(
        Client(coordinator or Coordinator()),
        model_name="cosmos3",
        interleaved_output=interleaved,
    )
    return TestClient(app)


# Stop strings never apply to an interleaved request, so its text matches
# its segments whether or not the client sets one.
@pytest.mark.parametrize("stop", [None, ["Fin"]])
def test_generate_and_chat_keep_ordered_terminal_manifest(stop):
    client = umm_app()
    generated = client.post(
        "/generate",
        json={"prompt": "Draw then describe", "sampling_params": {"stop": stop}},
    )
    assert generated.status_code == 200
    assert generated.json()["segments"] == [s.to_dict() for s in SEGMENTS]
    assert generated.json()["text"] == "Drawing.Finished."
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "cosmos3",
            "messages": [{"role": "user", "content": "Draw then describe"}],
            "stop": stop,
        },
    )
    assert chat.status_code == 200
    message = chat.json()["choices"][0]["message"]
    assert message["segments"] == [s.to_dict() for s in SEGMENTS]
    assert message["content"] == "Drawing.Finished."


def test_generate_sse_preserves_segment_order_and_explicit_final_snapshot():
    client = umm_app()
    response = client.post(
        "/generate", json={"prompt": "Draw then describe", "stream": True}
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
    assert final == {
        "finish_reason": "stop",
        "segments": [
            {key: value for key, value in s.to_dict().items() if key != "data"}
            for s in SEGMENTS
        ],
    }
    assert response.text.count(SEGMENTS[1].data["url"]) == 1
    assert events[-1] == "data: [DONE]"


@pytest.mark.parametrize("interleaved", [True, False])
def test_only_interleaved_pipelines_stream_generate(interleaved):
    coordinator = Coordinator()
    response = umm_app(coordinator, interleaved=interleaved).post(
        "/generate", json={"prompt": "hello", "stream": True, "return_logprob": False}
    )
    if interleaved:
        assert response.status_code == 200
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        assert response.status_code == 400
        assert "stream=true is not supported by /generate" in response.text
        assert coordinator.submitted == []


@pytest.mark.parametrize("stop", [None, ["Fin"]])
def test_chat_sse_exposes_media_segment_without_duplicate_text(stop):
    client = umm_app()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "cosmos3",
            "messages": [{"role": "user", "content": "Draw then describe"}],
            "stream": True,
            "stop": stop,
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


@pytest.mark.parametrize(
    "error,status,error_type,logged",
    [
        (
            UMMRequestError("UMM request exceeds max_outputs=17"),
            400,
            "invalid_request_error",
            False,
        ),
        (QueueFullError(), 503, "server_error", False),
        (RuntimeError("native worker failed"), 500, "server_error", True),
    ],
)
@pytest.mark.asyncio
async def test_generate_stream_error_reports_status_and_closes_the_stream(
    error, status, error_type, logged, caplog
):
    class FailingClient:
        closed = False

        async def generate(self, request, request_id):
            try:
                yield UMMGenerateChunk(request_id, segment=SEGMENTS[0])
                raise error
            finally:
                self.closed = True

    client = FailingClient()
    with caplog.at_level(logging.ERROR, logger="sglang_omni.serve.openai_api"):
        output = [
            event
            async for event in generate_stream(
                client, GenerateRequest(prompt="hello"), "r"
            )
        ]
    payload = json.loads(output[-1].split("data: ", 1)[1])
    assert output[-1].startswith("event: error")
    assert payload["error"] == {
        "message": str(error),
        "type": error_type,
        "code": status,
    }
    assert bool(caplog.records) is logged
    assert client.closed
    assert not any("[DONE]" in event for event in output)


def test_interleaved_generate_refuses_requested_rollout_artifacts():
    coordinator = Coordinator()
    response = umm_app(coordinator).post(
        "/generate", json={"prompt": "hello", "return_logprob": True}
    )
    assert response.status_code == 400
    assert "return_logprob" in response.json()["detail"]
    assert coordinator.submitted == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_completion_keeps_ordered_segments_with_their_data(stream):
    result = await Client(Coordinator()).completion(
        GenerateRequest(prompt="hello", stream=stream), request_id="r"
    )
    assert result.segments == SEGMENTS
    assert result.text == "Drawing.Finished."


class TextCoordinator(Coordinator):
    async def submit(self, request_id, request):
        return {"text": "hello", "finish_reason": "stop"}

    async def stream(self, request_id, request):
        yield StreamMessage(request_id, "decode", {"text": "hello"}, modality="text")
        yield CompleteMessage(
            request_id, "decode", True, result=await self.submit(request_id, request)
        )


def test_ordinary_results_carry_no_segment_keys():
    assert "segment" not in GenerateChunk("r", text="hello").to_dict()
    client = umm_app(TextCoordinator(), interleaved=False)
    generated = client.post(
        "/generate", json={"prompt": "hello", "return_logprob": False}
    ).json()
    assert generated["text"] == "hello" and "segments" not in generated
    chat = {"model": "qwen3-omni", "messages": [{"role": "user", "content": "hi"}]}
    message = client.post("/v1/chat/completions", json=chat).json()
    assert "segments" not in message["choices"][0]["message"]
    streamed = client.post("/v1/chat/completions", json={**chat, "stream": True})
    assert '"segment"' not in streamed.text and "hello" in streamed.text


@pytest.mark.parametrize("invalid", ["not_list", "wrong_execution", "reordered"])
def test_corrupt_final_result_is_rejected(invalid):
    result = final_result(streamed=False)
    if invalid == "not_list":
        result["segments"] = {"unexpected": 1}
    elif invalid == "wrong_execution":
        result["execution_id"] = "other"
    else:
        result["segments"].reverse()
    with pytest.raises(ValueError, match="UMM final"):
        Client.default_result_builder("r", result)


def test_untagged_result_segments_keep_the_generic_result_path():
    timestamped = {"id": 0, "start": 0.0, "end": 1.5, "text": "hello"}
    chunk = Client.default_result_builder(
        "r", {"text": "hello", "segments": [timestamped]}
    )
    assert chunk.text == "hello"
    assert not isinstance(chunk, UMMGenerateChunk)
