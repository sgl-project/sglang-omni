# SPDX-License-Identifier: Apache-2.0
"""Loopback chat and image services and importable tools for the cascade tool tests."""

import asyncio
import io
import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from sglang_omni.models.streaming_cascade.tasks import ToolOutcome
from sglang_omni.proto.tools import TaskArtifact

# Gates the slow tool waits on, by the gate name its arguments carry; a test creates them in its own loop.
TOOL_GATES: dict[str, asyncio.Event] = {}
TOOL_STARTED: dict[str, asyncio.Event] = {}


async def gated_tool(
    arguments: Mapping[str, object], options: Mapping[str, object]
) -> ToolOutcome:
    gate = str(arguments["gate"])
    TOOL_STARTED[gate].set()
    await TOOL_GATES[gate].wait()
    return ToolOutcome(
        summary=f"finished {gate}",
        artifact=TaskArtifact(type="image", media_type="image/jpeg", data="AAAA"),
    )


# Long enough that the reply which started the task ends first, as a real image service takes seconds.
SLEEPING_TOOL_S = 0.5


async def sleeping_tool(
    arguments: Mapping[str, object], options: Mapping[str, object]
) -> ToolOutcome:
    await asyncio.sleep(SLEEPING_TOOL_S)
    return ToolOutcome(
        summary=f"an image of {arguments['prompt']}",
        artifact=TaskArtifact(type="image", media_type="image/jpeg", data="AAAA"),
    )


async def failing_tool(
    arguments: Mapping[str, object], options: Mapping[str, object]
) -> ToolOutcome:
    raise RuntimeError("the service is down")


async def oversized_tool(
    arguments: Mapping[str, object], options: Mapping[str, object]
) -> ToolOutcome:
    return ToolOutcome(
        summary="big",
        artifact=TaskArtifact(type="image", media_type="image/jpeg", data="A" * 4000),
    )


def echo_tool(arguments: Mapping[str, object]) -> dict[str, object]:
    return {"echo": dict(arguments)}


def tool_call_events(
    name: str, arguments: dict[str, object], text: str = ""
) -> list[dict[str, object]]:
    """The SSE choices of a completion that says `text` and then calls one tool, its arguments streamed in two parts."""
    encoded = json.dumps(arguments)
    events: list[dict[str, object]] = (
        [{"delta": {"content": text}, "finish_reason": None}] if text else []
    )
    return [
        *events,
        {
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            },
            "finish_reason": None,
        },
        {
            "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": encoded[:5]}}]
            },
            "finish_reason": None,
        },
        {
            "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": encoded[5:]}}]
            },
            "finish_reason": None,
        },
        {"delta": {}, "finish_reason": "tool_calls"},
    ]


def text_events(text: str) -> list[dict[str, object]]:
    return [
        {"delta": {"content": text}, "finish_reason": None},
        {"delta": {}, "finish_reason": "stop"},
    ]


@dataclass
class ScriptedChat:
    base_url: str
    replies: list[list[dict[str, object]]]
    requests: list[dict[str, object]] = field(default_factory=list)


@pytest.fixture
def scripted_chat() -> Iterator[ScriptedChat]:
    """A chat completions service that streams its scripted replies in order and records each request."""
    chat = ScriptedChat(base_url="", replies=[])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chat.requests.append(request)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for choice in chat.replies.pop(0):
                self.wfile.write(
                    f"data: {json.dumps({'choices': [choice]})}\n\n".encode()
                )
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *arguments: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    chat.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield chat
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@dataclass
class FakeImageService:
    base_url: str
    pending_polls: int
    status: str = "SUCCEED"
    polls: int = 0
    authorizations: list[str] = field(default_factory=list)


@pytest.fixture
def modelscope_service() -> Iterator[FakeImageService]:
    """A ModelScope API-Inference image service: an async submit, pending_polls PENDING polls, then status, and a PNG at the result URL."""
    service = FakeImageService(base_url="", pending_polls=1)
    png = io.BytesIO()
    Image.new("RGB", (2048, 1024), (200, 120, 40)).save(png, format="PNG")

    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            service.authorizations.append(self.headers.get("Authorization", ""))
            assert self.path == "/v1/images/generations"
            assert (
                self.headers["X-ModelScope-Async-Mode"] == "true" and request["prompt"]
            )
            self.send(200, "application/json", b'{"task_id": "t1"}')

        def do_GET(self) -> None:
            if self.path == "/v1/tasks/t1":
                assert self.headers["X-ModelScope-Task-Type"] == "image_generation"
                service.polls += 1
                state = (
                    {"task_status": "PENDING"}
                    if service.polls <= service.pending_polls
                    else {
                        "task_status": service.status,
                        "output_images": [f"{service.base_url}/result.png"],
                    }
                )
                self.send(200, "application/json", json.dumps(state).encode())
            else:
                assert self.path == "/result.png"
                service.authorizations.append(self.headers.get("Authorization", ""))
                self.send(200, "image/png", png.getvalue())

        def log_message(self, *arguments: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    service.base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
