# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
from fastapi.responses import JSONResponse

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    ResourceHTTPConnection,
)
from sglang_omni.proto import OmniRequest, StagePayload, StreamMessage
from sglang_omni.scheduling import native_chat_scheduler
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.native_chat_scheduler import (
    NativeChatScheduler,
    build_chat_fields,
    create_native_chat_scheduler,
    native_chat_server_kwargs,
    warm_up,
)
from sglang_omni.serve.openai_api import build_chat_generate_request
from sglang_omni.serve.openai_errors import generation_error_status_code
from sglang_omni.serve.protocol import ChatCompletionRequest


class NativeRequest(SimpleNamespace):
    model_fields = {
        "model",
        "messages",
        "rid",
        "n",
        "temperature",
        "seed",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "logprobs",
        "return_meta_info",
        "return_token_ids",
    }


def payload(request_id="r1", inputs="Describe the scene", **params):
    return StagePayload(request_id, OmniRequest(inputs, params), None)


class Engine:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.server_args = SimpleNamespace(served_model_name=None)
        self.shutdown_count = 0
        self.aborts = []
        self.tokenizer_manager = SimpleNamespace(
            abort_request=self.abort_request, served_model_name="cosmos3"
        )

    def abort_request(self, rid="", abort_all=False):
        assert asyncio.get_running_loop() is self.loop
        self.aborts.append((rid, abort_all))

    def shutdown(self):
        self.shutdown_count += 1


class Chat:
    def __init__(self, engine):
        self.engine = engine
        self.calls = []
        self.files = {}
        self.started = threading.Event()
        self.active = 0
        self.peak = 0
        self.closed_streams = 0
        self.background_count = 0
        self.release = None

    async def handle_request(self, request, raw_request):
        assert asyncio.get_running_loop() is self.engine.loop
        assert raw_request is None
        self.calls.append(request)
        self.started.set()
        message = request.messages[0]
        content = message["content"] if isinstance(message, dict) else message.content
        for part in content if isinstance(content, list) else ():
            url = (part.get("image_url") or part.get("video_url") or {}).get("url", "")
            if url.startswith("file://"):
                # Native media loaders read a file URL as its unquoted path.
                self.files[url] = Path(unquote(urlparse(url).path)).read_bytes()
        if content == "hang":
            await asyncio.Event().wait()
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if content == "wait for release":
                if self.release is None:
                    self.release = asyncio.Event()
                await self.release.wait()
            else:
                await asyncio.sleep(0.02)
        finally:
            self.active -= 1
        if getattr(request, "stream", False):

            async def frames():
                try:
                    for text in ["你", "好", " 👩🏽‍🚀", " e\u0301"]:
                        yield "data: " + json.dumps(
                            {
                                "choices": [
                                    {"delta": {"content": text}, "finish_reason": None}
                                ]
                            }
                        ) + "\n\n"
                        if content == "fail after first delta":
                            yield 'data: {"error": {"message": "native failure"}}\n\n'
                    yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
                    yield 'data: {"choices": [], "usage": {"completion_tokens": 4}}\n\n'
                    yield "data: [DONE]\n\n"
                finally:
                    self.closed_streams += 1

            async def background():
                # The native disconnect abort sleeps before it inspects the
                # request. Awaiting it on the completion path adds that
                # sleep to every streamed completion.
                self.background_count += 1
                await asyncio.sleep(2)

            return SimpleNamespace(body_iterator=frames(), background=background)
        return JSONResponse(
            {
                "choices": [
                    {
                        "message": {"content": "你好"},
                        "finish_reason": "length",
                        "response_token_ids": [1, 2],
                        "meta_info": {
                            "output_token_logprobs": [
                                [-0.1, 1, "你"],
                                [-0.2, 2, "好"],
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                },
            }
        )


@contextmanager
def running(*, request_type=NativeRequest):
    engine = Engine()
    chat = Chat(engine)
    scheduler = NativeChatScheduler(engine, chat, request_type, "chat")
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        yield scheduler, engine, chat
    finally:
        scheduler.stop()
        scheduler.stop()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not scheduler.loop_thread.is_alive()
        assert engine.loop.is_closed()
        assert engine.shutdown_count == 1


def test_native_loop_thread_selects_the_serving_device(monkeypatch):
    from sglang_omni.platforms import current_platform

    selected: list[tuple[str, int]] = []
    monkeypatch.setattr(
        current_platform,
        "set_device",
        lambda device: selected.append((threading.current_thread().name, device)),
    )
    engine = Engine()
    scheduler = NativeChatScheduler(
        engine, Chat(engine), NativeRequest, "chat", device=5
    )
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        assert selected == [("native-chat-tokenizer", 5)]
    finally:
        scheduler.stop()
        thread.join(timeout=3)
    assert not thread.is_alive()


def enqueue(scheduler, item):
    scheduler.enqueue(IncomingMessage(item.request_id, "new_request", item))


def test_native_chat_request_tracks_manager_name_changes():
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    with running(request_type=ChatCompletionRequest) as (scheduler, engine, chat):
        engine.server_args.served_model_name = "initial-alias"
        engine.tokenizer_manager.served_model_name = "initial-alias"
        enqueue(scheduler, payload("before"))
        assert scheduler.outbox.get(timeout=3).type == "result"
        engine.tokenizer_manager.served_model_name = "/models/reloaded-Cosmos3"
        enqueue(scheduler, payload("after"))
        assert scheduler.outbox.get(timeout=3).type == "result"
        assert [request.model for request in chat.calls] == [
            "initial-alias",
            "/models/reloaded-Cosmos3",
        ]
        assert engine.server_args.served_model_name == "initial-alias"


def test_native_text_and_vision_parameters_preserved():
    original = {
        "messages": [{"role": "user", "content": "What is here?"}],
        "images": ["data:image/png,AAAA"],
    }
    fields = build_chat_fields(
        payload(
            inputs=original,
            temperature=0.0,
            seed=0,
            max_new_tokens=7,
            return_logprob=True,
        ),
        "cosmos3",
        NativeRequest.model_fields,
        "chat",
    )
    assert (
        fields["messages"][0]["content"][1]["image_url"]["url"] == original["images"][0]
    )
    assert original["messages"][0]["content"] == "What is here?"
    assert (
        fields["temperature"],
        fields["seed"],
        fields["max_completion_tokens"],
    ) == (0.0, 0, 7)
    assert fields["return_meta_info"] and fields["return_token_ids"]
    assert fields["rid"] == "r1"


@pytest.mark.parametrize(
    "request_",
    [
        ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], top_k=5),
        GenerateRequest(prompt="hi", sampling=SamplingParams(top_k=5)),
    ],
    ids=["chat", "sdk"],
)
def test_omitted_sampling_resolves_from_the_checkpoint_config(request_):
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest as NativeChatRequest,
    )

    if isinstance(request_, ChatCompletionRequest):
        request_ = build_chat_generate_request(request_)
    else:
        pass
    item = StagePayload("r1", Client.build_omni_request(request_), None)
    fields = build_chat_fields(
        item, "cosmos3", set(NativeChatRequest.model_fields), "chat"
    )
    sampling = NativeChatRequest(**fields).to_sampling_params(
        stop=[], model_generation_config={"temperature": 0.7, "top_p": 0.8}
    )
    assert (sampling["temperature"], sampling["top_p"], sampling["top_k"]) == (
        0.7,
        0.8,
        5,
    )


@pytest.mark.parametrize(
    "item",
    [
        payload(inputs=[]),
        payload(inputs=[1, 2]),
        payload(inputs={"messages": [1]}),
        payload(inputs={"messages": [], "images": ["image.png"]}),
        StagePayload("r1", OmniRequest("x", {}, {"video_fps": 2.0}), None),
        payload(stage_params=[]),
        payload(stage_params={"chat": ["bad"]}),
        payload(n=2),
        payload(stream=True, return_logprob=True),
        payload(tools=[{"function": {"name": "search"}}]),
    ],
)
def test_unsupported_contracts_reject_before_dispatch(item):
    with pytest.raises(ValueError):
        build_chat_fields(item, "cosmos3", NativeRequest.model_fields, "chat")


@pytest.mark.parametrize(
    ("configured", "reference", "outcome"),
    [
        (False, "data:image/png,AAAA", "same"),
        (False, "https://example.com/a.png", "same"),
        (False, "relative.png", "same"),
        (False, "file:///srv/a.png", "refused"),
        (True, "https://other.org/a.png", "refused"),
        (True, "https://example.com/a.png", "fetched"),
        (True, "https://example.com/unavailable.png", "server error"),
        (True, "https://example.com/missing.png", "client error"),
        (True, "file://{allowed}/a%23b%3Fc%25d.mp4", "a#b?c%d.mp4"),
        (True, "{allowed}/a#b?c%d.mp4", "a#b?c%d.mp4"),
        (True, "{outside}/a.png", "refused"),
    ],
)
def test_chat_media_follows_the_server_media_policy(
    monkeypatch, tmp_path, caplog, configured, reference, outcome
):
    import httpx

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "a#b?c%d.mp4").write_bytes(b"local video")
    reference = reference.format(allowed=allowed, outside=tmp_path)
    connection = ResourceHTTPConnection()
    connection.async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                {"/unavailable.png": 503, "/missing.png": 404}.get(
                    request.url.path, 200
                ),
                content=b"png",
                headers={"content-type": "image/png"},
            )
        )
    )
    connector = MultiModalResourceConnector(
        connection=connection,
        allowed_local_media_path=allowed if configured else None,
        allowed_media_domains=["example.com"] if configured else None,
    )
    monkeypatch.setattr(
        native_chat_scheduler, "get_global_resource_connector", lambda: connector
    )
    item = payload(inputs={"messages": [{"role": "user", "content": "x"}]})
    item.request.inputs["videos"] = [reference]
    with running() as (scheduler, _, chat):
        enqueue(scheduler, item)
        message = scheduler.outbox.get(timeout=3)
    if outcome == "refused":
        assert message.type == "error"
        assert str(message.data).startswith(native_chat_scheduler.REQUEST_ERROR)
        assert not chat.calls
    elif outcome.endswith("error"):
        # The shared classifier decides fetch failures, an upstream failure
        # is not the caller's error, and the stage logs a client error as one
        # warning line.
        status = 400 if outcome == "client error" else 500
        assert message.type == "error"
        assert generation_error_status_code(message.data) == status
        assert [
            (record.levelno, record.exc_info is not None)
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ] == [(logging.WARNING, False) if status == 400 else (logging.ERROR, True)]
        assert not chat.calls
    else:
        assert message.type == "result"
        url = chat.calls[0].messages[0]["content"][1]["video_url"]["url"]
        if outcome == "fetched":
            # Native read the fetched bytes from a file the stage then removed.
            assert chat.files == {url: b"png"}
            assert not os.path.exists(url.removeprefix("file://"))
        elif outcome == "same":
            assert url == reference
        else:
            # Native's video loader reads the handed reference back to the file.
            from sglang.srt.utils.common import get_video_bytes

            assert get_video_bytes(url) == (allowed / outcome).read_bytes()


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp_size": 2},
        {"dp_size": 2},
        {"pp_size": 2},
        {"nnodes": 2},
        {"base_gpu_id": 4},
        {"model_path": "other"},
        {"tool_call_parser": "qwen"},
    ],
)
def test_native_engine_cannot_escape_declared_placement(overrides):
    with pytest.raises(ValueError):
        native_chat_server_kwargs("checkpoint", 0, overrides)


def test_native_sizes_the_stage_unless_its_config_does():
    for overrides, budget, expected in [
        (None, None, None),
        (None, 0.4, 0.4),
        ({"mem_fraction_static": 0.5}, 0.4, 0.5),
    ]:
        kwargs = native_chat_server_kwargs("checkpoint", 0, overrides, None, budget)
        assert "context_length" not in kwargs
        assert kwargs.get("mem_fraction_static") == expected


def test_native_scheduler_receives_concurrent_requests_on_one_loop():
    with running() as (scheduler, _, chat):
        for rid in ("r1", "r2", "r3"):
            enqueue(scheduler, payload(rid))
        messages = [scheduler.outbox.get(timeout=3) for _ in range(3)]
        assert {m.request_id for m in messages} == {"r1", "r2", "r3"}
        assert all(m.type == "result" for m in messages)
        assert chat.peak > 1
        chunk = Client.default_result_builder("r1", messages[0].data.data)
        assert chunk.text == "你好"
        assert chunk.output_token_logprobs == [[-0.1, 1, "你"], [-0.2, 2, "好"]]
        assert chunk.usage.completion_tokens == 2


def test_stream_preserves_unicode_without_duplicate_terminal_text():
    with running() as (scheduler, engine, chat):
        enqueue(scheduler, payload(stream=True))
        chunks = []
        while True:
            msg = scheduler.outbox.get(timeout=3)
            if msg.type == "result":
                chunks.append(Client.default_result_builder("r1", msg.data.data))
                break
            assert msg.type == "stream"
            chunks.append(
                Client.default_stream_builder(
                    "r1",
                    StreamMessage(
                        request_id="r1",
                        from_stage="chat",
                        chunk=msg.data,
                    ),
                )
            )

        assert engine.aborts == []
        assert chat.calls[0].stream_options == {"include_usage": True}
        assert sum(chunk.finish_reason is not None for chunk in chunks) == 1
        assert chunks[-1].finish_reason == "stop"
        assert all(chunk.text for chunk in chunks[:-1])

        class StreamClient(Client):
            async def generate(self, request, request_id=None):
                for chunk in chunks:
                    yield chunk

        async def consume():
            return [
                part
                async for part in StreamClient(None).completion_stream(
                    GenerateRequest(prompt="Describe the scene", stream=True),
                    request_id="r1",
                )
            ]

        parts = asyncio.run(consume())
        assert "".join(part.text or "" for part in parts) == "你好 👩🏽‍🚀 e\u0301"
        assert parts[-1].usage.completion_tokens == 4
        assert chat.closed_streams == 1
        assert chat.background_count == 0


def test_interrupted_stream_aborts_the_native_request_at_once():
    with running() as (scheduler, engine, chat):
        enqueue(scheduler, payload(inputs="fail after first delta", stream=True))
        messages = []
        while True:
            msg = scheduler.outbox.get(timeout=3)
            messages.append(msg)
            if msg.type != "stream":
                break
        assert messages[-1].type == "error"
        assert "native failure" in str(messages[-1].data)
        assert chat.closed_streams == 1
        assert chat.background_count == 0
        assert engine.aborts == [("r1", False)]


def test_abort_reaches_native_loop_and_suppresses_terminal_result():
    with running() as (scheduler, engine, chat):
        enqueue(scheduler, payload(inputs="hang"))
        assert chat.started.wait(2)
        scheduler.abort("r1")
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.15)
        assert engine.aborts == [("r1", False)]


def test_abort_before_native_loop_runs_does_not_start_native_request():
    with running() as (scheduler, engine, chat):
        entered = threading.Event()
        release = threading.Event()

        def hold_loop():
            entered.set()
            assert release.wait(2)

        engine.loop.call_soon_threadsafe(hold_loop)
        assert entered.wait(2)
        enqueue(scheduler, payload())
        deadline = time.monotonic() + 2
        while "r1" not in scheduler.pending and time.monotonic() < deadline:
            time.sleep(0.005)
        scheduler.abort("r1")
        release.set()
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.15)
        assert not chat.calls


def test_shutdown_cancels_active_native_work_and_joins_threads():
    with running() as (scheduler, engine, chat):
        enqueue(scheduler, payload(inputs="hang"))
        assert chat.started.wait(2)
        scheduler.stop()
        assert ("", True) in engine.aborts
        assert not scheduler.loop_thread.is_alive()


def test_stage_sampling_is_preserved_and_stage_params_take_precedence():
    fields = build_chat_fields(
        payload(
            temperature=1.0,
            stage_sampling={"chat": {"temperature": 0.2}},
            stage_params={"chat": {"max_new_tokens": 9}},
        ),
        "cosmos3",
        NativeRequest.model_fields,
        "chat",
    )
    assert fields["temperature"] == 0.2
    assert fields["max_completion_tokens"] == 9


@pytest.mark.parametrize("model", ["vision", "text", "skipped"])
def test_warmup_sends_the_request_sglang_serve_sends(model):
    from sglang.srt.entrypoints.http_server import MINIMUM_PNG_PICTURE_BASE64

    engine = Engine()
    engine.server_args.skip_server_warmup = model == "skipped"
    engine.tokenizer_manager.model_config = SimpleNamespace(
        is_image_understandable_model=model != "text",
        hf_config=SimpleNamespace(architectures=["Cosmos3ForConditionalGeneration"]),
    )
    chat = Chat(engine)
    threads = []
    handle_request = chat.handle_request

    async def recorded(request, raw_request):
        threads.append(threading.current_thread())
        return await handle_request(request, raw_request)

    chat.handle_request = recorded
    # Warmup runs on the loop thread the scheduler starts, off the main thread.
    scheduler = NativeChatScheduler(engine, chat, NativeRequest, "chat")
    try:
        warm_up(engine, chat, NativeRequest)
    finally:
        scheduler.stop()
    if model == "skipped":
        assert not chat.calls
    else:
        assert threads == [scheduler.loop_thread]
        (request,) = chat.calls
        assert (request.temperature, request.max_completion_tokens) == (0.0, 8)
        content = request.messages[0]["content"]
        if model == "vision":
            assert content[0]["image_url"]["url"].endswith(MINIMUM_PNG_PICTURE_BASE64)
        else:
            assert content == "The capital city of France is"


@pytest.mark.parametrize("devices", [None, [3, 5]])
def test_factory_passes_resolved_device_to_native(monkeypatch, devices):
    import sglang

    from sglang_omni.platforms import current_platform
    from sglang_omni.utils import device as device_utils

    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=3),
    )
    pinned: list[int] = []
    monkeypatch.setattr(current_platform, "set_device", pinned.append)

    def startup(**kwargs):
        # No host CUDA context may exist while native sizes its KV pool.
        assert not pinned
        assert kwargs["base_gpu_id"] == 3
        if devices is not None:
            assert kwargs["gpu_id_step"] == 2
            assert kwargs["tp_size"] == 2
        raise RuntimeError("native startup reached")

    monkeypatch.setattr(sglang, "Engine", startup)
    with pytest.raises(RuntimeError, match="native startup reached"):
        create_native_chat_scheduler(
            "checkpoint",
            stage_name="chat",
            device=None,
            gpu_id=None,
            runtime_gpu_ids=devices,
        )


def test_factory_stops_its_scheduler_when_the_warmup_fails(monkeypatch):
    import sglang
    from sglang.srt.entrypoints.openai import serving_chat

    from sglang_omni.platforms import current_platform
    from sglang_omni.utils import device as device_utils

    engine = Engine()
    engine.server_args.skip_server_warmup = False
    engine.tokenizer_manager.model_config = SimpleNamespace(
        is_image_understandable_model=False
    )
    engine.template_manager = None

    class FailingChat:
        def __init__(self, tokenizer_manager, template_manager):
            pass

        async def handle_request(self, request, raw_request):
            return JSONResponse({"error": "warmup"}, status_code=500)

    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=0),
    )
    monkeypatch.setattr(current_platform, "set_device", lambda device: None)
    monkeypatch.setattr(sglang, "Engine", lambda **kwargs: engine)
    monkeypatch.setattr(serving_chat, "OpenAIServingChat", FailingChat)
    with pytest.raises(RuntimeError, match="warmup request failed"):
        create_native_chat_scheduler("checkpoint", stage_name="chat")
    assert engine.shutdown_count == 1
    assert engine.loop.is_closed()
