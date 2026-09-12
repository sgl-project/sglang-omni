# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.cosmos3.config import Cosmos3ReasonerPipelineConfig, Variants
from sglang_omni.models.cosmos3.reasoner import (
    NativeReasonerScheduler,
    build_chat_fields,
    native_reasoner_kwargs,
)
from sglang_omni.proto import OmniRequest, StagePayload, StreamMessage
from sglang_omni.scheduling.messages import IncomingMessage


class NativeRequest(SimpleNamespace):
    model_fields = {
        "model",
        "messages",
        "rid",
        "n",
        "temperature",
        "seed",
        "max_tokens",
        "stream",
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
        self.started = threading.Event()
        self.active = 0
        self.peak = 0
        self.closed_streams = 0
        self.background_count = 0

    async def handle_request(self, request, raw_request):
        assert asyncio.get_running_loop() is self.engine.loop
        assert raw_request is None
        self.calls.append(request)
        self.started.set()
        message = request.messages[0]
        content = message["content"] if isinstance(message, dict) else message.content
        if content == "hang":
            await asyncio.Event().wait()
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
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
                    yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
                    yield 'data: {"choices": [], "usage": {"completion_tokens": 4}}\n\n'
                    yield "data: [DONE]\n\n"
                finally:
                    self.closed_streams += 1

            async def background():
                self.background_count += 1

            return SimpleNamespace(body_iterator=frames(), background=background)
        return {
            "choices": [
                {
                    "message": {"content": "你好"},
                    "finish_reason": "length",
                    "response_token_ids": [1, 2],
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, 1, "你"], [-0.2, 2, "好"]]
                    },
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        }


@contextmanager
def running(*, request_type=NativeRequest):
    engine = Engine()
    chat = Chat(engine)
    scheduler = NativeReasonerScheduler(engine, chat, request_type, max_concurrency=4)
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        yield scheduler, engine, chat
    finally:
        scheduler.stop()
        scheduler.stop()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not scheduler._loop_thread.is_alive()
        assert engine.loop.is_closed()
        assert engine.shutdown_count == 1


def enqueue(scheduler, item):
    scheduler.enqueue(IncomingMessage(item.request_id, "new_request", item))


@pytest.mark.parametrize(
    ("raw_name", "resolved_name"),
    [(None, "/models/Cosmos3/transformer"), ("cosmos-alias", "cosmos-alias")],
)
def test_native_chat_request_uses_resolved_serving_model_name(raw_name, resolved_name):
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    with running(request_type=ChatCompletionRequest) as (scheduler, engine, chat):
        engine.server_args.served_model_name = raw_name
        engine.tokenizer_manager.served_model_name = resolved_name
        enqueue(scheduler, payload(response_format={"type": "json_object"}))
        result = scheduler.outbox.get(timeout=3)
        assert result.type == "result"
        request = chat.calls[0]
        assert isinstance(request, ChatCompletionRequest)
        assert request.model == resolved_name
        assert request.response_format.type == "json_object"
        assert engine.server_args.served_model_name == raw_name


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
        "images": ["data:image/png;base64,AAAA"],
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
    )
    assert (
        fields["messages"][0]["content"][1]["image_url"]["url"] == original["images"][0]
    )
    assert original["messages"][0]["content"] == "What is here?"
    assert (fields["temperature"], fields["seed"], fields["max_tokens"]) == (0.0, 0, 7)
    assert fields["return_meta_info"] and fields["return_token_ids"]
    assert fields["rid"] == "r1"


@pytest.mark.parametrize(
    "item",
    [
        payload(inputs=[]),
        payload(inputs=[1, 2]),
        payload(inputs={"messages": [], "images": ["image.png"]}),
        payload(stage_params=[]),
        payload(stage_sampling=[]),
        payload(stage_params={"reasoner": ["bad"]}),
        payload(n=2),
        payload(stream=True, return_logprob=True),
        payload(tools=[{"function": {"name": "search"}}]),
    ],
)
def test_unsupported_contracts_reject_before_dispatch(item):
    with pytest.raises(ValueError):
        build_chat_fields(item, "cosmos3", NativeRequest.model_fields)


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
        native_reasoner_kwargs("checkpoint", 0, overrides)


def test_native_scheduler_receives_concurrent_requests_on_one_loop():
    with running() as (scheduler, _, chat):
        for rid in ("r1", "r2", "r3"):
            enqueue(scheduler, payload(rid))
        messages = [scheduler.outbox.get(timeout=3) for _ in range(3)]
        assert {m.request_id for m in messages} == {"r1", "r2", "r3"}
        assert all(m.type == "result" for m in messages)
        assert chat.peak > 1
        chunk = Client._default_result_builder("r1", messages[0].data.data)
        assert chunk.text == "你好"
        assert chunk.output_token_logprobs == [[-0.1, 1, "你"], [-0.2, 2, "好"]]
        assert chunk.usage.completion_tokens == 2


def test_stream_preserves_unicode_without_duplicate_terminal_text():
    with running() as (scheduler, _, chat):
        enqueue(scheduler, payload(stream=True))
        chunks = []
        while True:
            msg = scheduler.outbox.get(timeout=3)
            if msg.type == "result":
                chunks.append(Client._default_result_builder("r1", msg.data.data))
                break
            assert msg.type == "stream"
            chunks.append(
                Client._default_stream_builder(
                    "r1",
                    StreamMessage(
                        request_id="r1",
                        from_stage="reasoner",
                        chunk=msg.data,
                    ),
                )
            )

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
        assert chat.closed_streams == chat.background_count == 1


def test_abort_reaches_native_loop_and_suppresses_terminal_result():
    with running() as (scheduler, engine, chat):
        enqueue(scheduler, payload(inputs="hang"))
        assert chat.started.wait(2)
        scheduler.abort("r1")
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.15)
        assert ("r1", False) in engine.aborts


def test_abort_during_executor_dispatch_does_not_start_native_request():
    with running() as (scheduler, _, chat):
        entered = threading.Event()
        release = threading.Event()
        original = scheduler._fn

        def delayed(item):
            entered.set()
            assert release.wait(2)
            return original(item)

        scheduler._fn = delayed
        enqueue(scheduler, payload())
        assert entered.wait(2)
        scheduler.abort("r1")
        release.set()
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.15)
        assert not chat.calls


def test_queued_abort_never_dispatches():
    with running() as (scheduler, _, chat):
        scheduler.abort("r1")
        enqueue(scheduler, payload())
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.15)
        assert not chat.calls


def test_shutdown_cancels_active_native_work_and_joins_threads():
    with running() as (scheduler, engine, chat):
        enqueue(scheduler, payload(inputs="hang"))
        assert chat.started.wait(2)
        scheduler.stop()
        assert ("", True) in engine.aborts
        assert not scheduler._loop_thread.is_alive()


def test_reasoner_variant_uses_native_child_process_ownership():
    config = Cosmos3ReasonerPipelineConfig(model_path="checkpoint")
    assert config.native_media_stage is None
    assert config.stages[0].allow_child_processes
    assert config.stages[0].name == "reasoner"
    assert Variants["text"] is Cosmos3ReasonerPipelineConfig


def test_stage_sampling_is_preserved_and_stage_params_take_precedence():
    fields = build_chat_fields(
        payload(
            temperature=1.0,
            stage_sampling={"reasoner": {"temperature": 0.2}},
            stage_params={"reasoner": {"max_new_tokens": 9}},
        ),
        "cosmos3",
        NativeRequest.model_fields,
    )
    assert fields["temperature"] == 0.2
    assert fields["max_tokens"] == 9
