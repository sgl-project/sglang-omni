# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sglang_omni.models.nemotron3_5_asr import stages
from sglang_omni.models.nemotron3_5_asr.config import (
    Nemotron3_5ASRFactoryArgs,
    Nemotron3_5ASRPipelineConfig,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler


def test_config_registers_one_terminal_non_engine_stage() -> None:
    config = Nemotron3_5ASRPipelineConfig(model_path="nvidia/nemotron")

    assert config.entry_stage == "asr"
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert len(config.stages) == 1
    stage = config.stages[0]
    assert not type(stage).engine_stage
    assert stage.factory.num_lookahead_tokens == 3
    assert stage.factory.dtype == "float32"
    assert stage.factory.max_batch_size == 8
    assert stage.factory.max_batch_wait_ms == 2.0
    assert stage.factory.max_pending_stream_messages == 256
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Nemotron3_5AsrForRNNT")
        is Nemotron3_5ASRPipelineConfig
    )


def test_config_rejects_lookahead_not_supported_by_the_pinned_checkpoint() -> None:
    with pytest.raises(ValidationError, match="num_lookahead_tokens"):
        Nemotron3_5ASRFactoryArgs(num_lookahead_tokens=1)


def test_stage_factory_uses_serial_true_batch_scheduler(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeRunner:
        prompt_dictionary = {"auto": 101, "en-US": 0}

        streaming_chunk_spec = {
            "sample_rate": 16000,
            "first_samples": 4,
            "subsequent_samples": 8,
            "first_frames": 1,
            "subsequent_frames": 2,
            "hop_length": 2,
            "n_fft": 4,
            "streaming_latency_ms": 10,
        }

        def __init__(self, model_path, **kwargs):
            calls["load"] = (model_path, kwargs)

        def new_streaming_decode_state(self):
            return SimpleNamespace(tokens=[0], durations=[0])

        def run_one(self, request):
            calls["one"] = request
            return request

        def run_batch(self, requests):
            calls["batch"] = list(requests)
            return list(requests)

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(stages, "Nemotron3_5ASRModelRunner", FakeRunner)
    monkeypatch.setattr(
        stages,
        "make_nemotron3_5_asr_request_builder",
        lambda **kwargs: lambda payload: SimpleNamespace(payload=payload),
    )

    scheduler = stages.create_nemotron3_5_asr_executor(
        "checkpoint",
        device="cpu",
        max_batch_size=4,
        max_batch_wait_ms=5,
    )

    assert isinstance(scheduler, StreamingSimpleScheduler)
    assert scheduler._batch_fn is not None
    assert scheduler._max_batch_size == 4
    assert scheduler._max_batch_wait_s == 0.005
    assert scheduler._fn("one").payload == "one"
    assert [item.payload for item in scheduler._batch_fn(["a", "b"])] == ["a", "b"]
    assert len(calls["batch"]) == 2

    scheduler.stop()
    assert calls["closed"] is True


def test_batch_request_build_failure_does_not_poison_valid_peer(monkeypatch) -> None:
    class FakeRunner:
        prompt_dictionary = {"auto": 101}
        streaming_chunk_spec = {
            "sample_rate": 16000,
            "first_samples": 4,
            "subsequent_samples": 8,
            "first_frames": 1,
            "subsequent_frames": 2,
            "hop_length": 2,
            "n_fft": 4,
            "streaming_latency_ms": 10,
        }

        def new_streaming_decode_state(self):
            return SimpleNamespace(tokens=[0], durations=[0])

        def __init__(self, *args, **kwargs):
            pass

        def run_one(self, request):
            return request

        def run_batch(self, requests):
            return [f"result:{request}" for request in requests]

        def close(self):
            pass

    def build(payload):
        if payload == "bad":
            raise ValueError("bad language")
        return payload

    monkeypatch.setattr(stages, "Nemotron3_5ASRModelRunner", FakeRunner)
    monkeypatch.setattr(
        stages,
        "make_nemotron3_5_asr_request_builder",
        lambda **kwargs: build,
    )
    scheduler = stages.create_nemotron3_5_asr_executor(
        "checkpoint", device="cpu", max_batch_size=2
    )
    loop = asyncio.new_event_loop()
    try:
        scheduler._run_non_streaming_batch(
            [
                IncomingMessage("good", "new_request", "good"),
                IncomingMessage("bad", "new_request", "bad"),
            ],
            loop,
        )
    finally:
        loop.close()
    emitted = [scheduler.outbox.get_nowait() for _ in range(2)]
    outputs = {message.request_id: message for message in emitted}

    assert outputs["good"].type == "result"
    assert outputs["good"].data == "result:good"
    assert outputs["bad"].type == "error"
    assert isinstance(outputs["bad"].data, ValueError)
