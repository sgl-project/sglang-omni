# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sglang_omni.models.nemotron3_5_asr import request_builders, stages
from sglang_omni.models.nemotron3_5_asr.config import Nemotron3_5ASRPipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


@pytest.fixture(autouse=True)
def torch_backend(monkeypatch):
    from sglang.srt.hardware_backend.mlx import runtime

    monkeypatch.setattr(runtime, "use_mlx", lambda: False)


def test_config_leaves_defaults_to_factory() -> None:
    config = Nemotron3_5ASRPipelineConfig(model_path="checkpoint")

    assert config.entry_stage == "asr"
    assert config.terminal_stages == ["asr"]
    assert len(config.stages) == 1
    assert config.stages[0].factory.model_dump(exclude_none=True) == {}
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Nemotron3_5AsrForRNNT")
        is Nemotron3_5ASRPipelineConfig
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("model_error", [False, True])
@pytest.mark.parametrize(
    "request_languages",
    [
        [("a", "auto")],
        [("a", "auto"), ("bad", "unknown"), ("b", "auto")],
    ],
    ids=["single", "mixed_batch"],
)
def test_factory_transcribes_single_and_batched_requests(
    monkeypatch: pytest.MonkeyPatch,
    model_error: bool,
    device: str,
    request_languages: list[tuple[str, str]],
) -> None:
    runner = Mock(spec=stages.Nemotron3_5ASRModelRunner)
    runner.prompt_dictionary = {"auto": 101}
    runner.streaming_chunk_spec = dict(
        sample_rate=16000,
        first_samples=4,
        subsequent_samples=8,
        first_frames=1,
        subsequent_frames=2,
        hop_length=2,
        n_fft=4,
        streaming_latency_ms=10,
    )
    runner.run_batch.side_effect = (
        RuntimeError("model failed")
        if model_error
        else lambda requests: [request.stage_payload for request in requests]
    )
    runner_factory = Mock(return_value=runner)
    monkeypatch.setattr(stages, "Nemotron3_5ASRModelRunner", runner_factory)
    monkeypatch.setattr(
        request_builders,
        "prepare_audio",
        lambda *args, **kwargs: SimpleNamespace(
            waveform=np.zeros(1600, dtype=np.float32),
            duration_s=0.1,
        ),
    )
    monkeypatch.setattr(stages, "resolve_device_spec", lambda *args: device)
    scheduler = stages.create_nemotron3_5_asr_executor("checkpoint", device=device)
    assert runner_factory.call_args.kwargs["device"] == device
    assert runner_factory.call_args.kwargs["dtype"] == "float32"
    assert scheduler.inbox.maxsize == 256
    payloads = [
        StagePayload(
            request_id=name,
            request=OmniRequest(inputs=b"audio", params={"language": language}),
            data=None,
        )
        for name, language in request_languages
    ]
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_non_streaming_batch(
            [
                IncomingMessage(payload.request_id, "new_request", payload)
                for payload in payloads
            ],
            loop,
        )
    finally:
        loop.close()
        scheduler.stop()
    outputs = {
        message.request_id: message
        for message in [scheduler.outbox.get_nowait() for _ in payloads]
    }
    for name, language in request_languages:
        if language != "auto":
            assert isinstance(outputs[name].data, ValueError)
            assert outputs[name].type == "error"
            continue
        assert outputs[name].type == ("error" if model_error else "result")
        if model_error:
            assert str(outputs[name].data) == "model failed"
        else:
            assert outputs[name].data.request_id == name
    assert [
        request.stage_payload.request_id
        for request in runner.run_batch.call_args.args[0]
    ] == [name for name, language in request_languages if language == "auto"]
    runner.close.assert_called_once()
