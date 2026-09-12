# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.nemotron3_5_asr.model_runner import Nemotron3_5ASRModelRunner
from sglang_omni.models.nemotron3_5_asr.request_builders import Nemotron3_5ASRRequest
from sglang_omni.proto import OmniRequest, StagePayload


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        batch_size = len(audio)
        return {
            "input_features": torch.zeros(batch_size, 5, 4),
            "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
            "prompt_ids": torch.arange(batch_size),
            "num_lookahead_tokens": 3,
        }

    def batch_decode(self, sequences, **kwargs):
        assert kwargs == {"skip_special_tokens": False}
        assert sequences.device.type == "cpu"
        return ["first <en-US>", "second <zh-CN>"][: sequences.shape[0]]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["input_features"].shape[0]
        return SimpleNamespace(
            sequences=torch.arange(batch_size * 3).reshape(batch_size, 3)
        )


def _request(request_id: str, language: str) -> Nemotron3_5ASRRequest:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=b"audio"),
        data=None,
    )
    return Nemotron3_5ASRRequest(
        waveform=np.zeros(1600, dtype=np.float32),
        duration_s=0.1,
        language=language,
        stage_payload=payload,
    )


def _runner() -> tuple[Nemotron3_5ASRModelRunner, FakeProcessor, FakeModel]:
    runner = object.__new__(Nemotron3_5ASRModelRunner)
    processor = FakeProcessor()
    model = FakeModel()
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.processor = processor
    runner.model = model
    runner._model_lock = threading.Lock()
    return runner, processor, model


def test_run_batch_pads_once_generates_once_and_preserves_order() -> None:
    runner, processor, model = _runner()

    results = runner.run_batch(
        [_request("request-a", "en-US"), _request("request-b", "zh-CN")]
    )

    assert len(processor.calls) == 1
    call = processor.calls[0]
    assert call["sampling_rate"] == 16000
    assert call["language"] == ["en-US", "zh-CN"]
    assert call["padding"] == "longest"
    assert call["return_tensors"] == "pt"
    assert len(model.calls) == 1
    assert model.calls[0]["input_features"].shape[0] == 2
    assert model.calls[0]["return_dict_in_generate"] is True
    assert [result.request_id for result in results] == ["request-a", "request-b"]
    assert [result.data["text"] for result in results] == [
        "first <en-US>",
        "second <zh-CN>",
    ]
    assert all(result.data["batch_size"] == 2 for result in results)


def test_run_one_delegates_to_run_batch(monkeypatch) -> None:
    runner, _, _ = _runner()
    request = _request("request-a", "en-US")
    expected = request.stage_payload
    calls: list[list[Nemotron3_5ASRRequest]] = []

    def fake_run_batch(requests):
        calls.append(list(requests))
        return [expected]

    monkeypatch.setattr(runner, "run_batch", fake_run_batch)

    assert runner.run_one(request) is expected
    assert calls == [[request]]
