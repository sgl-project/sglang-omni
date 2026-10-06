# SPDX-License-Identifier: Apache-2.0
"""Offline batching preserves request order and generation settings."""

from __future__ import annotations

import threading
from typing import Literal, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation.utils import GenerateDecoderOnlyOutput

from sglang_omni.models.nemotron3_5_asr.model_runner import Nemotron3_5ASRModelRunner
from sglang_omni.models.nemotron3_5_asr.request_builders import Nemotron3_5ASRRequest
from sglang_omni.proto.request import OmniRequest, StagePayload


class ProcessorCall(TypedDict):
    audio: list[NDArray[np.float32]]
    sampling_rate: int
    language: list[str]
    padding: Literal["longest"]
    return_tensors: Literal["pt"]


class GenerationCall(TypedDict):
    input_features: torch.Tensor
    attention_mask: torch.Tensor
    prompt_ids: torch.Tensor
    num_lookahead_tokens: int
    return_dict_in_generate: bool
    max_new_tokens: int | None


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[ProcessorCall] = []

    def __call__(
        self,
        audio: list[NDArray[np.float32]],
        *,
        sampling_rate: int,
        language: list[str],
        padding: Literal["longest"],
        return_tensors: Literal["pt"],
    ) -> BatchFeature:
        self.calls.append(
            dict(
                audio=audio,
                sampling_rate=sampling_rate,
                language=language,
                padding=padding,
                return_tensors=return_tensors,
            )
        )
        batch_size = len(audio)
        return BatchFeature(
            {
                "input_features": torch.zeros(batch_size, 5, 4),
                "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
                "prompt_ids": torch.arange(batch_size),
                "num_lookahead_tokens": 3,
            }
        )

    def batch_decode(
        self, sequences: torch.Tensor, *, skip_special_tokens: bool
    ) -> list[str]:
        assert not skip_special_tokens
        assert sequences.device.type == "cpu"
        return ["first <en-US>", "second <zh-CN>"][: sequences.shape[0]]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[GenerationCall] = []

    def generate(
        self,
        *,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        num_lookahead_tokens: int,
        return_dict_in_generate: bool,
        max_new_tokens: int | None = None,
    ) -> GenerateDecoderOnlyOutput:
        self.calls.append(
            dict(
                input_features=input_features,
                attention_mask=attention_mask,
                prompt_ids=prompt_ids,
                num_lookahead_tokens=num_lookahead_tokens,
                return_dict_in_generate=return_dict_in_generate,
                max_new_tokens=max_new_tokens,
            )
        )
        batch_size = input_features.shape[0]
        return GenerateDecoderOnlyOutput(
            sequences=torch.arange(batch_size * 3).reshape(batch_size, 3)
        )


def make_request(request_id: str, language: str) -> Nemotron3_5ASRRequest:
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


def make_runner() -> tuple[Nemotron3_5ASRModelRunner, FakeProcessor, FakeModel]:
    runner = object.__new__(Nemotron3_5ASRModelRunner)
    processor = FakeProcessor()
    model = FakeModel()
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.processor = processor
    runner.model = model
    runner.model_lock = threading.Lock()
    return runner, processor, model


def test_run_batch_pads_once_generates_once_and_preserves_order() -> None:
    runner, processor, model = make_runner()

    results = runner.run_batch(
        [make_request("request-a", "en-US"), make_request("request-b", "zh-CN")]
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
        "first",
        "second",
    ]
    assert [result.data["raw_text"] for result in results] == [
        "first <en-US>",
        "second <zh-CN>",
    ]
    assert [result.data["language"] for result in results] == ["en-US", "zh-CN"]
    for result in results:
        assert result.data["batch_size"] == 2
        assert result.data["duration_s"] == 0.1
        assert result.data["usage"]["engine_time_s"] == result.data["model_latency_s"]


def test_batch_groups_token_limits_without_changing_request_order() -> None:
    runner, _, model = make_runner()
    requests = [make_request(name, "en-US") for name in ("a", "b", "c")]
    requests[0].max_new_tokens = requests[2].max_new_tokens = 2
    requests[1].max_new_tokens = 5
    results = runner.run_batch(requests)
    assert [result.request_id for result in results] == ["a", "b", "c"]
    assert [call["max_new_tokens"] for call in model.calls] == [2, 5]
    assert [call["input_features"].shape[0] for call in model.calls] == [2, 1]
    assert model.calls[0]["input_features"].dtype == runner.dtype
    assert model.calls[0]["prompt_ids"].dtype == torch.long
