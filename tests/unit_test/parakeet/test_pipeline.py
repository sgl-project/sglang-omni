# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.parakeet import stages as parakeet_stages
from sglang_omni.models.parakeet.config import ParakeetASRPipelineConfig
from sglang_omni.models.parakeet.request_builders import ParakeetASRRequest
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.mark.parametrize(
    "architecture", ["ParakeetForTDT", "ParakeetForRNNT", "ParakeetForCTC"]
)
def test_registry_maps_every_parakeet_head(architecture: str) -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(architecture) is ParakeetASRPipelineConfig
    )


def test_pipeline_chunks_long_audio_without_a_native_limit() -> None:
    config = ParakeetASRPipelineConfig(model_path="nvidia/parakeet-tdt-0.6b-v3")
    chunking = config.resolved_audio_chunking

    assert chunking.allow_audio_chunking is True
    assert chunking.max_native_clip_s is None
    assert chunking.max_audio_clip_s == 120.0
    assert chunking.condition_on_previous_text is False


def test_factory_receives_placement_kwargs() -> None:
    config = ParakeetASRPipelineConfig(model_path="nvidia/parakeet-ctc-0.6b")

    assert resolve_stage_factory_args(config.stages[0], config) == {
        "model_path": "nvidia/parakeet-ctc-0.6b",
        "gpu_id": 0,
    }


def test_factory_defaults() -> None:
    signature = inspect.signature(parakeet_stages.create_parakeet_asr_executor)

    assert signature.parameters["dtype"].default == "float32"
    assert signature.parameters["max_batch_size"].default == 16
    assert signature.parameters["max_batch_audio_s"].default == 600.0


def fake_platform(device_type: str) -> SimpleNamespace:
    return SimpleNamespace(device_type=device_type, is_mps=lambda: device_type == "mps")


@pytest.mark.parametrize("device_type", ["cuda", "cpu", "xpu"])
def test_factory_refuses_hosts_other_than_apple_silicon(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    monkeypatch.setattr(parakeet_stages, "current_platform", fake_platform(device_type))
    with pytest.raises(ValueError, match="only on macOS Apple Silicon"):
        parakeet_stages.create_parakeet_asr_executor("unused")


def test_factory_refuses_the_cpu_device_on_apple_silicon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parakeet_stages, "current_platform", fake_platform("mps"))
    monkeypatch.setattr(
        "sglang_omni.utils.device.resolve_concrete_device",
        lambda device, gpu_id: torch.device(device),
    )
    with pytest.raises(ValueError, match="only on the MPS device"):
        parakeet_stages.create_parakeet_asr_executor("unused", device="cpu")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_batch_wait_ms": -1.0}, "max_batch_wait_ms"),
        ({"max_batch_audio_s": float("inf")}, "max_batch_audio_s"),
    ],
)
def test_factory_rejects_invalid_batch_limits_before_loading(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object], message: str
) -> None:
    monkeypatch.setattr(parakeet_stages, "current_platform", fake_platform("mps"))
    with pytest.raises(ValueError, match=message):
        parakeet_stages.create_parakeet_asr_executor("unused", **kwargs)


def test_plan_padded_batches_groups_similar_lengths_within_budget() -> None:
    batches = parakeet_stages.plan_padded_batches(
        [10, 100, 40, 90, 10], max_padded_samples=200
    )

    assert batches == [[1, 3], [2, 0, 4]]
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3, 4]


def test_plan_padded_batches_runs_an_oversized_request_alone() -> None:
    assert parakeet_stages.plan_padded_batches(
        [500, 20, 20], max_padded_samples=100
    ) == [[0], [1, 2]]
    assert parakeet_stages.plan_padded_batches([], max_padded_samples=100) == []


def make_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id, request=OmniRequest(inputs={}), data=None
    )


def fake_request_builder(payload: StagePayload) -> ParakeetASRRequest:
    if payload.request_id.startswith("bad"):
        raise ValueError(f"Parakeet ASR could not decode {payload.request_id}")
    else:
        pass
    length = int(payload.request_id.rsplit("-", 1)[1])
    return ParakeetASRRequest(
        waveform=np.zeros(length, dtype=np.float32),
        duration_s=length / 16000,
        language=None,
        stage_payload=payload,
    )


def test_batch_fn_keeps_results_in_request_order_and_isolates_failures() -> None:
    transcribed: list[list[int]] = []

    def transcribe(waveforms):
        transcribed.append([waveform.shape[0] for waveform in waveforms])
        return [f"len={waveform.shape[0]}" for waveform in waveforms]

    batch_fn = parakeet_stages.make_parakeet_batch_fn(
        request_builder=fake_request_builder,
        transcribe=transcribe,
        max_padded_samples=200,
    )
    results = batch_fn(
        [
            make_payload("ok-10"),
            make_payload("bad-1"),
            make_payload("ok-100"),
            make_payload("ok-90"),
        ]
    )

    assert transcribed == [[100, 90], [10]]
    assert [getattr(result, "request_id", None) for result in results] == [
        "ok-10",
        None,
        "ok-100",
        "ok-90",
    ]
    assert isinstance(results[1], ValueError)
    assert [results[i].data["text"] for i in (0, 2, 3)] == [
        "len=10",
        "len=100",
        "len=90",
    ]


def test_batch_fn_fails_only_the_group_whose_forward_raised() -> None:
    def transcribe(waveforms):
        if any(waveform.shape[0] == 100 for waveform in waveforms):
            raise RuntimeError("MPS out of memory")
        else:
            return ["ok"] * len(waveforms)

    batch_fn = parakeet_stages.make_parakeet_batch_fn(
        request_builder=fake_request_builder,
        transcribe=transcribe,
        max_padded_samples=100,
    )
    results = batch_fn([make_payload("ok-100"), make_payload("ok-20")])

    assert isinstance(results[0], RuntimeError)
    assert results[1].data["text"] == "ok"
