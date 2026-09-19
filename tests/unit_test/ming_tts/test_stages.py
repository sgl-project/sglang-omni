# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest


def test_ming_tts_audio_decode_factory_exposes_only_supported_contract() -> None:
    from sglang_omni.models.ming_tts.config import (
        MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
        MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
        MING_TTS_DEFAULT_STREAM_SLOTS,
        MING_TTS_DEFAULT_STREAMING_CUDA_GRAPH,
    )
    from sglang_omni.models.ming_tts.stages import create_audio_decode_executor

    parameters = inspect.signature(create_audio_decode_executor).parameters

    assert "decode_mode" not in parameters
    assert (
        parameters["initial_chunk_patches"].default
        == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    )
    assert (
        parameters["steady_chunk_patches"].default
        == MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    )
    assert (
        parameters["streaming_cuda_graph"].default
        is MING_TTS_DEFAULT_STREAMING_CUDA_GRAPH
    )
    assert parameters["stream_slots"].default == MING_TTS_DEFAULT_STREAM_SLOTS
    assert parameters["max_batch_size"].default == 1
    assert parameters["max_batch_wait_ms"].default == 0


def test_ming_tts_legacy_engine_factory_alias_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.ming_tts import stages

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()

    def fake_factory(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(stages, "create_sglang_tts_engine_executor", fake_factory)

    result = stages.create_tts_engine_executor("model", gpu_id=2)

    assert result is sentinel
    assert calls == [(("model",), {"gpu_id": 2})]


@pytest.mark.parametrize(
    ("factory_args", "error"),
    [
        ({"stream_slots": 0}, "stream_slots must be a positive integer"),
        ({"max_batch_size": 2}, "max_batch_size=1 only"),
        ({"max_batch_wait_ms": 1}, "max_batch_wait_ms=0 only"),
    ],
)
def test_ming_tts_audio_decode_factory_rejects_batch_config_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    factory_args: dict[str, int],
    error: str,
) -> None:
    from sglang_omni.models.ming_tts import stages

    def fail_if_called(model_path: str) -> str:
        raise AssertionError(f"unexpected checkpoint resolution for {model_path}")

    monkeypatch.setattr(stages, "_resolve_checkpoint", fail_if_called)

    with pytest.raises(ValueError, match=error):
        stages.create_audio_decode_executor("unused", **factory_args)


def _patch_audio_decode_factory_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    warmup_error: Exception | None = None,
) -> tuple[Any, list[dict[str, Any]], list[Any]]:
    import torch

    from sglang_omni.models.ming_tts import audio_decode, stages, streaming_vocoder

    decoder_calls: list[dict[str, Any]] = []
    schedulers: list[Any] = []

    class FakeDecoder:
        sample_rate = 44100

        def __init__(self, audio_vae: object, **kwargs: Any) -> None:
            del audio_vae
            decoder_calls.append(kwargs)

        def close(self) -> None:
            return None

    class FakeScheduler:
        def __init__(self, decoder: object, **kwargs: Any) -> None:
            del decoder, kwargs
            self.warmup_calls = 0
            self.stop_calls = 0
            schedulers.append(self)

        def warmup_now(self) -> None:
            self.warmup_calls += 1
            if warmup_error is not None:
                raise warmup_error

        def stop(self) -> None:
            self.stop_calls += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    # note (lennox): the factory resolves device through the platform layer now,
    # which CI's hidden-CUDA job would report as cpu; pin it like torch.cuda above.
    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cuda", raising=False)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda _: "checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_ming_tts_config",
        lambda _: SimpleNamespace(
            audio_tokenizer_config=object(),
            audio_patch_size=2,
            latent_dim=4,
        ),
    )
    monkeypatch.setattr(
        stages,
        "resolve_ming_tts_audio_vae_config",
        lambda *_args, **_kwargs: SimpleNamespace(dec_kwargs={"latent_dim": 4}),
    )
    monkeypatch.setattr(stages, "_load_ming_tts_audio_vae", lambda *_a, **_k: object())
    monkeypatch.setattr(
        stages,
        "get_gpu_device_info",
        lambda _: SimpleNamespace(total_memory_bytes=1),
    )
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda _: 0)
    monkeypatch.setattr(audio_decode, "MingAudioDecoder", FakeDecoder)
    monkeypatch.setattr(
        streaming_vocoder,
        "MingTTSStreamingVocoderScheduler",
        FakeScheduler,
    )
    return stages, decoder_calls, schedulers


def test_ming_tts_audio_decode_factory_binds_the_placed_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages, _decoder_calls, _schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    vae_loads: list[dict] = []
    monkeypatch.setattr(
        stages,
        "_load_ming_tts_audio_vae",
        lambda *args, **kwargs: vae_loads.append(kwargs) or object(),
    )

    stages.create_audio_decode_executor("unused", gpu_id=3)

    assert vae_loads[0]["device"] == torch.device("cuda", 3)


def _patch_xpu_device_module(
    monkeypatch: pytest.MonkeyPatch,
    **members: Any,
) -> None:
    """Route torch.get_device_module("xpu") to a fake module, other devices unchanged."""
    import torch

    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "xpu", raising=False)
    members.setdefault("device", lambda _device: nullcontext())
    members.setdefault(
        "current_stream",
        lambda _device: SimpleNamespace(synchronize=lambda: None),
    )
    fake_module = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 4,
        **members,
    )
    real_get_device_module = torch.get_device_module
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: (
            fake_module
            if torch.device(device).type == "xpu"
            else real_get_device_module(device)
        ),
    )


def test_ming_tts_audio_decode_factory_serves_a_non_cuda_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-CUDA accelerator is not refused: the factory asks the device's own module."""
    stages, _decoder_calls, _schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )
    import torch

    _patch_xpu_device_module(monkeypatch)
    vae_loads: list[dict] = []
    monkeypatch.setattr(
        stages,
        "_load_ming_tts_audio_vae",
        lambda *args, **kwargs: vae_loads.append(kwargs) or object(),
    )

    stages.create_audio_decode_executor("unused", device="xpu", gpu_id=1)

    assert vae_loads[0]["device"] == torch.device("xpu", 1)


def test_ming_tts_audio_decode_factory_rejects_a_device_it_cannot_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unservable is availability and visibility now, not vendor."""
    stages, _decoder_calls, _schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )
    import torch

    with pytest.raises(ValueError, match="requires an available accelerator"):
        stages.create_audio_decode_executor("unused", device="cpu", gpu_id=1)

    with pytest.raises(ValueError, match="is not visible"):
        stages.create_audio_decode_executor("unused", gpu_id=5)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires an available accelerator"):
        stages.create_audio_decode_executor("unused", gpu_id=0)


def test_ming_tts_audio_decode_factory_rejects_an_accelerator_without_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Available and visible is not enough: mps has no device context and no stream."""
    stages, _decoder_calls, _schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )
    import torch

    from sglang_omni.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "mps", raising=False)
    monkeypatch.setattr(
        stages,
        "_resolve_checkpoint",
        lambda path: pytest.fail(f"unexpected checkpoint resolution for {path}"),
    )
    streamless = SimpleNamespace(
        __name__="torch.mps",
        is_available=lambda: True,
        device_count=lambda: 1,
        synchronize=lambda: None,
    )
    real_get_device_module = torch.get_device_module
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: (
            streamless
            if torch.device(device).type == "mps"
            else real_get_device_module(device)
        ),
    )

    with pytest.raises(ValueError, match="has no torch.mps.device"):
        stages.create_audio_decode_executor("unused", device="mps", gpu_id=0)


def _patch_accelerator_off_cuda(
    monkeypatch: pytest.MonkeyPatch,
    stages: Any,
    *,
    total_memory_bytes: int | None = None,
    **members: Any,
) -> None:
    """An xpu device whose accounting must never reach the NVML-mapped helpers.

    Both of those resolve their argument through CUDA_VISIBLE_DEVICES, so on a
    host with NVML alongside xpu they would answer for an unrelated CUDA card.
    """

    def fail_if_called(logical_gpu_id: int) -> None:
        raise AssertionError(
            f"NVML was asked about logical gpu {logical_gpu_id} off CUDA"
        )

    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", fail_if_called)
    monkeypatch.setattr(stages, "get_gpu_device_info", fail_if_called)
    if total_memory_bytes is not None:
        members["get_device_properties"] = lambda _device: SimpleNamespace(
            total_memory=total_memory_bytes
        )
    _patch_xpu_device_module(monkeypatch, **members)


def test_ming_tts_audio_decode_factory_enforces_the_budget_without_nvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No NVML is not no budget: the accelerator's own allocator carries the check."""
    stages, _decoder_calls, _schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )
    _patch_accelerator_off_cuda(
        monkeypatch,
        stages,
        total_memory_bytes=8 << 30,
        memory_reserved=lambda _device: 6 << 30,
    )

    with pytest.raises(RuntimeError, match="exceeds its cumulative budget"):
        stages.create_audio_decode_executor(
            "unused",
            device="xpu",
            gpu_id=1,
            process_total_gpu_memory_fraction=0.5,
        )


def test_ming_tts_audio_decode_factory_records_where_the_bytes_came_from(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Allocator bytes under-report the driver context, so the log names the source."""
    stages, _decoder_calls, _schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )
    _patch_accelerator_off_cuda(
        monkeypatch,
        stages,
        total_memory_bytes=8 << 30,
        memory_reserved=lambda _device: 1 << 30,
    )

    with caplog.at_level(logging.INFO, logger=stages.__name__):
        stages.create_audio_decode_executor(
            "unused",
            device="xpu",
            gpu_id=1,
            process_total_gpu_memory_fraction=0.5,
        )

    assert "memory_source=torch_allocator memory_verification=verified" in caplog.text


def test_ming_tts_audio_decode_factory_warns_when_nothing_can_account(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A backend with neither NVML nor a caching allocator still degrades loudly."""
    stages, _decoder_calls, _schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )
    _patch_accelerator_off_cuda(monkeypatch, stages, total_memory_bytes=8 << 30)

    with caplog.at_level(logging.WARNING, logger=stages.__name__):
        stages.create_audio_decode_executor(
            "unused",
            device="xpu",
            gpu_id=1,
            process_total_gpu_memory_fraction=0.5,
        )

    assert "memory_verification=unavailable memory_source=unavailable" in caplog.text


def test_ming_tts_audio_decode_factory_keeps_nvml_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CUDA keeps NVML, which counts the driver context the allocator cannot see."""
    stages, _decoder_calls, _schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )
    monkeypatch.setattr(
        stages,
        "get_gpu_device_info",
        lambda _: SimpleNamespace(total_memory_bytes=100),
    )
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda _: 7)
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda _device: (_ for _ in ()).throw(
            AssertionError("the allocator answered for a device NVML already had")
        ),
    )

    with caplog.at_level(logging.INFO, logger=stages.__name__):
        stages.create_audio_decode_executor(
            "unused",
            process_total_gpu_memory_fraction=0.5,
        )

    assert (
        "device_total_bytes=100 process_budget_bytes=50 memory_source=nvml"
        in caplog.text
    )


@pytest.mark.parametrize(
    ("streaming_cuda_graph", "process_fraction"),
    [(False, None), (True, 1.0)],
)
def test_ming_tts_audio_decode_factory_forwards_backend_and_warms_up(
    monkeypatch: pytest.MonkeyPatch,
    streaming_cuda_graph: bool,
    process_fraction: float | None,
) -> None:
    stages, decoder_calls, schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch
    )

    result = stages.create_audio_decode_executor(
        "unused",
        stream_slots=3,
        streaming_cuda_graph=streaming_cuda_graph,
        process_total_gpu_memory_fraction=process_fraction,
    )

    assert len(decoder_calls) == 1
    assert decoder_calls[0]["stream_capacity"] == 3
    assert decoder_calls[0]["max_stream_step_latents"] == 8
    assert decoder_calls[0]["streaming_cuda_graph_required"] is streaming_cuda_graph
    assert result is schedulers[0]
    assert schedulers[0].warmup_calls == 1
    assert schedulers[0].stop_calls == 0


def test_ming_tts_audio_decode_factory_stops_scheduler_when_warmup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages, _decoder_calls, schedulers = _patch_audio_decode_factory_dependencies(
        monkeypatch,
        warmup_error=RuntimeError("warmup failed"),
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        stages.create_audio_decode_executor("unused")

    assert schedulers[0].warmup_calls == 1
    assert schedulers[0].stop_calls == 1


@pytest.mark.parametrize("field", ["initial_chunk_patches", "steady_chunk_patches"])
@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_audio_decode_factory_rejects_invalid_cadence_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    from sglang_omni.models.ming_tts import stages

    def fail_if_called(model_path: str) -> str:
        raise AssertionError(f"unexpected checkpoint resolution for {model_path}")

    monkeypatch.setattr(stages, "_resolve_checkpoint", fail_if_called)

    with pytest.raises(
        ValueError,
        match=f"{field} must be a positive integer",
    ):
        stages.create_audio_decode_executor(
            "unused",
            **{field: value},
        )
