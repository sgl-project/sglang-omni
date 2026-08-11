# SPDX-License-Identifier: Apache-2.0
"""ServerArgs must carry the platform-resolved device.

SGLang picks a backend off a CUDA-first ladder that consults the platform layer
last, so an unset device can contradict the platform Omni resolved.
"""

from __future__ import annotations

from typing import Any

import sglang_omni.platforms as platforms
from sglang_omni.scheduling.sglang_backend import server_args_builder


class _CapturedServerArgs:
    """Stands in for ServerArgs so no HF checkpoint is needed."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.enable_dp_attention = False


def _build(monkeypatch, **extra: Any) -> dict[str, Any]:
    monkeypatch.setattr(server_args_builder, "ServerArgs", _CapturedServerArgs)
    built = server_args_builder.build_sglang_server_args(
        "model", context_length=128, **extra
    )
    return built.kwargs


def test_unset_device_falls_back_to_the_resolved_platform(monkeypatch) -> None:
    assert _build(monkeypatch)["device"] == platforms.current_platform.device_type


def test_pinned_platform_reaches_server_args(monkeypatch) -> None:
    """A pinned platform must win over SGLang's own detection."""
    monkeypatch.setattr(platforms.current_platform, "device_type", "xpu", raising=False)
    assert _build(monkeypatch)["device"] == "xpu"


def test_caller_resolved_device_is_not_overwritten(monkeypatch) -> None:
    """A cpu stage on an accelerator host keeps cpu, index-free."""
    monkeypatch.setattr(platforms.current_platform, "device_type", "xpu", raising=False)
    assert _build(monkeypatch, device="cpu")["device"] == "cpu"


def _drive_build(monkeypatch, *, overrides, gpu_id=0):
    """Run SGLangGenerationEngineBuilder.build() far enough to reach the device
    reconciliation, then stop. Returns the device handed to SGLang.
    """
    import pytest

    from sglang_omni.scheduling import engine_factory

    captured: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def fake_server_args(_checkpoint, **kwargs):
        captured.update(kwargs)
        raise _Stop

    # build() imports sglang_backend locally, so patch the source module.
    from sglang_omni.scheduling import sglang_backend

    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", fake_server_args)

    class _Builder(engine_factory.SGLangGenerationEngineBuilder):
        model_name = "probe"
        context_length = 16

        def generation_defaults(self, *, dtype):
            del dtype
            return {"max_running_requests": 4}

        def make_adapters(self, *args, **kwargs):
            raise NotImplementedError

        def make_model_runner(self, *args, **kwargs):
            raise NotImplementedError

    builder = _Builder()
    with pytest.raises((_Stop, ValueError)) as raised:
        builder.build(
            "unused", device=None, gpu_id=gpu_id, server_args_overrides=overrides
        )
    if isinstance(raised.value, ValueError):
        raise raised.value
    return captured.get("device")


def test_an_operator_device_that_agrees_with_placement_is_passed_through(
    monkeypatch,
) -> None:
    resolved = platforms.current_platform.device_type

    assert _drive_build(monkeypatch, overrides={"device": resolved}) == resolved


def test_an_operator_device_that_contradicts_placement_is_rejected(monkeypatch) -> None:
    """setdefault() used to leave the override in place, so SGLang could run on a
    different device than the one this stage was placed on.
    """
    import pytest

    resolved = platforms.current_platform.device_type
    other = "cuda" if resolved != "cuda" else "xpu"

    with pytest.raises(ValueError, match="Omni owns placement"):
        _drive_build(monkeypatch, overrides={"device": other})


def test_placement_supplies_the_device_when_no_override_is_given(monkeypatch) -> None:
    resolved = platforms.current_platform.device_type

    assert _drive_build(monkeypatch, overrides=None) == resolved
