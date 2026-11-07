# SPDX-License-Identifier: Apache-2.0
"""Unit tests for device-spec retargeting (no accelerator required)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import sglang_omni.platforms.xpu as xpu_platform
import sglang_omni.utils.device as dev
from sglang_omni.platforms import current_platform


def test_none_is_the_only_automatic_selection():
    """Auto-selection is opt-in: only None consults current_platform, so a supplied
    device can never be silently retargeted."""
    live = current_platform.device_type

    assert dev.resolve_device_spec(None) == live
    assert dev.resolve_device_spec(None, 3) == ("cpu" if live == "cpu" else f"{live}:3")


def test_a_supplied_device_is_honored_with_its_placement_index():
    """The caller's explicit device type survives, and the index argument
    decides its number, including cpu and any live accelerator."""
    live = current_platform.device_type
    assert dev.resolve_device_spec("cpu") == "cpu"
    assert dev.resolve_device_spec("cpu", 5) == "cpu"
    if live == "cpu":
        return
    assert dev.resolve_device_spec(live) == live
    assert dev.resolve_device_spec(live, 5) == f"{live}:5"


def test_a_device_string_naming_its_own_index_is_rejected():
    """device names only the type; a caller who wants a card passes gpu_id, not
    'cpu:N'.
    """
    import pytest

    with pytest.raises(ValueError, match="names an index"):
        dev.resolve_device_spec("cpu:1")
    with pytest.raises(ValueError, match="names an index"):
        dev.resolve_device_spec("cpu:0", 5)


def test_a_device_this_host_cannot_serve_is_rejected_not_retargeted():
    """Silently remapping could not tell a legacy 'cuda' default from an operator
    genuinely asking for CUDA, so a mismatch is an error at the boundary now.
    """
    import pytest

    live = current_platform.device_type
    absent = "xpu" if live == "cuda" else "cuda"

    with pytest.raises(ValueError, match="this host resolved to"):
        dev.resolve_device_spec(f"{absent}:1")
    with pytest.raises(ValueError, match="this host resolved to"):
        dev.resolve_device_spec("npu:0")


def test_sglang_caps_xpu_free_memory_against_the_allocator(monkeypatch) -> None:
    """Omni sizes the KV pool from this number. SGLang derives it from the
    allocator, total device memory minus what this process has allocated, not
    from a driver free-memory query that can over-report.
    """
    from types import SimpleNamespace

    import torch
    from sglang.srt.utils.common import get_available_gpu_memory

    total_bytes = 64 << 30
    allocated_bytes = 24 << 30
    fake_xpu = SimpleNamespace(
        device_count=lambda: 1,
        current_device=lambda: 0,
        memory_allocated=lambda gpu_id: allocated_bytes,
        get_device_properties=lambda gpu_id: SimpleNamespace(total_memory=total_bytes),
    )
    monkeypatch.setattr(torch, "xpu", fake_xpu)

    free_gb = get_available_gpu_memory("xpu", 0, distributed=False, empty_cache=False)

    assert free_gb == (total_bytes - allocated_bytes) / (1 << 30)


@pytest.fixture
def fresh_video_decode_setting() -> Iterator[None]:
    """The setting is read once per process; each case reads its own and leaves none behind."""
    xpu_platform.video_decode_setting.cache_clear()
    yield
    xpu_platform.video_decode_setting.cache_clear()


@pytest.mark.parametrize(
    ("setting", "plugin_installed", "expected_device"),
    [
        (None, True, None),
        ("0", True, None),
        ("off", True, None),
        ("ON", True, "xpu"),
        ("XPU:3", True, "xpu:3"),
        ("1", False, None),
    ],
)
def test_xpu_decodes_video_on_the_media_engine_only_when_enabled_with_the_plugin(
    fresh_video_decode_setting: None,
    monkeypatch: pytest.MonkeyPatch,
    setting: str | None,
    plugin_installed: bool,
    expected_device: str | None,
) -> None:
    """Off by default, since decoding on a card the thinker has filled slows serving; a named card is used as given; without torchcodec-xpu, video decodes on the CPU."""
    if setting is None:
        monkeypatch.delenv(xpu_platform.VIDEO_DECODE_ENV, raising=False)
    else:
        monkeypatch.setenv(xpu_platform.VIDEO_DECODE_ENV, setting)
    monkeypatch.setattr(
        xpu_platform, "torchcodec_xpu_installed", lambda: plugin_installed
    )

    assert xpu_platform.XPUOmniPlatform().get_video_decode_device() == expected_device


@pytest.mark.parametrize("setting", ["sometimes", "xpu:", "cuda:0", "xpu:1,2"])
def test_an_unknown_video_decode_setting_is_rejected(
    fresh_video_decode_setting: None, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    monkeypatch.setenv(xpu_platform.VIDEO_DECODE_ENV, setting)
    with pytest.raises(ValueError, match="SGLANG_OMNI_XPU_VIDEO_DECODE"):
        xpu_platform.XPUOmniPlatform().get_video_decode_device()
