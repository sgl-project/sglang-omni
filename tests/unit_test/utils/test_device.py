# SPDX-License-Identifier: Apache-2.0
"""Host-independent cases of the stage device resolver."""

from types import SimpleNamespace

import pytest

from sglang_omni.platforms import current_platform
from sglang_omni.utils import device as dev


def test_a_placed_index_needs_no_accelerator_query(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cuda", raising=False)
    assert str(dev.resolve_concrete_device(None, 1)) == "cuda:1"
    assert str(dev.resolve_concrete_device("cuda", 3)) == "cuda:3"


def test_mps_falls_back_to_its_single_metal_device(monkeypatch):
    """torch.mps has no current_device(); Apple hosts expose exactly one."""
    monkeypatch.setattr(current_platform, "device_type", "mps", raising=False)
    assert str(dev.resolve_concrete_device(None, None)) == "mps:0"
    assert str(dev.resolve_concrete_device("mps", None)) == "mps:0"


def test_an_indexed_device_string_is_refused(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cuda", raising=False)
    with pytest.raises(ValueError, match="names an index"):
        dev.resolve_concrete_device("cuda:7", 2)


def test_a_foreign_device_type_is_refused(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cuda", raising=False)
    with pytest.raises(ValueError, match="this host resolved to"):
        dev.resolve_concrete_device("xpu", 0)


def test_the_audio_paths_stage_through_cuda_musa_and_xpu() -> None:
    """supports_device_streams names the device types whose torch module answers
    Stream, Event and graph_pool_handle the way torch.cuda does. NPU is absent
    until someone measures those paths on it."""
    for device_type in ("cuda", "musa", "xpu"):
        assert dev.supports_device_streams(SimpleNamespace(type=device_type))
    for device_type in ("cpu", "mps", "npu"):
        assert not dev.supports_device_streams(SimpleNamespace(type=device_type))
