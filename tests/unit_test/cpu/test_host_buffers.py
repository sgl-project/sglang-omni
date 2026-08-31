# SPDX-License-Identifier: Apache-2.0
"""Host staging buffers must not ask for pinned memory on a platform without it.

``pin_memory=True`` raises on a CPU-only torch build rather than degrading, and
the failure surfaces late — deep in a model step, after a full prefill. These
drive the shared helper directly so the contract is checked without standing a
model up.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni import platforms
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform


def test_only_cpu_refuses_pinned_host_memory():
    """Default stays permissive: an accelerator platform that forgot to override
    this would otherwise lose pinned staging and quietly slow every copy.
    """
    assert OmniPlatform().is_pin_memory_available() is True
    assert CPUOmniPlatform().is_pin_memory_available() is False


def _pingpong(owner, shape=(4,), dtype=torch.int32):
    return ModelRunner._pinned_pingpong(
        owner, "_bufs", "_slot", shape, dtype, realloc_on_grow=True
    )


def test_buffers_are_unpinned_when_the_platform_has_no_pinned_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        platforms.current_platform, "is_pin_memory_available", lambda: False
    )
    owner = SimpleNamespace(_bufs=[], _slot=0)

    buf = _pingpong(owner)

    assert buf.is_pinned() is False
    assert buf.device.type == "cpu"
    assert buf.shape == (4,)


def test_the_two_buffers_still_alternate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping the pin must not disturb the ping-pong itself: a step's host read
    would otherwise race the next step's copy into the same buffer.
    """
    monkeypatch.setattr(
        platforms.current_platform, "is_pin_memory_available", lambda: False
    )
    owner = SimpleNamespace(_bufs=[], _slot=0)

    first = _pingpong(owner)
    second = _pingpong(owner)
    third = _pingpong(owner)

    assert first is not second
    assert third is first


def test_a_pinning_platform_still_gets_pinned_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU branch must not become the path everyone takes. Allocation itself
    needs a real accelerator, so assert the request rather than the result.
    """
    monkeypatch.setattr(
        platforms.current_platform, "is_pin_memory_available", lambda: True
    )
    seen: dict[str, object] = {}
    real_empty = torch.empty

    def spy_empty(*args, **kwargs):
        seen["pin_memory"] = kwargs.get("pin_memory")
        kwargs["pin_memory"] = False  # this host may have no pinned allocator
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", spy_empty)
    owner = SimpleNamespace(_bufs=[], _slot=0)

    _pingpong(owner)

    assert seen["pin_memory"] is True
