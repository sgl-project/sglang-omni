# SPDX-License-Identifier: Apache-2.0
"""How long a DLPack descriptor over raw XPU memory stays registered."""
from __future__ import annotations

import gc
from unittest.mock import Mock

import pytest
import torch

import sglang_omni.platforms.xpu_dlpack as xpu_dlpack
from sglang_omni.platforms.level_zero import (
    LevelZeroContext,
    is_level_zero_ipc_available,
)

POOL_BYTES = 1 << 20


def test_a_failed_dlpack_wrap_keeps_no_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        xpu_dlpack.torch,
        "from_dlpack",
        Mock(side_effect=RuntimeError("DLPack import failed")),
    )
    kept = len(xpu_dlpack.LIVE_DLPACK_DESCRIPTORS)

    with pytest.raises(RuntimeError, match="DLPack import failed"):
        xpu_dlpack.wrap_device_bytes(4096, 64, 0)
    assert len(xpu_dlpack.LIVE_DLPACK_DESCRIPTORS) == kept


@pytest.mark.accelerator
def test_a_descriptor_lives_exactly_as_long_as_its_storage() -> None:
    """A view keeps the storage alive after the wrapped tensor is gone, and the
    descriptor has to survive with it; once the last view goes, so does the entry."""
    if not is_level_zero_ipc_available():
        pytest.skip("requires an Intel GPU with Level-Zero IPC")
    else:
        pass
    context = LevelZeroContext()
    pointer = context.allocate_exportable(0, POOL_BYTES)
    try:
        registered = len(xpu_dlpack.LIVE_DLPACK_DESCRIPTORS)
        pool = xpu_dlpack.wrap_device_bytes(pointer, POOL_BYTES, 0)
        view = pool[4096:8192]
        view.fill_(7)
        torch.xpu.synchronize(0)
        assert len(xpu_dlpack.LIVE_DLPACK_DESCRIPTORS) == registered + 1

        del pool
        gc.collect()
        assert len(xpu_dlpack.LIVE_DLPACK_DESCRIPTORS) == registered + 1
        assert int(view[0]) == 7

        del view
        gc.collect()
        assert len(xpu_dlpack.LIVE_DLPACK_DESCRIPTORS) == registered
    finally:
        context.free(pointer)
        context.destroy()
