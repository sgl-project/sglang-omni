# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.scheduling.stage_residency import StageResidency


class _RecordingRing:
    """Staging-ring stand-in that records the restore it is handed."""

    def __init__(self) -> None:
        self.restored: tuple[list[torch.Tensor], list[object]] | None = None

    def restore(self, storages: list[torch.Tensor], chunks: list[object]) -> None:
        self.restored = (storages, chunks)


def test_wake_lands_weights_without_a_device_wide_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wake's completion guarantee comes from the restore itself.

    ``StagingRing.restore`` ends with its own ``stream.synchronize()``, and
    ``torch_memory_saver.resume`` only remaps virtual storage, so ``wake``
    must not drain the whole device on its critical path.
    """
    residency = StageResidency(
        {"module": torch.nn.Linear(2, 2)}, torch.device("cpu"), source="ram"
    )
    residency.sleep()
    assert not residency.resident

    ring = _RecordingRing()
    residency.staging_ring = ring
    residency.device = torch.device("cuda")

    def forbid_device_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("wake() must not call torch.cuda.synchronize")

    monkeypatch.setattr(
        "sglang_omni.scheduling.stage_residency.gpu_memory_note",
        lambda device: "note",
    )
    monkeypatch.setattr(torch.cuda, "synchronize", forbid_device_sync)
    residency.wake()

    assert ring.restored is not None
    assert residency.resident
