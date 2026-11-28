# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.vendor.sglang import ascend_kda


class Recorder:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(self, gate: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.inputs.append(gate)
        return gate


def test_gate_is_made_contiguous() -> None:
    recorder = Recorder()
    module = SimpleNamespace(fused_kda_gate_npu=recorder)

    assert ascend_kda.patch_gate(module, "fused_kda_gate_npu") is True
    sliced = torch.zeros(2, 8)[:, :4]
    assert not sliced.is_contiguous()
    module.fused_kda_gate_npu(sliced)

    assert recorder.inputs[0].is_contiguous()
    assert torch.equal(recorder.inputs[0], sliced)


def test_absent_name_is_skipped() -> None:
    module = SimpleNamespace(fused_kda_gate=Recorder())

    assert ascend_kda.patch_gate(module, "fused_kda_gate_npu") is False
    assert ascend_kda.patch_gate(module, "fused_kda_gate") is True


def test_patch_is_idempotent() -> None:
    module = SimpleNamespace(fused_kda_gate_npu=Recorder())

    assert ascend_kda.patch_gate(module, "fused_kda_gate_npu") is True
    assert ascend_kda.patch_gate(module, "fused_kda_gate_npu") is False
