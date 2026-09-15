# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.mps.numerics import validate_sm_partition


@pytest.fixture
def cpu_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=56),
    )
    original_to = torch.Tensor.to

    def to_cpu(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_cpu)


def test_numerics_accepts_correct_bf16_without_changing_rng(cpu_cuda):
    state = torch.random.get_rng_state().clone()
    validate_sm_partition(56)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("error", [0.92, float("nan"), float("inf")])
def test_numerics_rejects_corruption_and_reports_actual_sm(
    cpu_cuda, monkeypatch, error
):
    mm = torch.mm
    monkeypatch.setattr(torch, "mm", lambda a, b: mm(a, b) * (1 + error))
    with pytest.raises(RuntimeError, match="56 SM.*unsafe.*different sm_cap"):
        validate_sm_partition(56)


def test_numerics_checks_reported_sm_before_gemm(cpu_cuda):
    with pytest.raises(RuntimeError, match="requested 48 SM.*reported 56 SM"):
        validate_sm_partition(48)
