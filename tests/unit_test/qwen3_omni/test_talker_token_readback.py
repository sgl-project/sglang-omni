# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner


class _CountingIds:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor
        self.tolist_calls = 0

    def tolist(self):
        self.tolist_calls += 1
        return self._tensor.tolist()

    def __getattr__(self, name):
        return getattr(self._tensor, name)


def _scheduler_output(n: int) -> SimpleNamespace:
    reqs = [SimpleNamespace(request_id=f"r{i}", data=None) for i in range(n)]
    return SimpleNamespace(requests=reqs, batch_data=None)


def _make_processor():
    sglang_backend = pytest.importorskip(
        "sglang_omni.scheduling.sglang_backend"
    )
    return sglang_backend.SGLangOutputProcessor()


def test_process_materializes_once_from_device_tensor():
    proc = _make_processor()
    ref = torch.tensor([11, 22, 33, 44])
    counting = _CountingIds(ref)
    model_output = SimpleNamespace(next_token_ids=counting)

    outputs = proc.process(model_output, _scheduler_output(4))

    assert counting.tolist_calls == 1
    assert [outputs[f"r{i}"].data for i in range(4)] == ref.tolist()


def test_process_prefers_staged_host_copy_and_skips_device_readback():
    proc = _make_processor()
    ref = torch.tensor([1, 2, 3])
    device_ids = _CountingIds(ref)
    host_copy = _CountingIds(ref)
    model_output = SimpleNamespace(
        next_token_ids=device_ids, _host_token_ids=host_copy
    )

    outputs = proc.process(model_output, _scheduler_output(3))

    assert device_ids.tolist_calls == 0
    assert host_copy.tolist_calls == 1
    assert [outputs[f"r{i}"].data for i in range(3)] == ref.tolist()


def _bare_runner() -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner._token_id_host_bufs = None
    runner._token_id_host_slot = 0
    return runner


def test_stage_token_ids_cpu_passthrough_no_event():
    runner = _bare_runner()
    ref = torch.tensor([7, 8, 9])
    result = SimpleNamespace()

    runner._stage_token_ids(result, ref)

    assert result._host_token_ids is ref
    assert result._host_token_ids_event is None
    assert runner._resolve_host_token_ids(result).tolist() == ref.tolist()


def test_resolve_host_token_ids_absent_returns_none():
    runner = _bare_runner()
    result = SimpleNamespace()

    assert runner._resolve_host_token_ids(result) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stage_token_ids_cuda_matches_reference():
    runner = _bare_runner()
    ref = torch.tensor([3, 5, 7, 9], device="cuda")
    result = SimpleNamespace()

    runner._stage_token_ids(result, ref)
    host = runner._resolve_host_token_ids(result)

    assert not host.is_cuda
    assert host.tolist() == ref.cpu().tolist()
