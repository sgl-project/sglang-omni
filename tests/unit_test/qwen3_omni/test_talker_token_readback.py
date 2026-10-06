# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner


class CountingIds:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.wrapped_tensor = tensor
        self.tolist_calls = 0

    def tolist(self):
        self.tolist_calls += 1
        return self.wrapped_tensor.tolist()

    def __getattr__(self, name):
        return getattr(self.wrapped_tensor, name)


def scheduler_output(n: int) -> SimpleNamespace:
    reqs = [SimpleNamespace(request_id=f"r{i}", data=None) for i in range(n)]
    return SimpleNamespace(requests=reqs, batch_data=None)


def make_processor():
    sglang_backend = pytest.importorskip("sglang_omni.scheduling.sglang_backend")
    return sglang_backend.SGLangOutputProcessor()


def test_process_materializes_once_from_device_tensor():
    proc = make_processor()
    ref = torch.tensor([11, 22, 33, 44])
    counting = CountingIds(ref)
    model_output = SimpleNamespace(next_token_ids=counting)

    outputs = proc.process(model_output, scheduler_output(4))

    assert counting.tolist_calls == 1
    assert [outputs[f"r{i}"].data for i in range(4)] == ref.tolist()


def test_process_prefers_staged_host_copy_and_skips_device_readback():
    proc = make_processor()
    device_ids = CountingIds(torch.tensor([1, 2, 3]))
    host_ref = torch.tensor([91, 92, 93])
    host_copy = CountingIds(host_ref)
    model_output = SimpleNamespace(next_token_ids=device_ids)

    outputs = proc.process(model_output, scheduler_output(3), host_token_ids=host_copy)

    assert device_ids.tolist_calls == 0
    assert host_copy.tolist_calls == 1
    assert [outputs[f"r{i}"].data for i in range(3)] == host_ref.tolist()


def bare_runner() -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    return runner


def test_stage_token_ids_cpu_passthrough_no_event():
    runner = bare_runner()
    ref = torch.tensor([7, 8, 9])
    result = SimpleNamespace()

    runner.stage_token_ids(result, ref)

    assert result._host_token_ids is ref  # noqa: leading-underscore  # production name
    assert (
        result._host_token_ids_event is None
    )  # noqa: leading-underscore  # production name
    assert runner.resolve_host_token_ids(result).tolist() == ref.tolist()


def test_resolve_host_token_ids_absent_returns_none():
    runner = bare_runner()
    result = SimpleNamespace()

    assert runner.resolve_host_token_ids(result) is None


def finalize_scheduler_output(n: int) -> SimpleNamespace:
    reqs = [
        SimpleNamespace(
            request_id=f"r{i}",
            data=SimpleNamespace(generation_steps=0, extra_model_outputs={}),
        )
        for i in range(n)
    ]
    return SimpleNamespace(requests=reqs, batch_data=None)


def finalize_runner() -> ModelRunner:
    runner = bare_runner()
    runner.output_processor = make_processor()
    return runner


def test_finalize_populates_host_token_ids_when_staged():
    runner = finalize_runner()
    host = torch.tensor([41, 42, 43])
    batch_result = SimpleNamespace(
        next_token_ids=torch.tensor([1, 2, 3]),
        can_run_cuda_graph=False,
        _host_token_ids=host,
        _host_token_ids_event=None,
    )
    schedule_batch = SimpleNamespace(is_prefill_only=False)

    out = runner.finalize(
        batch_result,
        forward_batch=None,
        schedule_batch=schedule_batch,
        scheduler_output=finalize_scheduler_output(3),
    )

    assert out.host_token_ids is host
    assert [out.outputs[f"r{i}"].data for i in range(3)] == host.tolist()


def test_finalize_host_token_ids_none_without_stage():
    runner = finalize_runner()
    batch_result = SimpleNamespace(
        next_token_ids=torch.tensor([1, 2, 3]),
        can_run_cuda_graph=False,
    )
    schedule_batch = SimpleNamespace(is_prefill_only=False)

    out = runner.finalize(
        batch_result,
        forward_batch=None,
        schedule_batch=schedule_batch,
        scheduler_output=finalize_scheduler_output(3),
    )

    assert out.host_token_ids is None
    assert [out.outputs[f"r{i}"].data for i in range(3)] == [1, 2, 3]


def test_pingpong_alternates_slots_and_reuses_backing_buffer(monkeypatch):
    real_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *a, **k: real_empty(*a, **{**k, "pin_memory": False}),
    )
    runner = bare_runner()
    like = torch.zeros(3, dtype=torch.long)

    first = runner.next_token_id_host_buf(like, 3)
    second = runner.next_token_id_host_buf(like, 3)
    third = runner.next_token_id_host_buf(like, 3)

    assert first is not second
    assert third is first


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stage_token_ids_cuda_matches_reference():
    runner = bare_runner()
    ref = torch.tensor([3, 5, 7, 9], device="cuda")
    result = SimpleNamespace()

    runner.stage_token_ids(result, ref)
    host = runner.resolve_host_token_ids(result)

    assert not host.is_cuda
    assert host.tolist() == ref.cpu().tolist()
