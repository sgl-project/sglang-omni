# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.engines.omni.types import RequestOutput, SchedulerRequest
from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_ar import S2ProStepOutput
from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_sglang_ar import (
    S2ProSGLangIterationController,
    S2ProSGLangModelRunner,
    S2ProSGLangRequestData,
    S2ProSGLangResourceManager,
)


class _FakeReq:
    def __init__(self) -> None:
        self.output_ids: list[int] = []
        self.decode_batch_idx = 0
        self.is_chunked = 0
        self.finished_reason = None
        self.finished_len = None
        self.prefix_indices = []
        self.req_pool_idx = 0
        self.origin_input_ids = [1, 2, 3]
        self.fill_ids = [1, 2, 3]
        self.cache_protected_len = 0

    def finished(self) -> bool:
        return self.finished_reason is not None


class _FakeTreeCache:
    def __init__(self) -> None:
        self.calls: list[_FakeReq] = []

    def cache_unfinished_req(self, req: _FakeReq) -> None:
        self.calls.append(req)


def _make_request_data(req: _FakeReq, *, max_new_tokens: int = 4) -> S2ProSGLangRequestData:
    return S2ProSGLangRequestData(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        req=req,
        max_new_tokens=max_new_tokens,
    )


def _make_step_output(semantic_token: int, codebooks: list[int] | None = None) -> RequestOutput:
    if codebooks is None:
        codebooks = [101, 202]
    codes = torch.tensor(
        [[semantic_token], *[[value] for value in codebooks]],
        dtype=torch.long,
    )
    return RequestOutput(
        request_id="req-1",
        data=S2ProStepOutput(codes=codes),
        finished=False,
    )


def test_iteration_controller_marks_eos_finished_without_unfinished_cache() -> None:
    tree_cache = _FakeTreeCache()
    controller = S2ProSGLangIterationController(tree_cache=tree_cache, im_end_token_id=999)
    req = _FakeReq()
    data = _make_request_data(req, max_new_tokens=8)
    request = SchedulerRequest(request_id="req-1", data=data)
    output = _make_step_output(999)

    controller.update_request(request, output)

    assert req.output_ids == [999]
    assert req.finished() is True
    assert controller.is_finished(request, output) is True
    assert req.finished_len == 1
    assert tree_cache.calls == []


def test_iteration_controller_marks_length_finished_on_limit() -> None:
    tree_cache = _FakeTreeCache()
    controller = S2ProSGLangIterationController(tree_cache=tree_cache, im_end_token_id=999)
    req = _FakeReq()
    data = _make_request_data(req, max_new_tokens=2)
    request = SchedulerRequest(request_id="req-1", data=data)
    first_output = _make_step_output(111)

    controller.update_request(request, first_output)
    assert req.finished() is False
    assert controller.is_finished(request, first_output) is False
    assert len(tree_cache.calls) == 1

    second_output = _make_step_output(222)
    controller.update_request(request, second_output)
    assert req.finished() is True
    assert controller.is_finished(request, second_output) is True
    assert req.finished_len == 2
    assert req.output_ids == [111, 222]
    assert len(tree_cache.calls) == 1


def test_build_outputs_clones_model_buffers() -> None:
    text_model = SimpleNamespace(
        _output_codes=torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.long),
        _output_semantic_ids=torch.tensor([11, 21], dtype=torch.long),
    )
    model_worker = SimpleNamespace(model_runner=SimpleNamespace(model=text_model))
    runner = S2ProSGLangModelRunner(
        model_worker=model_worker,
        batch_planner=SimpleNamespace(),
        semantic_begin_id=0,
        semantic_end_id=1000,
    )

    req = _FakeReq()
    data = _make_request_data(req)
    scheduler_output = SimpleNamespace(
        requests=[SchedulerRequest(request_id="req-1", data=data)]
    )

    outputs = runner._build_outputs(scheduler_output)
    text_model._output_codes[0, 0] = 999

    assert outputs["req-1"].data.codes[0, 0].item() == 11


def test_resource_manager_free_clears_request_state(monkeypatch) -> None:
    released: list[_FakeReq] = []

    def _fake_release(req: _FakeReq, tree_cache: object) -> None:
        released.append(req)

    monkeypatch.setattr(
        "sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_sglang_ar.release_kv_cache",
        _fake_release,
    )

    req = _FakeReq()
    data = _make_request_data(req)
    data._previous_semantic_tokens = [1, 2]
    data._last_codebook_values = torch.tensor([7, 8], dtype=torch.long)
    request = SchedulerRequest(request_id="req-1", data=data)

    mgr = S2ProSGLangResourceManager(
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
        tree_cache=object(),
    )
    mgr.free(request)

    assert released == [req]
    assert data._previous_semantic_tokens == []
    assert data._last_codebook_values is None
