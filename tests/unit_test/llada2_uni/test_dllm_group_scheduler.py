# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import sys
import threading
from array import array
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import torch


class _AddReqResult(Enum):
    CONTINUE = 1
    NO_TOKEN = 2


class _Req:
    def __init__(
        self,
        rid,
        origin_input_text,
        origin_input_ids,
        sampling_params,
        **kwargs,
    ) -> None:
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids = array("q", origin_input_ids)
        self.sampling_params = sampling_params
        self.vocab_size = kwargs.get("vocab_size")
        self.eos_token_ids = kwargs.get("eos_token_ids")
        self.dllm_config = kwargs.get("dllm_config")
        self.output_ids = array("q")
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.last_node = object()
        self.req_pool_idx = None
        self._finished = False
        self.init_count = 0

    def init_next_round_input(self, *args) -> None:
        self.init_count += 1

    def finished(self) -> bool:
        return self._finished


module_names = (
    "sglang",
    "sglang.srt",
    "sglang.srt.managers",
    "sglang.srt.managers.schedule_batch",
    "sglang.srt.managers.schedule_policy",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.common",
    "sglang.srt.model_executor",
    "sglang.srt.model_executor.forward_batch_info",
    "sglang.srt.speculative",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.sampling",
    "sglang.srt.sampling.sampling_params",
)
for module_name in module_names:
    sys.modules.setdefault(module_name, ModuleType(module_name))

sys.modules["sglang.srt.managers.schedule_batch"].Req = _Req
sys.modules["sglang.srt.managers.schedule_batch"].ScheduleBatch = object
sys.modules["sglang.srt.managers.schedule_policy"].AddReqResult = _AddReqResult
sys.modules["sglang.srt.managers.schedule_policy"].PrefillAdder = object
sys.modules["sglang.srt.mem_cache.common"].release_kv_cache = lambda *args: None
sys.modules["sglang.srt.model_executor.forward_batch_info"].ForwardBatch = object
sys.modules["sglang.srt.speculative.spec_info"].SpeculativeAlgorithm = SimpleNamespace(
    NONE=0
)
sys.modules["sglang.srt.sampling.sampling_params"].SamplingParams = SimpleNamespace

model_runner_base = ModuleType("sglang_omni.model_runner.base")
model_runner_base.resolve_deferred_prefill_inputs = lambda *args: None
sys.modules.setdefault("sglang_omni.model_runner.base", model_runner_base)

from sglang_omni.scheduling import dllm_scheduler as scheduler_module  # noqa: E402
from sglang_omni.scheduling.dllm_group import (  # noqa: E402
    DllmCompanionSpec,
    DllmRequestGroupSpec,
)
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler  # noqa: E402
from sglang_omni.scheduling.messages import IncomingMessage  # noqa: E402


def _scheduler() -> DllmScheduler:
    scheduler = object.__new__(DllmScheduler)
    scheduler._dllm_group_members = {}
    scheduler._dllm_rid_to_group = {}
    scheduler._dllm_hidden_rids = set()
    scheduler._dllm_orphaned_rids = set()
    return scheduler


def _primary() -> _Req:
    sampling_params = SimpleNamespace(
        max_new_tokens=4,
        custom_params={"keep": 1},
    )
    request = _Req(
        "request-1",
        "",
        [99, 10, 11],
        sampling_params,
        vocab_size=1000,
        eos_token_ids={2},
        dllm_config=object(),
    )
    request.tokenizer = object()
    request.omni_model_inputs = None
    request._omni_consumed = None
    return request


def _spec() -> DllmRequestGroupSpec:
    return DllmRequestGroupSpec(
        companions=(
            DllmCompanionSpec(
                role="unconditional",
                input_ids=(99, 99, 7),
                left_pad_length=2,
            ),
            DllmCompanionSpec(
                role="no_image",
                input_ids=(6, 5, 4),
                left_pad_length=0,
            ),
        ),
        primary_left_pad_length=1,
        algorithm_args={"cfg_scale": 4.0},
    )


def test_scheduler_materializes_one_typed_atomic_group() -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())

    assert [request.origin_input_ids.tolist() for request in requests] == [
        [99, 10, 11],
        [99, 99, 7],
        [6, 5, 4],
    ]
    assert [request.omni_dllm_group_member.role for request in requests] == [
        "conditional",
        "unconditional",
        "no_image",
    ]
    assert scheduler._dllm_group_members["request-1"] == tuple(
        request.rid for request in requests
    )
    assert scheduler._dllm_hidden_rids == {
        requests[1].rid,
        requests[2].rid,
    }
    assert requests[1].sampling_params is not requests[0].sampling_params


def test_scheduler_attaches_ordered_cpu_and_device_group_metadata() -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    for request in requests:
        request.is_dllm_prefill = lambda: True
    batch = SimpleNamespace(reqs=requests)
    forward_batch = SimpleNamespace(
        batch_size=3,
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens_cpu=[3, 3, 3],
        positions=torch.arange(3, dtype=torch.int32).repeat(3),
        seq_lens=torch.tensor([3, 3, 3], dtype=torch.int32),
    )

    scheduler._attach_forward_group(forward_batch, batch)

    assert forward_batch.omni_dllm_group.roles == (
        "conditional",
        "unconditional",
        "no_image",
    )
    assert forward_batch.dllm_left_pad_lens_cpu == (1, 2, 0)
    assert forward_batch.omni_dllm_group_is_prefill is True


def test_scheduler_keeps_ordinary_text_batch_free_of_image_vocab_metadata() -> None:
    scheduler = _scheduler()
    request = _primary()
    batch = SimpleNamespace(reqs=[request])
    forward_batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([99, 10, 11]),
    )

    scheduler._attach_forward_group(forward_batch, batch)

    assert not hasattr(forward_batch, "omni_dllm_image_token_offsets")


def test_scheduler_attaches_native_image_vocab_boundary_without_cfg() -> None:
    scheduler = _scheduler()
    request = _primary()
    request.omni_dllm_image_token_offset = 321000
    batch = SimpleNamespace(reqs=[request])
    forward_batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([99, 10, 11]),
    )

    scheduler._attach_forward_group(forward_batch, batch)

    assert forward_batch.omni_dllm_image_token_offsets.tolist() == [321000]


def test_partial_group_admission_rolls_back_all_request_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    scheduler._waiting_queue = list(requests)
    scheduler._staging_queue = []
    scheduler.server_args = SimpleNamespace(page_size=1, max_prefill_tokens=128)
    scheduler.tree_cache = SimpleNamespace(dec_lock_ref=lambda node: None)
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.req_to_token_pool = object()
    scheduler.model_config = object()
    scheduler.dllm_config = SimpleNamespace(block_size=4)
    scheduler._chunked_prefill_size = 4

    class _PartialAdder:
        def __init__(self, *args, **kwargs) -> None:
            self.can_run_list = []

        def add_one_req(self, request, **kwargs):
            if not self.can_run_list:
                self.can_run_list.append(request)
                return _AddReqResult.CONTINUE
            return _AddReqResult.NO_TOKEN

    monkeypatch.setattr(scheduler_module, "PrefillAdder", _PartialAdder)

    assert scheduler._schedule_next_batch() is None
    assert [request.init_count for request in requests] == [0, 0, 0]
    assert scheduler._waiting_queue == requests
    assert scheduler._staging_queue == []


def test_impossible_group_capacity_is_rejected_before_admission() -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    scheduler.server_args = SimpleNamespace(
        page_size=1,
        max_prefill_tokens=128,
        max_running_requests=2,
    )
    scheduler.dllm_config = SimpleNamespace(block_size=4)

    with pytest.raises(RuntimeError, match="requires 3 running requests"):
        scheduler._validate_request_group_capacity(requests)


def test_group_prefill_budget_includes_every_physical_row() -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    scheduler.server_args = SimpleNamespace(
        page_size=1,
        max_prefill_tokens=15,
        max_running_requests=3,
    )
    scheduler.dllm_config = SimpleNamespace(block_size=4)

    with pytest.raises(RuntimeError, match="max_prefill_tokens"):
        scheduler._validate_request_group_capacity(requests)


def test_impossible_new_group_returns_error_without_entering_scheduler() -> None:
    scheduler = _scheduler()
    primary = _primary()
    primary.omni_dllm_group_spec = _spec()
    scheduler.server_args = SimpleNamespace(
        page_size=1,
        max_prefill_tokens=128,
        max_running_requests=2,
    )
    scheduler.dllm_config = SimpleNamespace(block_size=4)
    scheduler._waiting_queue = []
    scheduler._staging_queue = []
    scheduler._rid_to_req_data = {}
    scheduler._request_builder = lambda value: SimpleNamespace(req=value)
    scheduler._abort_lock = threading.Lock()
    scheduler._aborted_request_ids = set()
    scheduler.inbox = queue.Queue()
    scheduler.outbox = queue.Queue()
    scheduler.tree_cache = object()
    scheduler.inbox.put(IncomingMessage(primary.rid, "new_request", primary))

    scheduler._drain_and_purge()

    error = scheduler.outbox.get_nowait()
    assert error.request_id == primary.rid
    assert error.type == "error"
    assert isinstance(error.data, RuntimeError)
    assert "requires 3 running requests" in str(error.data)
    assert scheduler._waiting_queue == []
    assert scheduler._rid_to_req_data == {}
    assert scheduler._dllm_group_members == {}


def test_request_builder_failure_is_isolated_from_later_request() -> None:
    scheduler = _scheduler()
    failure = ValueError("invalid request")
    valid = _primary()
    scheduler.server_args = SimpleNamespace(
        page_size=1,
        max_prefill_tokens=128,
        max_running_requests=3,
    )
    scheduler.dllm_config = SimpleNamespace(block_size=4)
    scheduler._waiting_queue = []
    scheduler._staging_queue = []
    scheduler._rid_to_req_data = {}

    def _build(value):
        if value == "invalid":
            raise failure
        return SimpleNamespace(req=value)

    scheduler._request_builder = _build
    scheduler._abort_lock = threading.Lock()
    scheduler._aborted_request_ids = set()
    scheduler.inbox = queue.Queue()
    scheduler.outbox = queue.Queue()
    scheduler.tree_cache = object()
    scheduler.inbox.put(IncomingMessage("request-invalid", "new_request", "invalid"))
    scheduler.inbox.put(IncomingMessage(valid.rid, "new_request", valid))

    scheduler._drain_and_purge()

    error = scheduler.outbox.get_nowait()
    assert error.request_id == "request-invalid"
    assert error.type == "error"
    assert error.data is failure
    with pytest.raises(queue.Empty):
        scheduler.outbox.get_nowait()
    assert scheduler._waiting_queue == [valid]
    assert scheduler._staging_queue == []
    assert scheduler._rid_to_req_data == {valid.rid: SimpleNamespace(req=valid)}
    assert scheduler._dllm_group_members == {}
    assert scheduler._dllm_rid_to_group == {}
    assert scheduler._dllm_hidden_rids == set()
    assert scheduler._dllm_orphaned_rids == set()


def test_group_materialization_failure_clears_primary_request_state() -> None:
    scheduler = _scheduler()
    invalid = _primary()
    invalid.rid = "request-invalid-group"
    invalid.omni_dllm_group_spec = DllmRequestGroupSpec(
        companions=(
            DllmCompanionSpec(
                role="unconditional",
                input_ids=(99, 7),
                left_pad_length=1,
            ),
        )
    )
    valid = _primary()
    valid.rid = "request-valid"
    scheduler.server_args = SimpleNamespace(
        page_size=1,
        max_prefill_tokens=128,
        max_running_requests=3,
    )
    scheduler.dllm_config = SimpleNamespace(block_size=4)
    scheduler._waiting_queue = []
    scheduler._staging_queue = []
    scheduler._rid_to_req_data = {}
    scheduler._request_builder = lambda value: SimpleNamespace(req=value)
    scheduler._abort_lock = threading.Lock()
    scheduler._aborted_request_ids = set()
    scheduler.inbox = queue.Queue()
    scheduler.outbox = queue.Queue()
    scheduler.tree_cache = object()
    scheduler.inbox.put(IncomingMessage(invalid.rid, "new_request", invalid))
    scheduler.inbox.put(IncomingMessage(valid.rid, "new_request", valid))

    scheduler._drain_and_purge()

    error = scheduler.outbox.get_nowait()
    assert error.request_id == invalid.rid
    assert error.type == "error"
    assert isinstance(error.data, ValueError)
    assert scheduler._waiting_queue == [valid]
    assert scheduler._staging_queue == []
    assert set(scheduler._rid_to_req_data) == {valid.rid}
    assert scheduler._dllm_group_members == {}
    assert scheduler._dllm_rid_to_group == {}
    assert scheduler._dllm_hidden_rids == set()
    assert scheduler._dllm_orphaned_rids == set()


def test_group_phases_follow_conditional_row() -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    requests[0].dllm_phase = "decode"
    requests[1].dllm_phase = "prefill"
    requests[2].dllm_phase = "prefill"

    scheduler._synchronize_request_group_phases(requests)

    assert [request.dllm_phase for request in requests] == [
        "decode",
        "decode",
        "decode",
    ]


def test_grouped_fdfo_prefill_result_is_not_treated_as_generated_tokens() -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    for request in requests:
        request.is_dllm_prefill = lambda: True
    scheduler.dllm_config = SimpleNamespace(
        first_done_first_out_mode=True,
        block_size=4,
    )
    batch = SimpleNamespace(reqs=requests)
    result = SimpleNamespace(
        next_token_ids=[[], [], []],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    scheduler._apply_results(batch, result)

    assert all(request.output_ids == array("q") for request in requests)


def test_abort_of_primary_releases_every_physical_row_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    scheduler._waiting_queue = []
    scheduler._staging_queue = list(requests)
    scheduler._rid_to_req_data = {requests[0].rid: object()}
    scheduler._request_builder = lambda value: value
    scheduler._abort_lock = threading.Lock()
    scheduler._aborted_request_ids = {requests[0].rid}
    scheduler.inbox = queue.Queue()
    scheduler.tree_cache = object()
    released: list[str] = []
    monkeypatch.setattr(
        scheduler_module,
        "release_kv_cache",
        lambda request, tree_cache: released.append(request.rid),
    )

    scheduler._drain_and_purge()

    assert released == [request.rid for request in requests]
    assert scheduler._staging_queue == []
    assert scheduler._dllm_group_members == {}
    assert scheduler._dllm_rid_to_group == {}


def test_group_finish_retires_and_releases_every_row_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    requests[0]._finished = True
    scheduler._waiting_queue = []
    scheduler._staging_queue = list(requests)
    scheduler.tree_cache = object()
    scheduler.req_to_token_pool = SimpleNamespace(free=lambda request: None)
    released: list[str] = []
    excluded: list[object] = []
    monkeypatch.setattr(
        scheduler_module,
        "release_kv_cache",
        lambda request, tree_cache: released.append(request.rid),
    )
    batch = SimpleNamespace(
        reqs=requests,
        filter_batch=lambda *, chunked_req_to_exclude: excluded.extend(
            chunked_req_to_exclude
        ),
    )

    scheduler._post_step(batch)

    assert released == [request.rid for request in requests]
    assert set(excluded) == set(requests)
    assert scheduler._staging_queue == []
    assert scheduler._dllm_group_members == {}


def test_group_result_adapter_failure_retires_group_and_allows_later_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    requests = scheduler._materialize_request_group(_primary(), _spec())
    for request in requests:
        request.output_ids = array("q")
        request.output_ids_through_stop = request.output_ids
        request.full_untruncated_fill_ids = array("q", [99, 10, 11, -1])
        request.extend_range = SimpleNamespace(end=4)
        request.finished_reason = None
        request.update_finish_state = lambda *, new_accepted_len=1: None
        request.is_dllm_prefill = lambda: False
        request._finished = True

    scheduler.dllm_config = SimpleNamespace(
        first_done_first_out_mode=False,
        block_size=1,
    )
    scheduler._waiting_queue = []
    scheduler._staging_queue = list(requests)
    primary_data = SimpleNamespace(req=requests[0])
    scheduler._rid_to_req_data = {requests[0].rid: primary_data}
    scheduler.outbox = queue.Queue()
    adapter_failure = RuntimeError("adapter failed")
    adapted: list[str] = []

    def _adapt(req_data):
        if req_data is primary_data:
            raise adapter_failure
        adapted.append(req_data.req.rid)
        return req_data.req.rid

    scheduler._result_adapter = _adapt
    scheduler.tree_cache = object()
    scheduler.req_to_token_pool = SimpleNamespace(free=lambda request: None)
    released: list[str] = []
    excluded: list[object] = []
    monkeypatch.setattr(
        scheduler_module,
        "release_kv_cache",
        lambda request, tree_cache: released.append(request.rid),
    )
    batch = SimpleNamespace(
        reqs=requests,
        filter_batch=lambda *, chunked_req_to_exclude: excluded.extend(
            chunked_req_to_exclude
        ),
    )
    result = SimpleNamespace(
        next_token_ids=[[7], [7], [7]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    scheduler._apply_results(batch, result)
    scheduler._post_step(batch)

    error = scheduler.outbox.get_nowait()
    assert error.request_id == requests[0].rid
    assert error.type == "error"
    assert error.data is adapter_failure
    assert released == [request.rid for request in requests]
    assert set(excluded) == set(requests)
    assert scheduler._waiting_queue == []
    assert scheduler._staging_queue == []
    assert scheduler._rid_to_req_data == {}
    assert scheduler._dllm_group_members == {}
    assert scheduler._dllm_rid_to_group == {}
    assert scheduler._dllm_hidden_rids == set()
    assert scheduler._dllm_orphaned_rids == set()

    later = _primary()
    later.rid = "request-later"
    later.output_ids = array("q")
    later.output_ids_through_stop = later.output_ids
    later.full_untruncated_fill_ids = array("q", [99, 10, 11, -1])
    later.extend_range = SimpleNamespace(end=4)
    later.finished_reason = None
    later.update_finish_state = lambda *, new_accepted_len=1: None
    later.is_dllm_prefill = lambda: False
    later._finished = True
    scheduler._rid_to_req_data[later.rid] = SimpleNamespace(req=later)
    later_batch = SimpleNamespace(reqs=[later])
    later_result = SimpleNamespace(
        next_token_ids=[[8]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    scheduler._apply_results(later_batch, later_result)

    success = scheduler.outbox.get_nowait()
    assert success.request_id == later.rid
    assert success.type == "result"
    assert success.data == later.rid
    assert adapted == [later.rid]
