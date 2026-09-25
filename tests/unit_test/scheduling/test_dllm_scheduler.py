# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from array import array
from types import SimpleNamespace

import pytest
from sglang.srt.managers.schedule_batch import ReqKvInfo

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.scheduling import dllm_scheduler as dllm_scheduler_module
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


class ReqDouble:
    def __init__(self, *, rid: str = "req", block_size: int = 4) -> None:
        self.rid = rid
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.origin_input_ids = array("q", [1, 2])
        self.full_untruncated_fill_ids = self.origin_input_ids + array(
            "q", [-1] * block_size
        )
        self.extend_range = SimpleNamespace(end=len(self.full_untruncated_fill_ids))
        self.output_ids: list[int] = []
        self.output_ids_through_stop: list[int] = self.output_ids
        self.finished_reason = None
        self.kv = ReqKvInfo(req_pool_idx=3)
        self.accepted_lengths: list[int] = []
        self.is_finished = False

    def update_finish_state(self, *, new_accepted_len: int = 1) -> None:
        self.accepted_lengths.append(new_accepted_len)

    def finished(self) -> bool:
        return self.is_finished


def make_scheduler(*, fdfo: bool, block_size: int = 4) -> DllmScheduler:
    scheduler = object.__new__(DllmScheduler)
    scheduler.dllm_config = SimpleNamespace(
        first_done_first_out_mode=fdfo,
        block_size=block_size,
    )
    scheduler.rid_to_req_data = {}
    scheduler.result_adapter = lambda value: value
    scheduler.outbox = SimpleNamespace(put=lambda value: None)
    scheduler.max_requests_per_round = None
    return scheduler


def _admission_req(rid: str) -> SimpleNamespace:
    return SimpleNamespace(
        rid=rid,
        inflight_middle_chunks=0,
        init_next_round_input=lambda *args: None,
    )


def _capped_round(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cap: int | None,
    staging: list,
    waiting: list,
) -> tuple[DllmScheduler, list]:
    from sglang.srt.runtime_context import get_context

    scheduler = _scheduler(fdfo=True)
    scheduler.tree_cache = object()
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.req_to_token_pool = object()
    scheduler.model_config = object()
    scheduler.chunked_prefill_size = 16
    scheduler.max_requests_per_round = cap
    scheduler.staging_queue = list(staging)
    scheduler.waiting_queue = list(waiting)
    admitted: list = []

    class _Adder:
        def __init__(self, *args, **kwargs) -> None:
            self.can_run_list = admitted

        def add_dllm_staging_req(self, req):
            self.can_run_list.append(req)
            return dllm_scheduler_module.AddReqResult.CONTINUE

        def add_one_req(self, req, **kwargs):
            self.can_run_list.append(req)
            return dllm_scheduler_module.AddReqResult.CONTINUE

    monkeypatch.setattr(dllm_scheduler_module, "PrefillAdder", _Adder)
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ScheduleBatch",
        SimpleNamespace(
            init_new=lambda **kwargs: SimpleNamespace(
                prepare_for_extend=lambda: None, reqs=kwargs["reqs"]
            )
        ),
    )

    with get_context().override_server_args(page_size=1, max_prefill_tokens=16):
        scheduler.schedule_next_batch()

    return scheduler, admitted


def test_a_capped_round_admits_nothing_beside_the_request_already_denoising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _admission_req("staged")
    waiting = _admission_req("waiting")

    scheduler, admitted = _capped_round(
        monkeypatch, cap=1, staging=[staged], waiting=[waiting]
    )

    assert admitted == [staged]
    assert scheduler.waiting_queue == [waiting]
    assert scheduler.staging_queue == [staged]


def test_an_uncapped_round_still_batches_every_request_that_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _admission_req("staged")
    first = _admission_req("first")
    second = _admission_req("second")

    scheduler, admitted = _capped_round(
        monkeypatch, cap=None, staging=[staged], waiting=[first, second]
    )

    assert admitted == [staged, first, second]
    assert scheduler.waiting_queue == []


def test_a_cap_counts_the_whole_round_not_the_new_arrivals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _admission_req("staged")
    first = _admission_req("first")
    second = _admission_req("second")

    scheduler, admitted = _capped_round(
        monkeypatch, cap=2, staging=[staged], waiting=[first, second]
    )

    assert admitted == [staged, first]
    assert scheduler.waiting_queue == [second]


def test_model_worker_fdfo_forwards_carried_states_and_all_result_fields() -> None:
    carried_states = [{"round": 1}, None]
    next_states = [{"round": 2}, None]
    calls = []

    class Algorithm:
        fdfo = True

        def run(self, model_runner, forward_batch, algo_states):
            calls.append((model_runner, forward_batch, algo_states))
            return (
                "logits",
                [[10, 11, 12, 13], [20, 21, 22, 23]],
                [0, 4],
                next_states,
                True,
            )

    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = Algorithm()
    worker.model_runner = object()
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(dllm_algo_state=carried_states[0]),
            SimpleNamespace(dllm_algo_state=carried_states[1]),
        ]
    )

    result = ModelWorker.forward_batch_generation(
        worker,
        "forward-batch",
        batch=batch,
    )

    assert calls == [(worker.model_runner, "forward-batch", carried_states)]
    assert result.next_token_ids == [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
    ]
    assert result.accept_length_per_req_cpu == [0, 4]
    assert result.dllm_algo_state == next_states
    assert result.can_run_cuda_graph is True


def test_model_worker_sync_dllm_accepts_five_field_result() -> None:
    class Algorithm:
        fdfo = False

        def run(self, model_runner, forward_batch, algo_states):
            assert algo_states is None
            return ("logits", [[10, 11]], None, None, False)

    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = Algorithm()
    worker.model_runner = object()

    result = ModelWorker.forward_batch_generation(worker, "forward-batch")

    assert result.logits_output == "logits"
    assert result.next_token_ids == [[10, 11]]
    assert result.accept_length_per_req_cpu is None
    assert result.dllm_algo_state is None
    assert result.can_run_cuda_graph is False


def test_dllm_scheduler_event_loop_passes_schedule_batch_to_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = object.__new__(DllmScheduler)
    batch = SimpleNamespace(output_ids=None)
    forwarded = []

    scheduler.running = True
    scheduler.drain_and_purge = lambda: None
    scheduler.schedule_next_batch = lambda: batch
    scheduler.apply_results = lambda *_: None

    def stop_after_step(_batch) -> None:
        scheduler.running = False

    scheduler.post_step = stop_after_step
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(device="cpu"),
        forward_batch_generation=lambda forward_batch, *, batch: (
            forwarded.append((forward_batch, batch))
            or SimpleNamespace(next_token_ids=[])
        ),
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "resolve_deferred_prefill_inputs",
        lambda *_: None,
    )
    monkeypatch.setattr(
        dllm_scheduler_module,
        "ForwardBatch",
        SimpleNamespace(init_new=lambda *args, **kwargs: "forward-batch"),
    )

    scheduler._event_loop()  # noqa: leading-underscore  # production name

    assert forwarded == [("forward-batch", batch)]


def test_dllm_staging_admission_uses_dllm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.runtime_context import get_context

    scheduler = make_scheduler(fdfo=True)
    scheduler.tree_cache = object()
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.req_to_token_pool = object()
    scheduler.model_config = object()
    scheduler.chunked_prefill_size = 16
    scheduler.waiting_queue = []
    req = SimpleNamespace(
        rid="req",
        inflight_middle_chunks=0,
        init_next_round_input=lambda: None,
    )
    scheduler.staging_queue = [req]
    created = {}

    class Adder:
        def __init__(self, *args, **kwargs) -> None:
            created["dllm_config"] = kwargs.get("dllm_config")
            self.can_run_list = []

        def add_dllm_staging_req(self, value):
            created["staging_req"] = value
            self.can_run_list.append(value)
            return dllm_scheduler_module.AddReqResult.CONTINUE

    class Batch:
        @staticmethod
        def init_new(**kwargs):
            created["batch_reqs"] = kwargs["reqs"]
            return SimpleNamespace(prepare_for_extend=lambda: None)

    monkeypatch.setattr(dllm_scheduler_module, "PrefillAdder", Adder)
    monkeypatch.setattr(dllm_scheduler_module, "ScheduleBatch", Batch)

    with get_context().override_server_args(page_size=1, max_prefill_tokens=16):
        batch = scheduler.schedule_next_batch()

    assert batch is not None
    assert created["dllm_config"] is scheduler.dllm_config
    assert created["staging_req"] is req
    assert created["batch_reqs"] == [req]


def test_fdfo_unresolved_block_carries_tokens_state_and_resident_kv() -> None:
    scheduler = make_scheduler(fdfo=True)
    req = ReqDouble()
    scheduler.staging_queue = [req]
    cache_calls = []
    free_calls = []
    scheduler.tree_cache = SimpleNamespace(
        cache_unfinished_req=lambda *args, **kwargs: cache_calls.append((args, kwargs))
    )
    scheduler.req_to_token_pool = SimpleNamespace(
        free=lambda value: free_calls.append(value)
    )
    batch = SimpleNamespace(
        reqs=[req],
        filter_batch=lambda **kwargs: None,
    )
    state = {"round": 2}
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=[0],
        dllm_algo_state=[state],
    )

    scheduler.apply_results(batch, result)
    scheduler.post_step(batch)

    assert req.dllm_incomplete_ids == array("q", [10, 11, 12, 13])
    assert req.dllm_algo_state is state
    assert req.output_ids == []
    assert req.accepted_lengths == []
    assert scheduler.staging_queue == [req]
    assert cache_calls == []
    assert free_calls == []
    assert req.kv.req_pool_idx == 3


def test_fdfo_resolved_block_commits_fill_ids_and_output_tokens() -> None:
    scheduler = make_scheduler(fdfo=True)
    req = ReqDouble()
    req.dllm_incomplete_ids = array("q", [7, 8, 9, 10])
    req.dllm_algo_state = {"round": 1}
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=[4],
        dllm_algo_state=[None],
    )

    scheduler.apply_results(batch, result)

    assert req.dllm_incomplete_ids == array("q")
    assert req.dllm_algo_state is None
    assert req.full_untruncated_fill_ids == array("q", [1, 2, 10, 11, 12, 13])
    assert req.output_ids == [10, 11, 12, 13]
    assert req.accepted_lengths == [4]


def test_fdfo_result_requires_accept_lengths() -> None:
    scheduler = make_scheduler(fdfo=True)
    batch = SimpleNamespace(reqs=[ReqDouble()])
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    with pytest.raises(AssertionError, match="missing accept lengths"):
        scheduler.apply_results(batch, result)


def test_sync_dllm_result_commits_generated_suffix() -> None:
    scheduler = make_scheduler(fdfo=False)
    req = ReqDouble()
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(
        next_token_ids=[[10, 11]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    scheduler.apply_results(batch, result)

    assert req.full_untruncated_fill_ids == array("q", [1, 2, -1, -1, 10, 11])
    assert req.output_ids == [10, 11]
    assert req.accepted_lengths == [2]
