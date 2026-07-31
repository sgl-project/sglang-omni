# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from array import array
from types import SimpleNamespace

import pytest

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.scheduling import dllm_scheduler as dllm_scheduler_module
from sglang_omni.scheduling.dllm_scheduler import DllmScheduler


class _ReqDouble:
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
        self.req_pool_idx = 3
        self.accepted_lengths: list[int] = []
        self._finished = False

    def update_finish_state(self, *, new_accepted_len: int = 1) -> None:
        self.accepted_lengths.append(new_accepted_len)

    def finished(self) -> bool:
        return self._finished


def _scheduler(*, fdfo: bool, block_size: int = 4) -> DllmScheduler:
    scheduler = object.__new__(DllmScheduler)
    scheduler.dllm_config = SimpleNamespace(
        first_done_first_out_mode=fdfo,
        block_size=block_size,
    )
    scheduler._rid_to_req_data = {}
    scheduler._result_adapter = lambda value: value
    scheduler.outbox = SimpleNamespace(put=lambda value: None)
    return scheduler


def test_model_worker_fdfo_forwards_carried_states_and_all_result_fields() -> None:
    carried_states = [{"round": 1}, None]
    next_states = [{"round": 2}, None]
    calls = []

    class _Algorithm:
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
    worker.dllm_algorithm = _Algorithm()
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
    class _Algorithm:
        fdfo = False

        def run(self, model_runner, forward_batch, algo_states):
            assert algo_states is None
            return ("logits", [[10, 11]], None, None, False)

    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = _Algorithm()
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

    scheduler._running = True
    scheduler._drain_and_purge = lambda: None
    scheduler._schedule_next_batch = lambda: batch
    scheduler._apply_results = lambda *_: None

    def stop_after_step(_batch) -> None:
        scheduler._running = False

    scheduler._post_step = stop_after_step
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

    scheduler._event_loop()

    assert forwarded == [("forward-batch", batch)]


def test_dllm_staging_admission_uses_dllm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(fdfo=True)
    scheduler.server_args = SimpleNamespace(
        page_size=1,
        max_prefill_tokens=16,
    )
    scheduler.tree_cache = object()
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.req_to_token_pool = object()
    scheduler.model_config = object()
    scheduler._chunked_prefill_size = 16
    scheduler._waiting_queue = []
    req = SimpleNamespace(
        rid="req",
        inflight_middle_chunks=0,
        init_next_round_input=lambda: None,
    )
    scheduler._staging_queue = [req]
    created = {}

    class _Adder:
        def __init__(self, *args, **kwargs) -> None:
            created["dllm_config"] = kwargs.get("dllm_config")
            self.can_run_list = []

        def add_dllm_staging_req(self, value):
            created["staging_req"] = value
            self.can_run_list.append(value)
            return dllm_scheduler_module.AddReqResult.CONTINUE

    class _Batch:
        @staticmethod
        def init_new(**kwargs):
            created["batch_reqs"] = kwargs["reqs"]
            return SimpleNamespace(prepare_for_extend=lambda: None)

    monkeypatch.setattr(dllm_scheduler_module, "PrefillAdder", _Adder)
    monkeypatch.setattr(dllm_scheduler_module, "ScheduleBatch", _Batch)

    batch = scheduler._schedule_next_batch()

    assert batch is not None
    assert created["dllm_config"] is scheduler.dllm_config
    assert created["staging_req"] is req
    assert created["batch_reqs"] == [req]


def test_fdfo_unresolved_block_carries_tokens_state_and_resident_kv() -> None:
    scheduler = _scheduler(fdfo=True)
    req = _ReqDouble()
    scheduler._staging_queue = [req]
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

    scheduler._apply_results(batch, result)
    scheduler._post_step(batch)

    assert req.dllm_incomplete_ids == array("q", [10, 11, 12, 13])
    assert req.dllm_algo_state is state
    assert req.output_ids == []
    assert req.accepted_lengths == []
    assert scheduler._staging_queue == [req]
    assert cache_calls == []
    assert free_calls == []
    assert req.req_pool_idx == 3


def test_fdfo_resolved_block_commits_fill_ids_and_output_tokens() -> None:
    scheduler = _scheduler(fdfo=True)
    req = _ReqDouble()
    req.dllm_incomplete_ids = array("q", [7, 8, 9, 10])
    req.dllm_algo_state = {"round": 1}
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=[4],
        dllm_algo_state=[None],
    )

    scheduler._apply_results(batch, result)

    assert req.dllm_incomplete_ids == array("q")
    assert req.dllm_algo_state is None
    assert req.full_untruncated_fill_ids == array("q", [1, 2, 10, 11, 12, 13])
    assert req.output_ids == [10, 11, 12, 13]
    assert req.accepted_lengths == [4]


def test_fdfo_result_requires_accept_lengths() -> None:
    scheduler = _scheduler(fdfo=True)
    batch = SimpleNamespace(reqs=[_ReqDouble()])
    result = SimpleNamespace(
        next_token_ids=[[10, 11, 12, 13]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    with pytest.raises(AssertionError, match="missing accept lengths"):
        scheduler._apply_results(batch, result)


def test_sync_dllm_result_commits_generated_suffix() -> None:
    scheduler = _scheduler(fdfo=False)
    req = _ReqDouble()
    batch = SimpleNamespace(reqs=[req])
    result = SimpleNamespace(
        next_token_ids=[[10, 11]],
        accept_length_per_req_cpu=None,
        dllm_algo_state=None,
    )

    scheduler._apply_results(batch, result)

    assert req.full_untruncated_fill_ids == array("q", [1, 2, -1, -1, 10, 11])
    assert req.output_ids == [10, 11]
    assert req.accepted_lengths == [2]
