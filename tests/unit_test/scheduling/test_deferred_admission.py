# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import DeferredAdmission


class StubScheduler:
    admit_or_defer_built_request = OmniScheduler.admit_or_defer_built_request
    drain_request_admission_results = OmniScheduler.drain_request_admission_results

    def __init__(self) -> None:
        self.request_admission_lock = threading.RLock()
        self.pending_request_admissions: dict = {}
        self.aborted_request_ids: set[str] = set()
        self.admitted: list[str] = []
        self.errors: list[tuple[str, Exception]] = []

    def enqueue_built_request(
        self,
        payload,
        pending_stream_done,
        req_data,
        *,
        request_admission_lock_held=False,
    ) -> None:
        del pending_stream_done, request_admission_lock_held
        self.admitted.append(req_data.req.rid)

    def emit_request_error(self, request_id: str, exc: Exception) -> None:
        self.errors.append((request_id, exc))

    def abort(self, request_id: str) -> None:
        self.aborted_request_ids.add(request_id)
        self.pending_request_admissions.pop(request_id, None)


def make_deferred(request_id: str):  # noqa: ANN202
    future: concurrent.futures.Future[None] = concurrent.futures.Future()
    payload = SimpleNamespace(request_id=request_id)
    value = SimpleNamespace(req=SimpleNamespace(rid=request_id))
    return payload, value, future, DeferredAdmission(value=value, ready=future)


def test_request_waits_outside_lm_queue_until_dependency_completes() -> None:
    scheduler = StubScheduler()
    payload, _, future, deferred = make_deferred("r1")

    scheduler.admit_or_defer_built_request(payload, False, deferred)
    assert scheduler.admitted == []
    assert list(scheduler.pending_request_admissions) == ["r1"]

    future.set_result(None)
    scheduler.drain_request_admission_results()
    assert scheduler.admitted == ["r1"]
    assert scheduler.pending_request_admissions == {}


def test_aborted_deferred_request_is_never_admitted() -> None:
    scheduler = StubScheduler()
    payload, _, future, deferred = make_deferred("r1")
    scheduler.admit_or_defer_built_request(payload, False, deferred)

    scheduler.abort("r1")
    future.set_result(None)
    scheduler.drain_request_admission_results()

    assert scheduler.admitted == []
    assert not future.cancelled()


def test_failed_dependency_emits_error_without_admission() -> None:
    scheduler = StubScheduler()
    payload, _, future, deferred = make_deferred("r1")
    scheduler.admit_or_defer_built_request(payload, False, deferred)

    future.set_exception(RuntimeError("encode failed"))
    scheduler.drain_request_admission_results()

    assert scheduler.admitted == []
    assert len(scheduler.errors) == 1
    assert scheduler.errors[0][0] == "r1"
    assert "encode failed" in str(scheduler.errors[0][1])
    assert "r1" in scheduler.aborted_request_ids


class WaitPolicyScheduler:
    request_build_queue_fits_workers = OmniScheduler.request_build_queue_fits_workers

    def __init__(
        self,
        *,
        workers: int = 8,
        pending: int = 0,
        backlog: int = 0,
        has_executor: bool = True,
    ) -> None:
        self.request_admission_lock = threading.RLock()
        self.request_build_max_workers = workers
        self.request_build_executor = object() if has_executor else None
        self.pending_request_builds = {f"p{i}": None for i in range(pending)}
        self.backlogged_request_build_payloads = [object() for _ in range(backlog)]


def test_request_build_queue_fits_workers_when_builds_fit_in_pool() -> None:
    scheduler = WaitPolicyScheduler(workers=8, pending=8, backlog=0)
    assert scheduler.request_build_queue_fits_workers() is True

    scheduler = WaitPolicyScheduler(workers=8, pending=1, backlog=0)
    assert scheduler.request_build_queue_fits_workers() is True


def test_defer_admission_when_build_backlog_exceeds_workers() -> None:
    scheduler = WaitPolicyScheduler(workers=8, pending=8, backlog=1)
    assert scheduler.request_build_queue_fits_workers() is False

    scheduler = WaitPolicyScheduler(workers=8, pending=9, backlog=0)
    assert scheduler.request_build_queue_fits_workers() is False


def test_keep_deferred_admission_without_build_executor() -> None:
    scheduler = WaitPolicyScheduler(workers=1, pending=0, backlog=0, has_executor=False)
    assert scheduler.request_build_queue_fits_workers() is False


@pytest.mark.parametrize(("stop", "stop_regex"), [("X", None), (None, "X+")])
def test_dflash_rejects_stop_patterns_before_scheduling(
    stop: str | None, stop_regex: str | None
) -> None:
    scheduler = Mock(spec=OmniScheduler)
    scheduler.deferred_request_payloads = {}
    scheduler.active_session_unit.return_value = None
    scheduler.draft_worker = Mock()
    sampling = SamplingParams(temperature=0, stop=stop, stop_regex=stop_regex)
    request = SimpleNamespace(sampling_params=sampling)

    OmniScheduler.enqueue_built_request(
        scheduler,
        SimpleNamespace(request_id="explicit-stop"),
        False,
        SimpleNamespace(req=request),
    )

    scheduler.abort.assert_called_once_with("explicit-stop")
    error = scheduler.emit_request_error.call_args.args[1]
    assert isinstance(error, ValueError)
    assert "stop strings or stop regexes" in str(error)
    scheduler.normalize_req_token_arrays.assert_not_called()


@pytest.mark.parametrize("has_draft_worker", [False, True])
@pytest.mark.parametrize("stop_strings", [None, []])
def test_supported_stop_configuration_reaches_normal_admission(
    has_draft_worker: bool,
    stop_strings: list[str] | None,
) -> None:
    scheduler = Mock(spec=OmniScheduler)
    scheduler.deferred_request_payloads = {}
    scheduler.active_session_unit.return_value = None
    scheduler.draft_worker = Mock() if has_draft_worker else None
    sampling = SamplingParams(
        temperature=0,
        stop=stop_strings if has_draft_worker else "X",
        stop_regex=stop_strings,
        stop_token_ids=[99],
        max_new_tokens=17,
    )
    if has_draft_worker:
        sampling.normalize(None)
    else:
        pass
    request = SimpleNamespace(
        rid="supported-stop",
        origin_input_ids=[],
        sampling_params=sampling,
        custom_logit_processor=None,
        return_logprob=False,
        return_hidden_states=False,
        return_sampling_mask=False,
    )

    OmniScheduler.enqueue_built_request(
        scheduler,
        SimpleNamespace(request_id=request.rid),
        False,
        SimpleNamespace(req=request, return_logprob=False),
    )

    scheduler.normalize_req_token_arrays.assert_called_once_with(request)
    error = scheduler.emit_request_error.call_args.args[1]
    assert str(error).startswith("Request has no prompt tokens")
    assert sampling.stop_token_ids == {99}
    assert sampling.max_new_tokens == 17
