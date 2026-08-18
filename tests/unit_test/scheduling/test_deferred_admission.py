# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import threading
from types import SimpleNamespace

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import DeferredAdmission


class _StubScheduler:
    _admit_or_defer_built_request = OmniScheduler._admit_or_defer_built_request
    _drain_request_admission_results = OmniScheduler._drain_request_admission_results

    def __init__(self) -> None:
        self._request_admission_lock = threading.RLock()
        self._pending_request_admissions: dict = {}
        self._aborted_request_ids: set[str] = set()
        self.admitted: list[str] = []
        self.errors: list[tuple[str, Exception]] = []

    def _enqueue_built_request(
        self,
        payload,
        pending_stream_done,
        req_data,
        *,
        request_admission_lock_held=False,
    ) -> None:
        del pending_stream_done, request_admission_lock_held
        self.admitted.append(req_data.req.rid)

    def _emit_request_error(self, request_id: str, exc: Exception) -> None:
        self.errors.append((request_id, exc))

    def abort(self, request_id: str) -> None:
        self._aborted_request_ids.add(request_id)
        self._pending_request_admissions.pop(request_id, None)


def _deferred(request_id: str):  # noqa: ANN202
    future: concurrent.futures.Future[None] = concurrent.futures.Future()
    payload = SimpleNamespace(request_id=request_id)
    value = SimpleNamespace(req=SimpleNamespace(rid=request_id))
    return payload, value, future, DeferredAdmission(value=value, ready=future)


def test_request_waits_outside_lm_queue_until_dependency_completes() -> None:
    scheduler = _StubScheduler()
    payload, _, future, deferred = _deferred("r1")

    scheduler._admit_or_defer_built_request(payload, False, deferred)
    assert scheduler.admitted == []
    assert list(scheduler._pending_request_admissions) == ["r1"]

    future.set_result(None)
    scheduler._drain_request_admission_results()
    assert scheduler.admitted == ["r1"]
    assert scheduler._pending_request_admissions == {}


def test_aborted_deferred_request_is_never_admitted() -> None:
    scheduler = _StubScheduler()
    payload, _, future, deferred = _deferred("r1")
    scheduler._admit_or_defer_built_request(payload, False, deferred)

    scheduler.abort("r1")
    future.set_result(None)
    scheduler._drain_request_admission_results()

    assert scheduler.admitted == []
    assert not future.cancelled()


def test_failed_dependency_emits_error_without_admission() -> None:
    scheduler = _StubScheduler()
    payload, _, future, deferred = _deferred("r1")
    scheduler._admit_or_defer_built_request(payload, False, deferred)

    future.set_exception(RuntimeError("encode failed"))
    scheduler._drain_request_admission_results()

    assert scheduler.admitted == []
    assert len(scheduler.errors) == 1
    assert scheduler.errors[0][0] == "r1"
    assert "encode failed" in str(scheduler.errors[0][1])
    assert "r1" in scheduler._aborted_request_ids
