# SPDX-License-Identifier: Apache-2.0
"""Cancellation and shutdown ordering around early Fun-ASR CPU work."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from weakref import ref

import pytest

from tests.unit_test.fixtures.fun_asr import harness  # noqa: F401
from tests.unit_test.fixtures.fun_asr import await_build, enqueue, payload, process


@pytest.mark.parametrize("claimed", [False, True])
def test_abort_running_preparation_discards_result_and_frees_slot(harness, claimed):
    scheduler = harness.scheduler
    harness.frontend.release.clear()
    request = payload()
    enqueue(scheduler, request)
    assert harness.frontend.started.wait(10)
    preparation = scheduler.cpu_preparation
    if claimed:
        process(scheduler)
    scheduler.abort(request.request_id)
    assert scheduler.cpu_preparation is preparation
    assert request.request_id not in scheduler.pending_request_builds
    assert scheduler.inbox.empty()
    completed = threading.Event()
    preparation.cpu_future.add_done_callback(lambda future: completed.set())
    harness.frontend.release.set()
    assert completed.wait(10)
    assert scheduler.cpu_preparation is None
    assert not harness.encoder.started.is_set()
    scheduler.drain_request_build_results()
    assert not harness.admitted
    enqueue(scheduler, payload("next"))
    process(scheduler)
    await_build(scheduler, "next")
    assert [result.req.rid for result in harness.admitted] == ["next"]


def test_abort_during_encoder_does_not_publish_or_close_encoder_early(harness):
    scheduler = harness.scheduler
    harness.encoder.release.clear()
    enqueue(scheduler, payload())
    process(scheduler)
    assert harness.encoder.started.wait(10)
    preparation = scheduler.cpu_preparation
    scheduler.abort("audio")
    assert scheduler.cpu_preparation is preparation
    assert preparation.build_future.cancelled()
    completed = threading.Event()
    preparation.finish_future.add_done_callback(lambda future: completed.set())
    harness.encoder.release.set()
    assert completed.wait(10)
    assert scheduler.cpu_preparation is None
    scheduler.drain_request_build_results()
    assert not harness.admitted
    assert scheduler.outbox.empty()


@pytest.mark.parametrize("phase", ["cpu", "encoder"])
def test_shutdown_waits_for_running_work_before_encoder_close(harness, phase):
    scheduler = harness.scheduler
    gate = harness.frontend if phase == "cpu" else harness.encoder
    gate.release.clear()
    enqueue(scheduler, payload())
    if phase == "encoder":
        process(scheduler)
    assert gate.started.wait(10)
    entered_shutdown = threading.Event()
    original_cancel = scheduler.cancel_cpu_preparation

    def cancel(preparation):
        original_cancel(preparation)
        entered_shutdown.set()

    scheduler.cancel_cpu_preparation = cancel
    first = threading.Thread(target=scheduler.shutdown_resources)
    second = threading.Thread(target=scheduler.shutdown_resources)
    first.start()
    assert entered_shutdown.wait(10)
    second.start()
    assert not harness.encoder.closed.is_set()
    gate.release.set()
    first.join(10)
    second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert harness.encoder.closed.is_set()
    assert scheduler.cpu_preparation is None
    assert not harness.admitted
    assert harness.encoder.calls == (phase == "encoder")
    enqueue(scheduler, payload("after-close"))
    assert scheduler.cpu_preparation is None


@pytest.mark.parametrize("phase", ["cpu_queued", "finish_queued"])
def test_cancelled_work_holds_reservation_until_executor_consumes_it(harness, phase):
    scheduler = harness.scheduler
    request = payload()
    if phase == "finish_queued":
        enqueue(scheduler, request)
        scheduler.cpu_preparation.cpu_future.result(timeout=10)
    release = threading.Event()
    started = [threading.Event(), threading.Event()]

    def occupy(event):
        event.set()
        assert release.wait(10)

    blockers = [
        scheduler.request_build_executor.submit(occupy, event) for event in started
    ]
    try:
        assert all(event.wait(10) for event in started)
        if phase == "cpu_queued":
            enqueue(scheduler, request)
        else:
            process(scheduler)
        preparation = scheduler.cpu_preparation
        scheduler.abort(request.request_id)
        assert scheduler.cpu_preparation is preparation
        assert scheduler.request_build_slots_in_use() == 0
        enqueue(scheduler, payload("later"))
        assert scheduler.cpu_preparation is preparation
        consumed = threading.Event()
        future = (
            preparation.cpu_future
            if phase == "cpu_queued"
            else preparation.finish_future
        )
        future.add_done_callback(lambda result: consumed.set())
        release.set()
        assert consumed.wait(10)
        assert scheduler.cpu_preparation is None
        assert harness.encoder.calls == 0
        assert harness.frontend.calls == (phase == "finish_queued")
        process(scheduler)
        await_build(scheduler, "later")
        assert [result.req.rid for result in harness.admitted] == ["later"]
    finally:
        release.set()
        for blocker in blockers:
            blocker.result(timeout=10)


@pytest.mark.parametrize("phase", ["early_submit", "finish_submit", "encoder"])
def test_submit_and_encoder_failures_do_not_leak_or_admit(harness, monkeypatch, phase):
    scheduler = harness.scheduler
    request = payload()
    executor = scheduler.request_build_executor
    original_submit = executor.submit

    def fail_submission(*args, **kwargs):
        monkeypatch.setattr(executor, "submit", original_submit)
        raise RuntimeError("injected submit failure")

    if phase == "early_submit":
        monkeypatch.setattr(executor, "submit", fail_submission)
        enqueue(scheduler, request)
        assert scheduler.cpu_preparation is None
    else:
        enqueue(scheduler, request)
        scheduler.cpu_preparation.cpu_future.result(timeout=10)
        if phase == "finish_submit":
            monkeypatch.setattr(executor, "submit", fail_submission)
        else:

            def fail_encoder(item):
                raise ValueError("injected encoder failure")

            monkeypatch.setattr(harness.encoder, "encode_item", fail_encoder)
    process(scheduler)
    if phase == "early_submit":
        await_build(scheduler, request.request_id)
        assert len(harness.admitted) == 1
        assert harness.frontend.calls == harness.encoder.calls == 1
    else:
        pending = scheduler.pending_request_builds.get(request.request_id)
        if pending is not None:
            with pytest.raises((RuntimeError, ValueError), match="injected"):
                pending[2].result(timeout=10)
            scheduler.drain_request_build_results()
        assert not harness.admitted
        error = scheduler.outbox.get_nowait()
        assert error.type == "error"
        assert error.request_id == request.request_id
        assert scheduler.outbox.empty()
    assert scheduler.cpu_preparation is None
    assert scheduler.request_build_slots_in_use() == 0


def test_completed_preparation_does_not_retain_request_through_future_callbacks(
    harness,
):
    scheduler = harness.scheduler
    scheduler.request_build_executor.shutdown()
    scheduler.request_build_executor = ThreadPoolExecutor(max_workers=1)
    enqueue(scheduler, payload())
    reference = ref(scheduler.cpu_preparation)
    process(scheduler)
    await_build(scheduler, "audio")
    scheduler.request_build_executor.submit(lambda: None).result(timeout=10)
    assert reference() is None
    assert len(harness.admitted) == 1


def test_cpu_failure_racing_with_claim_is_reported_once(harness):
    scheduler = harness.scheduler
    callback_started = threading.Event()
    release_callback = threading.Event()
    callback_finished = threading.Event()
    main_thread = threading.get_ident()
    original = scheduler.cpu_prepared
    errors = []

    def prepared(reference, future):
        if threading.get_ident() != main_thread:
            callback_started.set()
            assert release_callback.wait(10)
            try:
                original(reference, future)
            except Exception as error:
                errors.append(error)
            finally:
                callback_finished.set()
        else:
            original(reference, future)

    scheduler.cpu_prepared = prepared
    request = payload()
    request.request.params["max_new_tokens"] = -1
    try:
        enqueue(scheduler, request)
        preparation = scheduler.cpu_preparation
        assert callback_started.wait(10)
        process(scheduler)
        assert preparation.build_future.done()
        error = scheduler.outbox.get_nowait()
        assert error.type == "error"
        assert scheduler.outbox.empty()
    finally:
        release_callback.set()
    assert callback_finished.wait(10)
    assert not errors
    assert harness.encoder.calls == 0
    assert not harness.admitted
