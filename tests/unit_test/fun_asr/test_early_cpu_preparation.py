# SPDX-License-Identifier: Apache-2.0
"""Exercise real ingress/build control flow with CPU audio and no model worker."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.fun_asr.request_builders as builders
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto.session import SESSION_METADATA_KEY
from tests.unit_test.fixtures.fun_asr import harness  # noqa: F401
from tests.unit_test.fixtures.fun_asr import await_build, enqueue, payload, process


@pytest.mark.parametrize("early", [False, True])
def test_stage_arrival_prepares_during_run_without_admission(harness, early):
    """An occupied synchronous loop used to delay all input preparation."""
    scheduler = harness.scheduler
    run_started, release_run = threading.Event(), threading.Event()
    admitted = threading.Event()
    errors = []
    scheduler.running = True
    scheduler._engine_paused = False  # noqa: leading-underscore
    scheduler.process_admin_requests = lambda: None
    scheduler.self_check_during_idle = lambda: None

    def await_pending_build():
        with scheduler.request_admission_lock:
            pending = tuple(scheduler.pending_request_builds.values())
        for _, _, future in pending:
            future.result(timeout=10)

    scheduler.sleep_during_idle = await_pending_build
    scheduler.process_batch_result = lambda batch, result: None
    first = True

    def select_batch():
        nonlocal first
        if first:
            first = False
            return SimpleNamespace(reqs=[])
        if harness.admitted:
            scheduler.running = False
            admitted.set()
        return None

    def run_batch(batch):
        run_started.set()
        assert release_run.wait(10)
        return None

    scheduler.get_next_batch_to_run = select_batch
    scheduler.run_batch = run_batch

    def run():
        scheduler.scheduler_thread_id = threading.get_ident()
        try:
            scheduler.event_loop_normal()
        except Exception as error:
            errors.append(error)
            admitted.set()

    stage = Stage.__new__(Stage)
    stage.scheduler = scheduler
    stage.aborted = set()
    stage.role = "single"
    stage.name = "asr"
    stage.tp_fanout = None
    stage.owns_external_io = True
    if not early:
        scheduler.enqueue = scheduler.inbox.put
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert run_started.wait(10)
        asyncio.run(stage.execute(payload()))
        if early:
            assert harness.frontend.started.wait(10)
        else:
            assert not harness.frontend.started.is_set()
        assert not harness.encoder.started.is_set()
        assert not harness.admitted
        assert scheduler.outbox.empty()
        release_run.set()
        assert admitted.wait(10)
    finally:
        release_run.set()
        scheduler.running = False
        thread.join(10)
    assert not thread.is_alive()
    assert not errors
    assert len(harness.admitted) == 1
    assert harness.frontend.calls == harness.encoder.calls == 1
    assert harness.admission_threads == [thread.ident]
    assert scheduler.cpu_preparation is None


def test_inflight_preparation_chains_without_blocking_single_worker(harness):
    scheduler = harness.scheduler
    scheduler.request_build_executor.shutdown()
    scheduler.request_build_executor = ThreadPoolExecutor(max_workers=1)
    harness.frontend.release.clear()
    request = payload()
    enqueue(scheduler, request)
    assert harness.frontend.started.wait(10)
    process(scheduler)
    assert not scheduler.pending_request_builds[request.request_id][2].done()
    assert not harness.encoder.started.is_set()
    harness.frontend.release.set()
    await_build(scheduler, request.request_id)
    assert len(harness.admitted) == 1
    assert harness.frontend.calls == harness.encoder.calls == 1


def test_completed_preparation_is_retained_and_parameters_are_snapshotted(harness):
    scheduler = harness.scheduler
    request = payload()
    request.request.params.update(hotwords=["original"], temperature=0.0)
    enqueue(scheduler, request)
    preparation = scheduler.cpu_preparation
    prepared = preparation.cpu_future.result(timeout=10)
    request.request.params["hotwords"].append("later")
    request.request.params["temperature"] = 0.7
    enqueue(scheduler, payload("second"))
    assert scheduler.cpu_preparation is preparation
    assert harness.encoder.calls == 0
    process(scheduler)
    await_build(scheduler, request.request_id)
    await_build(scheduler, "second")
    assert harness.admitted[0].temperature == 0.0
    assert harness.admitted[0].prompt_token_ids == prepared.prompt_token_ids
    assert [result.req.rid for result in harness.admitted] == ["audio", "second"]


def test_duplicate_messages_do_not_prepare_or_encode_twice(harness):
    request = payload()
    harness.frontend.release.clear()
    enqueue(harness.scheduler, request)
    assert harness.frontend.started.wait(10)
    enqueue(harness.scheduler, request)
    enqueue(harness.scheduler, payload(request.request_id))
    process(harness.scheduler)
    harness.frontend.release.set()
    await_build(harness.scheduler, request.request_id)
    assert len(harness.admitted) == 1
    assert harness.frontend.calls == harness.encoder.calls == 1


def test_prepare_error_is_only_reported_after_scheduler_claim(harness):
    request = payload()
    request.request.params["max_new_tokens"] = -1
    enqueue(harness.scheduler, request)
    preparation = harness.scheduler.cpu_preparation
    with pytest.raises(ValueError, match="max_new_tokens"):
        preparation.cpu_future.result(timeout=10)
    assert harness.scheduler.outbox.empty()
    process(harness.scheduler)
    error = harness.scheduler.outbox.get_nowait()
    assert error.type == "error"
    assert error.request_id == request.request_id
    assert harness.encoder.calls == 0
    assert harness.scheduler.cpu_preparation is None
    assert not harness.admitted


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_reservation_is_bounded_without_double_counting_admission(harness, limit):
    scheduler = harness.scheduler
    scheduler.request_build_max_pending = limit
    scheduler.max_queued_requests = 2
    harness.frontend.release.clear()
    first, second = payload("first"), payload("second")
    enqueue(scheduler, first)
    enqueue(scheduler, second)
    assert scheduler.queued_admission_count() == 0
    process(scheduler)
    assert scheduler.request_build_slots_in_use() <= limit
    assert scheduler.outbox.empty()
    harness.frontend.release.set()
    await_build(scheduler, "first")
    process(scheduler)
    await_build(scheduler, "second")
    assert [result.req.rid for result in harness.admitted] == ["first", "second"]
    assert scheduler.outbox.empty()
    assert scheduler.request_build_slots_in_use() == 0


@pytest.mark.parametrize(
    "source", ["path", "mutable", "session", "refresh", "prefetched"]
)
def test_unsupported_inputs_are_not_prepared_at_ingress(harness, source):
    request = payload()
    if source == "path":
        request.request.inputs = {"audio_path": "/unused.wav"}
    elif source == "mutable":
        request.request.inputs["audio_bytes"] = bytearray(
            request.request.inputs["audio_bytes"]
        )
    elif source == "session":
        request.request.metadata[SESSION_METADATA_KEY] = {}
    elif source == "refresh":
        request.request.params["_asr_streaming"] = True
    else:
        request.prefetched_stream_done = True
    enqueue(harness.scheduler, request)
    assert harness.scheduler.cpu_preparation is None
    assert not harness.frontend.started.is_set()
    assert harness.scheduler.inbox.get_nowait().data is request


def test_reused_identifier_cannot_adopt_a_previous_payload_preparation(harness):
    scheduler = harness.scheduler
    harness.frontend.release.clear()
    first = payload()
    enqueue(scheduler, first)
    assert harness.frontend.started.wait(10)
    scheduler.abort(first.request_id)
    scheduler.aborted_request_ids.discard(first.request_id)
    second = payload(first.request_id)
    second.request.params["temperature"] = 0.7
    enqueue(scheduler, second)
    process(scheduler)
    harness.frontend.release.set()
    await_build(scheduler, second.request_id)
    assert len(harness.admitted) == 1
    assert harness.admitted[0].temperature == 0.7
    assert harness.encoder.calls == 1


def test_queue_rejection_discards_already_prepared_input(harness):
    scheduler = harness.scheduler
    enqueue(scheduler, payload())
    scheduler.cpu_preparation.cpu_future.result(timeout=10)
    scheduler.max_queued_requests = 1
    scheduler.waiting_queue = [SimpleNamespace(rid="prior")]
    process(scheduler)
    assert scheduler.cpu_preparation is None
    assert harness.encoder.calls == 0
    assert not harness.admitted
    assert scheduler.outbox.get_nowait().type == "error"


def test_backlog_keeps_priority_over_early_preparation(harness):
    scheduler = harness.scheduler
    scheduler.backlogged_request_build_payloads.append(payload("prior"))
    enqueue(scheduler, payload("next"))
    assert scheduler.cpu_preparation is None
    process(scheduler)
    await_build(scheduler, "prior")
    await_build(scheduler, "next")
    assert [result.req.rid for result in harness.admitted] == ["prior", "next"]


@pytest.mark.parametrize("restriction", ["tp", "overlap", "stream_input", "full_pool"])
def test_scheduler_restrictions_keep_the_original_inbox_path(harness, restriction):
    scheduler = harness.scheduler
    if restriction == "tp":
        scheduler.tp_size = 2
    elif restriction == "overlap":
        scheduler.enable_overlap = True
    elif restriction == "stream_input":
        scheduler.stream_chunk_handler = lambda request, chunk: None
    else:
        for index in range(scheduler.request_build_max_pending):
            request = payload(str(index))
            scheduler.pending_request_builds[request.request_id] = (
                request,
                False,
                Future(),
            )
    request = payload("new")
    enqueue(scheduler, request)
    assert scheduler.cpu_preparation is None
    assert not harness.frontend.started.is_set()
    assert scheduler.inbox.get_nowait().data is request


def test_early_phase_never_constructs_req_or_calls_encoder(harness, monkeypatch):
    original = builders.Req

    def forbidden_request(*args, **kwargs):
        pytest.fail("early CPU preparation must not construct Req")

    monkeypatch.setattr(builders, "Req", forbidden_request)
    enqueue(harness.scheduler, payload())
    prepared = harness.scheduler.cpu_preparation.cpu_future.result(timeout=10)
    assert prepared.multimodal_inputs.mm_items[0].feature.device.type == "cpu"
    assert harness.encoder.calls == 0
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(builders, "Req", original)
    process(harness.scheduler)
    await_build(harness.scheduler, "audio")
    assert len(harness.admitted) == 1


@pytest.mark.parametrize("abort_head", [False, True])
def test_preparation_does_not_reduce_capacity_at_the_backlog_limit(harness, abort_head):
    scheduler = harness.scheduler
    scheduler.request_build_max_pending = 2
    scheduler.request_build_backlog_limit = 2
    harness.frontend.release.clear()
    enqueue(scheduler, payload("head"))
    assert harness.frontend.started.wait(10)
    if abort_head:
        scheduler.abort("head")
        enqueue(scheduler, payload("replacement"))
    for index in range(3):
        enqueue(scheduler, payload(str(index)))
    process(scheduler)
    assert len(scheduler.pending_request_builds) == 2
    assert len(scheduler.backlogged_request_build_payloads) == 2
    assert scheduler.outbox.empty()
    harness.frontend.release.set()
    for request_id in list(scheduler.pending_request_builds):
        await_build(scheduler, request_id)
    process(scheduler)
    for request_id in list(scheduler.pending_request_builds):
        await_build(scheduler, request_id)
    expected_head = "replacement" if abort_head else "head"
    assert [result.req.rid for result in harness.admitted] == [
        expected_head,
        "0",
        "1",
        "2",
    ]
    assert scheduler.outbox.empty()


def test_ingress_does_not_steal_slots_already_selected_by_scheduler(harness):
    scheduler = harness.scheduler
    selected = threading.Event()
    resume = threading.Event()
    original_stage = scheduler.stage_request_build_payloads
    errors = []

    def stage(requests):
        result = original_stage(requests)
        selected.set()
        assert resume.wait(10)
        return result

    scheduler.stage_request_build_payloads = stage
    scheduler.request_build_max_pending = 2
    scheduler.inbox.put(
        SimpleNamespace(type="new_request", request_id="prior", data=payload("prior"))
    )

    def consume():
        try:
            process(scheduler)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert selected.wait(10)
        enqueue(scheduler, payload("new"))
        assert scheduler.cpu_preparation is None
    finally:
        resume.set()
        worker.join(10)
    assert not worker.is_alive()
    assert not errors
    scheduler.stage_request_build_payloads = original_stage
    await_build(scheduler, "prior")
    process(scheduler)
    await_build(scheduler, "new")
    assert [result.req.rid for result in harness.admitted] == ["prior", "new"]


def test_ingress_cannot_reserve_between_recv_and_processing(harness):
    scheduler = harness.scheduler
    scheduler.request_build_max_pending = 2
    scheduler.request_build_backlog_limit = 2
    received, resume = threading.Event(), threading.Event()
    original = scheduler.drain_local_inbox
    errors = []

    def drain():
        messages = original()
        received.set()
        assert resume.wait(10)
        return messages

    scheduler.drain_local_inbox = drain
    for index in range(4):
        scheduler.inbox.put(
            SimpleNamespace(
                type="new_request", request_id=str(index), data=payload(str(index))
            )
        )

    def consume():
        try:
            process(scheduler)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert received.wait(10)
        enqueue(scheduler, payload("arriving"))
        assert scheduler.cpu_preparation is None
    finally:
        resume.set()
        worker.join(10)
    assert not worker.is_alive()
    assert not errors
    assert scheduler.outbox.empty()
    scheduler.drain_local_inbox = original
    for _ in range(3):
        for request_id in list(scheduler.pending_request_builds):
            await_build(scheduler, request_id)
        process(scheduler)
    assert [result.req.rid for result in harness.admitted] == [
        "0",
        "1",
        "2",
        "3",
        "arriving",
    ]
    assert scheduler.outbox.empty()
