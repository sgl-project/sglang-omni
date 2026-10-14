# SPDX-License-Identifier: Apache-2.0
"""Native SpecV2 ownership with real batches, results and CPU event-order fakes."""
from __future__ import annotations

import queue
import threading
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import FactoryArgs
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.scheduling.spec_v2_helpers import make_owner
from tests.unit_test.scheduling.test_speculative_output import (
    make_request,
    resolve_tokens,
)


@pytest.fixture
def owner():
    return make_owner()


def test_launch_uses_native_relay_isolation_and_does_not_host_wait(owner):
    scheduler, batch, device = owner
    original_lengths = batch.seq_lens
    pending = scheduler.launch_spec_v2(batch)
    assert batch.forward_mode == ForwardMode.DECODE
    assert batch.seq_lens is original_lengths
    assert batch.sampling_info is batch.original_sampling
    assert batch.input_ids is None
    assert batch.spec_info.future_indices is batch.req_pool_indices
    assert pending.result.next_token_ids.shape == (24,)
    assert pending.result.accept_lens.tolist() == [2, 3, 1]
    assert [row.pool_index for row in pending.rows] == [7, 2, 12]
    assert not any(event[0] == "wait" for event in device.log)
    assert device.log.index(("resolve_inputs",)) < device.log.index(
        ("native_forward", ForwardMode.DECODE)
    )
    assert ("stream", "forward", "copy") in device.log
    assert ("record", pending.writer_done.name, "forward") in device.log
    assert ("barrier", "schedule", pending.writer_done.name) in device.log
    assert ("record", pending.result.copy_done.name, "copy") in device.log
    assert (
        scheduler.extra_reference
        is scheduler.batch_record_buf[scheduler.batch_record_ct][-1]
    )
    assert pending.batch.reqs is not batch.reqs
    assert scheduler.stored_results == []


def test_next_native_launch_precedes_previous_host_resolve(owner):
    scheduler, batch, device = owner
    first = scheduler.launch_spec_v2(batch)
    second = scheduler.launch_spec_v2(batch)
    scheduler.pending_spec_v2 = second
    scheduler.resolve_spec_v2(first)
    forwards = [i for i, event in enumerate(device.log) if event[0] == "native_forward"]
    assert len(forwards) == 2
    assert forwards[1] < device.log.index(("wait", first.result.copy_done.name))
    assert scheduler.stored_results == [(first.batch, first.result)]
    assert scheduler.pending_spec_v2 is second


@pytest.mark.parametrize("mutation", ["slot", "generation", "handle", "rid", "object"])
def test_result_rejects_identity_drift_without_changing_flattened_rows(owner, mutation):
    scheduler, batch, device = owner
    pending = scheduler.launch_spec_v2(batch)
    if mutation == "slot":
        batch.reqs[1].kv.req_pool_idx = 9
    elif mutation == "generation":
        scheduler.req_to_token_pool.req_generation[2] += 1
    elif mutation == "handle":
        batch.reqs[1].cache_request_handle = Mock()
    elif mutation == "rid":
        batch.reqs[1].rid = "replacement"
    else:
        pending.batch.reqs[1] = make_request("b", 2, 20)
    with pytest.raises(RuntimeError, match="identity|generation"):
        scheduler.resolve_spec_v2(pending)
    assert pending.result.next_token_ids.shape == (24,)
    assert scheduler.stored_results == []


def test_native_resolver_keeps_padded_rows_with_finished_overshoot(owner):
    scheduler, batch, device = owner
    pending = scheduler.launch_spec_v2(batch)
    batch.reqs[0].finished_reason = FINISH_LENGTH(20)
    batch.reqs[0].kv.req_pool_idx = None
    scheduler.resolve_spec_v2(pending)
    assert scheduler.stored_results[0][0].reqs == batch.reqs
    resolved = resolve_tokens(
        batch,
        [list(range(10, 18)), list(range(20, 28)), list(range(30, 38))],
        [2, 3, 1],
    )
    assert resolved[1:] == [[20, 21, 22], [30]]


def test_drain_retains_pending_until_writer_and_copy_finish(owner):
    scheduler, batch, device = owner
    pending = scheduler.launch_spec_v2(batch)
    scheduler.pending_spec_v2 = pending
    scheduler.drain_spec_v2()
    assert device.log.index(("wait", pending.writer_done.name)) < device.log.index(
        ("wait", pending.result.copy_done.name)
    )
    assert scheduler.pending_spec_v2 is None
    assert scheduler.stored_results == [(pending.batch, pending.result)]


def test_failed_device_drain_cannot_release_pending(owner):
    scheduler, batch, device = owner
    pending = scheduler.launch_spec_v2(batch)
    scheduler.pending_spec_v2 = pending
    pending.writer_done.fail = True
    scheduler.terminate_prefill_worker = Mock(side_effect=SystemExit("retire"))
    with pytest.raises(SystemExit, match="retire"):
        scheduler.drain_spec_v2()
    assert scheduler.pending_spec_v2 is pending
    assert not scheduler.stored_results
    scheduler.handle_batch_failure.assert_not_called()


def test_launch_failure_drains_all_streams_before_cleanup(owner):
    scheduler, batch, device = owner
    pending = scheduler.launch_spec_v2(batch)
    scheduler.pending_spec_v2 = pending
    scheduler.handle_batch_failure.side_effect = lambda batch, error: device.log.append(
        ("cleanup",)
    )
    scheduler.fail_spec_v2(batch, RuntimeError("submission failed"))
    cleanup = device.log.index(("cleanup",))
    assert all(
        device.log.index(("drain", name)) < cleanup
        for name in ("forward", "copy", "schedule")
    )


def test_sync_prefill_seeds_future_map_before_next_decode(owner):
    scheduler, batch, device = owner
    batch.forward_mode = ForwardMode.EXTEND
    batch.spec_info = None
    result = scheduler._run_batch(
        batch
    )  # noqa: leading-underscore  # Actual synchronous forward boundary.
    assert ("publish_lengths", [7, 2, 12]) in device.log
    assert ("stash", [7, 2, 12]) in device.log
    assert batch.spec_info is result.next_draft_input
    assert batch.spec_info.future_indices is batch.req_pool_indices
    assert scheduler.wait_prefill_publication.called


@pytest.mark.parametrize(
    "mode",
    [
        ForwardMode.EXTEND,
        ForwardMode.MIXED,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.IDLE,
    ],
)
def test_launch_rejects_non_decode_scheduler_modes(owner, mode):
    scheduler, batch, device = owner
    batch.forward_mode = mode
    with pytest.raises(ValueError, match="prepared DECODE"):
        scheduler.launch_spec_v2(batch)
    assert device.log == []


def test_actual_admin_entry_drains_even_without_queue_snapshot(owner):
    scheduler, batch, device = owner
    scheduler.pending_spec_v2 = scheduler.launch_spec_v2(batch)
    scheduler.admin_pause_generation = Mock(return_value={"success": True})
    scheduler.run_admin_action("pause_generation", {})
    assert scheduler.pending_spec_v2 is None
    scheduler.admin_pause_generation.assert_called_once()


def test_stopped_external_admin_cannot_run_during_retained_shutdown(owner):
    scheduler, batch, device = owner
    scheduler.scheduler_thread_id = threading.get_ident() + 1
    scheduler.running = False
    scheduler.run_admin_action = Mock()
    assert scheduler.admin("pause_generation")["success"] is False
    scheduler.run_admin_action.assert_not_called()


@pytest.mark.parametrize("flag", ["enable_spec_v2_overlap"])
def test_independent_typed_flag_defaults_disabled_and_round_trips(flag: str) -> None:
    configuration = Qwen3OmniPipelineConfig(model_path="unused")
    factory_values = configuration.stage_named("thinker").factory.model_dump(
        exclude_none=True
    )
    assert not factory_values.get(flag, False)
    for text, expected in (("false", False), ("true", True)):
        direct = FactoryArgs(**{flag: text}).model_dump(exclude_none=True)
        assert direct[flag] is expected
        resolved = ConfigManager(configuration).merge_config(
            [(f"thinker.factory.{flag}", text)]
        )
        assert getattr(resolved.stage_named("thinker").factory, flag) is expected
        assert resolved.stage_named("thinker").factory.enable_async_decode is True
    with pytest.raises(ValidationError, match=flag):
        FactoryArgs(**{flag: "invalid"})


def test_real_loop_launches_next_native_before_resolving_previous(owner):
    scheduler, batch, device = owner
    scheduler.running = True
    scheduler.inbox = queue.Queue()
    scheduler.admin_queue = queue.Queue()
    scheduler.process_admin_requests = Mock()
    scheduler.recv_requests = Mock(return_value=[])
    scheduler.take_deferred_request_payloads = Mock(return_value=[])
    scheduler.process_input_requests = Mock()
    scheduler._engine_paused = (
        False  # noqa: leading-underscore  # Native scheduler state.
    )
    scheduler.waiting_queue = []
    scheduler.running_batch = batch
    batch.check_decode_mem = Mock(return_value=True)
    choices = iter([batch, batch, None])
    scheduler.get_next_batch_to_run = lambda: next(choices)
    scheduler.self_check_during_idle = Mock()
    scheduler.sleep_during_idle = lambda: setattr(scheduler, "running", False)
    scheduler.event_loop_spec_v2()
    forwards = [i for i, event in enumerate(device.log) if event[0] == "native_forward"]
    host_waits = [i for i, event in enumerate(device.log) if event[0] == "wait"]
    assert len(forwards) == 2 and len(scheduler.stored_results) == 2
    assert forwards[1] < host_waits[0]
    assert scheduler.pending_spec_v2 is None


def test_stop_retains_scheduler_resources_before_callback_or_pending_exists(owner):
    scheduler, batch, device = owner
    scheduler.request_admission_lock = threading.RLock()
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.wait_for_prefill_publication_on_stop = True
    scheduler.shutdown_resources = Mock()
    scheduler.running = True
    assert scheduler.stop() is True
    assert scheduler.running is False
    scheduler.shutdown_resources.assert_not_called()


def test_listener_abort_is_queued_while_native_result_is_owned(owner):
    scheduler, batch, device = owner
    scheduler.pending_spec_v2 = scheduler.launch_spec_v2(batch)
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.tp_size = 1
    scheduler.is_entry_rank = True
    scheduler.inbox = queue.Queue()
    caller = threading.Thread(target=scheduler.abort, args=("a",))
    caller.start()
    caller.join(timeout=2)
    assert not caller.is_alive()
    message = scheduler.inbox.get_nowait()
    assert message.type == "abort" and message.request_id == "a"
    assert not batch.reqs[0].finished()
    assert not any(event[0] == "wait" for event in device.log)


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_real_loop_drains_before_synchronous_phase(owner, mode):
    scheduler, batch, device = owner
    scheduler.running = True
    scheduler.inbox = queue.Queue()
    scheduler.admin_queue = queue.Queue()
    scheduler.process_admin_requests = Mock()
    scheduler.recv_requests = Mock(return_value=[])
    scheduler.take_deferred_request_payloads = Mock(return_value=[])
    scheduler.process_input_requests = Mock()
    scheduler._engine_paused = (
        False  # noqa: leading-underscore  # Native scheduler state.
    )
    scheduler.waiting_queue = []
    scheduler.running_batch = batch
    batch.check_decode_mem = Mock(return_value=True)
    selection_count = 0

    def select():
        nonlocal selection_count
        selection_count += 1
        if selection_count == 2:
            batch.forward_mode = mode
        elif selection_count == 3:
            return None
        return batch

    scheduler.get_next_batch_to_run = select
    scheduler.self_check_during_idle = Mock()
    scheduler.sleep_during_idle = lambda: setattr(scheduler, "running", False)
    scheduler.event_loop_spec_v2()
    forward_positions = [
        i for i, event in enumerate(device.log) if event[0] == "native_forward"
    ]
    waits = [i for i, event in enumerate(device.log) if event[0] == "wait"]
    assert forward_positions[0] < waits[0] < forward_positions[1]
    assert device.log[forward_positions[1]] == ("native_forward", mode)
    assert scheduler.pending_spec_v2 is None


def test_sync_prefill_submission_failure_drains_before_normal_error_cleanup(owner):
    scheduler, batch, device = owner
    batch.forward_mode = ForwardMode.EXTEND
    scheduler.draft_worker.forward_batch_generation = Mock(
        side_effect=RuntimeError("partial prefill")
    )
    scheduler.close_prefill_publication = Mock()
    scheduler.handle_batch_failure.side_effect = lambda batch, error: device.log.append(
        ("cleanup",)
    )
    scheduler.run_batch(batch)
    cleanup = device.log.index(("cleanup",))
    assert all(
        device.log.index(("drain", name)) < cleanup
        for name in ("forward", "copy", "schedule")
    )


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.DECODE])
def test_dflash_failure_fence_does_not_depend_on_publication(
    owner, monkeypatch, overlap, early, mode
):
    scheduler, batch, device = owner
    scheduler.enable_spec_v2_overlap = overlap
    scheduler.wait_for_prefill_publication_on_stop = early or overlap
    scheduler.pending_prefill_publication = None
    scheduler.device = "cuda:0"
    batch.forward_mode = mode
    for request in batch.reqs:
        request.sampling_params.max_new_tokens = 1

    def submit_then_fail(live_batch, **kwargs):
        assert live_batch is batch
        device.log.append(("target_submitted",))
        raise RuntimeError("draft append failed after target submission")

    scheduler.draft_worker.forward_batch_generation = submit_then_fail
    synchronize = Mock(
        side_effect=lambda target: device.log.append(("device_drain", target))
    )
    monkeypatch.setattr(device, "synchronize", synchronize, raising=False)
    scheduler.handle_batch_failure.side_effect = lambda batch, error: device.log.append(
        ("cleanup",)
    )
    scheduler.run_batch(batch)
    cleanup = device.log.index(("cleanup",))
    assert device.log.index(("target_submitted",)) < cleanup
    if overlap:
        assert all(
            device.log.index(("drain", name)) < cleanup
            for name in ("forward", "copy", "schedule")
        )
        synchronize.assert_not_called()
    else:
        synchronize.assert_called_once_with("cuda:0")
        assert device.log.index(("device_drain", "cuda:0")) < cleanup


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("failed_fence", ["device", "forward", "copy", "schedule"])
def test_dflash_unknown_writer_retires_without_normal_cleanup(
    owner, monkeypatch, early, failed_fence
):
    scheduler, batch, device = owner
    scheduler.enable_spec_v2_overlap = failed_fence != "device"
    scheduler.wait_for_prefill_publication_on_stop = early
    scheduler.pending_prefill_publication = None
    scheduler.device = "cuda:0"
    scheduler.draft_worker.forward_batch_generation = Mock(
        side_effect=RuntimeError("partially submitted native forward")
    )
    failed_synchronize = Mock(side_effect=RuntimeError("writer completion unknown"))
    if failed_fence == "device":
        monkeypatch.setattr(device, "synchronize", failed_synchronize, raising=False)
    else:
        monkeypatch.setattr(
            getattr(scheduler, failed_fence + "_stream"),
            "synchronize",
            failed_synchronize,
        )
    scheduler.terminate_prefill_worker = Mock(side_effect=SystemExit("retire worker"))
    with pytest.raises(SystemExit, match="retire worker"):
        scheduler.run_batch(batch)
    failed_synchronize.assert_called_once()
    scheduler.terminate_prefill_worker.assert_called_once()
    scheduler.handle_batch_failure.assert_not_called()


def test_ordinary_ar_error_keeps_existing_cleanup_without_dflash_fence(
    owner, monkeypatch
):
    scheduler, batch, device = owner
    scheduler.enable_spec_v2_overlap = False
    scheduler.draft_worker = None
    scheduler.pending_prefill_publication = None
    scheduler._run_batch = Mock(
        side_effect=RuntimeError("ordinary AR error")
    )  # noqa: leading-underscore  # Native batch entry under test.
    synchronize = Mock(
        side_effect=AssertionError("ordinary AR must not enter the DFlash fence")
    )
    monkeypatch.setattr(device, "synchronize", synchronize, raising=False)
    scheduler.run_batch(batch)
    synchronize.assert_not_called()
    scheduler.handle_batch_failure.assert_called_once()


@pytest.mark.parametrize("overlap", [False, True])
def test_successful_dflash_batch_does_not_add_failure_fence(
    owner, monkeypatch, overlap
):
    scheduler, batch, device = owner
    scheduler.enable_spec_v2_overlap = overlap
    expected = Mock()
    scheduler._run_batch = Mock(
        return_value=expected
    )  # noqa: leading-underscore  # Native batch entry under test.
    synchronize = Mock(side_effect=AssertionError("no exceptional fence on success"))
    monkeypatch.setattr(device, "synchronize", synchronize, raising=False)
    assert scheduler.run_batch(batch) is expected
    synchronize.assert_not_called()
    assert not any(event[0] == "drain" for event in device.log)


def test_start_finally_drains_last_pending_before_shutdown(owner):
    scheduler, batch, device = owner
    scheduler.pending_spec_v2 = scheduler.launch_spec_v2(batch)
    scheduler.request_admission_lock = threading.RLock()
    scheduler.event_loop_spec_v2 = Mock()
    scheduler.close_prefill_publication = Mock()
    scheduler.emit_remaining_model_path_ends = Mock()
    scheduler.shutdown_request_build_executor = Mock()
    scheduler.discard_pending_request_admissions = Mock()
    scheduler.shutdown_resources = lambda: device.log.append(("shutdown",))
    scheduler.start()
    assert scheduler.pending_spec_v2 is None
    assert scheduler.stored_results
    assert device.log.index(("wait", "event-1")) < device.log.index(("shutdown",))


@pytest.mark.parametrize(
    "maximum,committed,drain",
    [
        (1, 0, True),
        (8, 0, True),
        (9, 0, False),
        (9, 1, True),
        (20, 11, False),
        (20, 12, True),
    ],
)
def test_cap_boundary_blocks_an_unbudgeted_extra_verify(
    owner, maximum, committed, drain
):
    scheduler, batch, device = owner
    pending = scheduler.launch_spec_v2(batch)
    scheduler.pending_spec_v2 = pending
    batch.reqs[0].sampling_params.max_new_tokens = maximum
    batch.reqs[0].output_ids.extend([10] * committed)
    assert scheduler.spec_v2_pending_may_reach_cap() is drain


@pytest.mark.parametrize(
    "tensor_parallel,pipeline_parallel,enabled",
    [(1, 1, True), (2, 1, True), (1, 2, True), (2, 1, False)],
)
def test_real_constructor_reads_published_parallel_context_before_worker_access(
    tensor_parallel, pipeline_parallel, enabled
):
    class ReachedWorker(Exception):
        pass

    class WorkerBoundary:
        @property
        def gpu_id(self):
            raise ReachedWorker

    with get_context().override_server_args(
        tp_size=tensor_parallel, pp_size=pipeline_parallel
    ) as arguments:
        expected = (
            ValueError
            if enabled and (tensor_parallel != 1 or pipeline_parallel != 1)
            else ReachedWorker
        )
        with pytest.raises(expected):
            OmniScheduler(
                WorkerBoundary(),
                None,
                None,
                None,
                arguments,
                None,
                draft_worker=Mock(),
                enable_spec_v2_overlap=enabled,
            )


@pytest.mark.parametrize("transition", ["idle", "first_prefill", "finished_pending"])
def test_real_empty_running_batch_has_no_decode_allocator(owner, transition):
    scheduler, batch, device = owner
    scheduler.running = True
    scheduler.inbox = queue.Queue()
    scheduler.admin_queue = queue.Queue()
    scheduler.process_admin_requests = Mock()
    scheduler.recv_requests = Mock(return_value=[])
    scheduler.take_deferred_request_payloads = Mock(return_value=[])
    scheduler.process_input_requests = Mock()
    scheduler._engine_paused = (
        False  # noqa: leading-underscore  # Native scheduler state.
    )
    scheduler.waiting_queue = []
    scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    assert scheduler.running_batch.token_to_kv_pool_allocator is None
    if transition == "finished_pending":
        scheduler.pending_spec_v2 = scheduler.launch_spec_v2(batch)
        for request in batch.reqs:
            request.finished_reason = FINISH_LENGTH(20)
            request.kv.req_pool_idx = None
        choices = iter([None])
    elif transition == "first_prefill":
        batch.forward_mode = ForwardMode.EXTEND
        batch.spec_info = None
        choices = iter([batch, None])
    else:
        choices = iter([None])
    scheduler.get_next_batch_to_run = lambda: next(choices)
    scheduler.self_check_during_idle = Mock()
    scheduler.sleep_during_idle = lambda: setattr(scheduler, "running", False)
    scheduler.event_loop_spec_v2()
    assert scheduler.pending_spec_v2 is None
    scheduler.self_check_during_idle.assert_called_once()
    if transition == "first_prefill":
        assert ("publish_lengths", [7, 2, 12]) in device.log
        assert ("stash", [7, 2, 12]) in device.log
        assert batch.spec_info.future_indices is batch.req_pool_indices
    elif transition == "finished_pending":
        assert len(scheduler.stored_results) == 1
        assert any(event[0] == "wait" for event in device.log)
    else:
        assert scheduler.stored_results == []


def test_nonempty_pending_memory_pressure_drains_before_batch_selection(owner):
    scheduler, batch, device = owner
    scheduler.pending_spec_v2 = scheduler.launch_spec_v2(batch)
    scheduler.running = True
    scheduler.inbox = queue.Queue()
    scheduler.admin_queue = queue.Queue()
    scheduler.process_admin_requests = Mock()
    scheduler.recv_requests = Mock(return_value=[])
    scheduler.take_deferred_request_payloads = Mock(return_value=[])
    scheduler.process_input_requests = Mock()
    scheduler._engine_paused = (
        False  # noqa: leading-underscore  # Native scheduler state.
    )
    scheduler.waiting_queue = []
    scheduler.running_batch = batch
    batch.check_decode_mem = Mock(return_value=False)

    def select():
        assert scheduler.pending_spec_v2 is None
        assert len(scheduler.stored_results) == 1
        return None

    scheduler.get_next_batch_to_run = select
    scheduler.self_check_during_idle = Mock()
    scheduler.sleep_during_idle = lambda: setattr(scheduler, "running", False)
    scheduler.event_loop_spec_v2()
    batch.check_decode_mem.assert_called_once()
