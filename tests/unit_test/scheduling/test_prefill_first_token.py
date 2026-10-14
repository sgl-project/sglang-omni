# SPDX-License-Identifier: Apache-2.0
"""First-token publication preserves normal speculative commit ownership."""
import asyncio
import gc
import queue
import threading
import weakref
from array import array
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from pydantic import ValidationError
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import FactoryArgs
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_stream_output_builder,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling import omni_scheduler as scheduler_module
from sglang_omni.scheduling.message import OutgoingMessage, PendingStreamOutput
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from tests.unit_test.model_runner.test_speculative_target_worker import (
    make_speculative_target_probe,
)
from tests.unit_test.pipeline.helpers import make_stage


@pytest.fixture(params=[1])
def publication(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    request_count = request.param
    requests = []
    request_data_by_id = {}
    for request_index in range(request_count):
        request_id = "request" if request_index == 0 else f"request-{request_index}"
        sampling = SamplingParams(temperature=0, max_new_tokens=8)
        sampling.normalize(None)
        model_request = Req(
            request_id,
            "prompt",
            array("q", [1, 2]),
            sampling,
            stream=True,
            eos_token_ids={99},
        )
        model_request.kv.req_pool_idx = 3 + request_index
        model_request._omni_terminal_claimed = (
            False  # noqa: leading-underscore  # Shared ownership field.
        )
        request_data = SGLangARRequestData(
            req=model_request,
            stage_payload=StagePayload(
                request_id=request_id,
                request=OmniRequest(inputs="prompt", params={"stream": True}),
                data={},
            ),
        )
        model_request.omni_data = request_data
        request_data.output_ids = model_request.output_ids
        requests.append(model_request)
        request_data_by_id[request_id] = request_data
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.reqs = requests
    batch.decoding_reqs = None
    batch.forward_mode = ForwardMode.EXTEND
    batch.is_prefill_only = False
    batch.return_logprob = False
    batch.return_hidden_states = False
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.request_admission_lock = threading.RLock()
    scheduler.running = True
    scheduler.wait_for_prefill_publication_on_stop = True
    scheduler.pending_prefill_publication = None
    scheduler.session_bridge = None
    scheduler.aborted_request_ids = set()
    scheduler.stream_output_builder = make_thinker_stream_output_builder(
        speech_enabled=False
    )
    scheduler.outbox = queue.Queue()
    scheduler.first_emit_done = set()
    scheduler.find_request_data = request_data_by_id.get
    events = []
    copied = torch.arange(7, 7 + request_count)

    class HostToken:
        def copy_(self, source, non_blocking=False):
            assert non_blocking
            events.append("copy")

        def __getitem__(self, index):
            return copied[index]

    event = Mock()
    event.record.side_effect = lambda stream: events.append("record")
    event.synchronize.side_effect = lambda: events.append("ready")
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: HostToken())
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: events.append("drain")),
    )
    token_ids = SimpleNamespace(
        is_cuda=True,
        numel=lambda: request_count,
        shape=(request_count,),
        dtype=torch.int64,
        device="cuda",
    )
    result = GenerationBatchResult(next_token_ids=token_ids)
    return scheduler, batch, result, copied, event, events


def publish_and_resolve(scheduler, batch, result):
    scheduler.publish_prefill_first_token(batch, result)
    if scheduler.pending_prefill_publication is not None:
        pending_output = scheduler.outbox.get_nowait()
        assert isinstance(pending_output, PendingStreamOutput)
        pending_output.resolve()
        scheduler.wait_prefill_publication()


def test_publish_then_commit_emits_once(publication):
    scheduler, batch, result, _, _, events = publication
    request = batch.reqs[0]
    original_ids = request.output_ids
    publish_and_resolve(scheduler, batch, result)
    assert events == ["copy", "record", "ready"]
    assert scheduler.outbox.qsize() == 1
    assert request.output_ids is original_ids and not request.output_ids
    assert request.kv.req_pool_idx == 3
    assert request.omni_data.speculative_first_token_id == 7
    request.output_ids.append(7)
    scheduler.emit_speculative_tokens(request)
    assert scheduler.outbox.qsize() == 1
    message = scheduler.outbox.get_nowait()
    assert message.target == "decode" and message.data.tolist() == [7]
    request.output_ids.append(8)
    scheduler.emit_speculative_tokens(request)
    assert scheduler.outbox.get_nowait().data.tolist() == [8]


@pytest.mark.parametrize(
    "state",
    [
        "decode",
        "middle",
        "cap1",
        "nonstream",
        "stop",
        "session",
        "hidden",
        "logprob",
        "abort",
    ],
)
def test_ineligible_requests_do_no_copy(publication, state):
    scheduler, batch, result, _, _, events = publication
    req = batch.reqs[0]
    if state == "decode":
        batch.forward_mode = ForwardMode.DECODE
    elif state == "middle":
        req.inflight_middle_chunks = 1
    elif state == "cap1":
        req.sampling_params.max_new_tokens = 1
    elif state == "nonstream":
        req.omni_data.stage_payload.request.params["stream"] = False
    elif state == "stop":
        req.sampling_params.stop_token_ids = {17}
    elif state == "session":
        scheduler.session_bridge = object()
    elif state == "hidden":
        batch.return_hidden_states = True
    elif state == "logprob":
        batch.return_logprob = True
    elif state == "abort":
        req.to_finish = FINISH_ABORT()
    else:
        raise AssertionError(state)
    publish_and_resolve(scheduler, batch, result)
    assert not events and scheduler.outbox.empty()


@pytest.mark.parametrize("mutation", ["abort", "slot", "cache", "data", "rid"])
def test_identity_is_rechecked_after_copy_wait(publication, mutation):
    scheduler, batch, result, _, event, _ = publication
    req = batch.reqs[0]

    def changed():
        if mutation == "abort":
            req.to_finish = FINISH_ABORT()
        elif mutation == "slot":
            req.kv.req_pool_idx += 1
        elif mutation == "cache":
            req.cache_request_handle = object()
        elif mutation == "data":
            req.omni_data = None
        elif mutation == "rid":
            req.rid = "rebound"
        else:
            raise AssertionError(mutation)

    event.synchronize.side_effect = changed
    publish_and_resolve(scheduler, batch, result)
    assert scheduler.outbox.empty()


def test_eos_falls_back_to_normal_commit(publication):
    scheduler, batch, result, copied, _, _ = publication
    copied[0] = 99
    publish_and_resolve(scheduler, batch, result)
    assert scheduler.outbox.empty()
    assert batch.reqs[0].omni_data.speculative_emitted_tokens == 0


def test_first_committed_id_must_match(publication):
    scheduler, batch, result, _, _, _ = publication
    publish_and_resolve(scheduler, batch, result)
    batch.reqs[0].output_ids.append(8)
    with pytest.raises(AssertionError):
        scheduler.emit_speculative_tokens(batch.reqs[0])


@pytest.mark.parametrize("publication", [2, 3], indirect=True)
def test_batch_publication_preserves_rows_and_commits_once(publication):
    scheduler, batch, result, copied, _, events = publication
    original_slots = [request.kv.req_pool_idx for request in batch.reqs]
    publish_and_resolve(scheduler, batch, result)
    assert events == ["copy", "record", "ready"]
    for request, token_id, request_slot in zip(batch.reqs, copied, original_slots):
        message = scheduler.outbox.get_nowait()
        assert message.request_id == request.rid
        assert message.target == "decode" and message.data.tolist() == [int(token_id)]
        assert not request.output_ids
        assert request.kv.req_pool_idx == request_slot
    for request, token_id in zip(batch.reqs, copied):
        request.output_ids.append(int(token_id))
        scheduler.emit_speculative_tokens(request)
        assert scheduler.outbox.empty()
        request.output_ids.append(int(token_id) + 10)
        scheduler.emit_speculative_tokens(request)
        committed_message = scheduler.outbox.get_nowait()
        assert committed_message.request_id == request.rid
        assert committed_message.data.tolist() == [int(token_id) + 10]


@pytest.mark.parametrize("publication", [3], indirect=True)
@pytest.mark.parametrize(
    "state", ["cancel", "stop", "nonstream", "middle", "cap1", "eos"]
)
def test_one_ineligible_row_does_not_block_other_rows(publication, state):
    scheduler, batch, result, copied, _, events = publication
    request = batch.reqs[1]
    if state == "cancel":
        request.to_finish = FINISH_ABORT()
    elif state == "stop":
        request.sampling_params.stop_token_ids = {17}
    elif state == "nonstream":
        request.omni_data.stage_payload.request.params["stream"] = False
    elif state == "middle":
        request.inflight_middle_chunks = 1
    elif state == "cap1":
        request.sampling_params.max_new_tokens = 1
    elif state == "eos":
        copied[1] = 99
    else:
        raise AssertionError(state)
    publish_and_resolve(scheduler, batch, result)
    assert events == ["copy", "record", "ready"]
    assert scheduler.outbox.qsize() == 2
    assert scheduler.outbox.get_nowait().request_id == batch.reqs[0].rid
    assert scheduler.outbox.get_nowait().request_id == batch.reqs[2].rid
    assert request.omni_data.speculative_emitted_tokens == 0
    assert all(not request.output_ids for request in batch.reqs)


@pytest.mark.parametrize("publication", [3], indirect=True)
@pytest.mark.parametrize("mutation", ["cancel", "slot", "cache", "data", "rid"])
def test_batch_revalidates_each_row_after_copy_wait(publication, mutation):
    scheduler, batch, result, _, event, _ = publication
    original_requests = tuple(batch.reqs)
    request = original_requests[1]
    request_data = request.omni_data

    def changed():
        if mutation == "cancel":
            request.to_finish = FINISH_ABORT()
        elif mutation == "slot":
            request.kv.req_pool_idx += 1
        elif mutation == "cache":
            request.cache_request_handle = object()
        elif mutation == "data":
            request.omni_data = None
        elif mutation == "rid":
            request.rid = "rebound"
        else:
            raise AssertionError(mutation)

    event.synchronize.side_effect = changed
    publish_and_resolve(scheduler, batch, result)
    assert scheduler.outbox.qsize() == 2
    assert scheduler.outbox.get_nowait().request_id == original_requests[0].rid
    assert scheduler.outbox.get_nowait().request_id == original_requests[2].rid
    assert request_data.speculative_emitted_tokens == 0


@pytest.mark.parametrize("publication", [3], indirect=True)
def test_repeated_batch_publication_does_not_copy_or_emit_twice(publication):
    scheduler, batch, result, _, _, events = publication
    publish_and_resolve(scheduler, batch, result)
    publish_and_resolve(scheduler, batch, result)
    assert events == ["copy", "record", "ready"]
    assert scheduler.outbox.qsize() == 3


@pytest.mark.parametrize("publication", [3], indirect=True)
@pytest.mark.parametrize("has_committed_decode_token", [False, True])
def test_mixed_batch_publishes_only_new_prefill_rows(
    publication, has_committed_decode_token
):
    scheduler, batch, result, _, _, events = publication
    batch.forward_mode = ForwardMode.MIXED
    decoding_request = batch.reqs[1]
    batch.decoding_reqs = [decoding_request]
    if has_committed_decode_token:
        decoding_request.output_ids.append(17)
        decoding_request.omni_data.speculative_emitted_tokens = 1
    else:
        pass
    publish_and_resolve(scheduler, batch, result)
    assert events == ["copy", "record", "ready"]
    assert scheduler.outbox.qsize() == 2
    for request_index in [0, 2]:
        message = scheduler.outbox.get_nowait()
        assert message.request_id == batch.reqs[request_index].rid
        assert message.data.tolist() == [7 + request_index]
        assert not batch.reqs[request_index].output_ids
    assert decoding_request.omni_data.speculative_emitted_tokens == int(
        has_committed_decode_token
    )


@pytest.mark.parametrize("publication", [3], indirect=True)
def test_mixed_batch_without_row_roles_uses_normal_publication(publication):
    scheduler, batch, result, _, _, events = publication
    batch.forward_mode = ForwardMode.MIXED
    publish_and_resolve(scheduler, batch, result)
    assert not events and scheduler.outbox.empty()


@pytest.mark.parametrize("publication", [3], indirect=True)
def test_mixed_middle_chunk_is_not_a_first_output(publication):
    scheduler, batch, result, _, _, events = publication
    batch.forward_mode = ForwardMode.MIXED
    batch.decoding_reqs = [batch.reqs[1]]
    batch.reqs[0].inflight_middle_chunks = 1
    publish_and_resolve(scheduler, batch, result)
    assert events == ["copy", "record", "ready"]
    assert scheduler.outbox.qsize() == 1
    message = scheduler.outbox.get_nowait()
    assert message.request_id == batch.reqs[2].rid
    assert message.data.tolist() == [9]


@pytest.mark.parametrize("publication", [3], indirect=True)
def test_mixed_row_roles_are_frozen_before_native_changes_batch(publication):
    scheduler, batch, result, _, event, _ = publication
    batch.forward_mode = ForwardMode.MIXED
    batch.decoding_reqs = []
    event.synchronize.side_effect = lambda: setattr(
        batch, "decoding_reqs", [batch.reqs[1]]
    )
    publish_and_resolve(scheduler, batch, result)
    assert scheduler.outbox.qsize() == 3
    for request in batch.reqs:
        assert scheduler.outbox.get_nowait().request_id == request.rid


@pytest.mark.parametrize("publication", [1, 3], indirect=True)
def test_real_abort_during_copy_wait_retains_current_batch_storage(publication):
    scheduler, batch, result, _, event, events = publication
    request = batch.reqs[0]
    request_data = request.omni_data
    original_requests = tuple(batch.reqs)
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.tp_size = 1
    scheduler.running_batch = None
    scheduler.cur_batch = batch
    scheduler.last_batch = None
    scheduler.async_pending = None
    scheduler.waiting_queue = []
    scheduler.aborted_request_id_order = deque()
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.backlogged_request_build_payloads = deque()
    scheduler.pending_stream_ingress = {}
    scheduler.deferred_request_payloads = {}
    scheduler.dirty_deferred_request_ids = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.emit_model_path_end_once = Mock()
    scheduler.drain_inbox_for_request = Mock()
    scheduler.run_abort_callback = Mock()
    scheduler.release_request_kv_cache = Mock()

    def cancel_while_waiting():
        # A listener thread must enter abort while the publishing thread waits.
        errors = []

        def cancel():
            try:
                scheduler.abort(request.rid)
            except Exception as error:
                errors.append(error)

        listener = threading.Thread(target=cancel)
        listener.start()
        listener.join(timeout=2)
        assert not listener.is_alive(), "Copy wait must not hold admission lock"
        assert not errors
        assert isinstance(request.to_finish, FINISH_ABORT)
        assert tuple(scheduler.cur_batch.reqs) == original_requests
        assert request.omni_data is request_data
        assert request.kv.req_pool_idx == 3
        scheduler.release_request_kv_cache.assert_not_called()
        scheduler.run_abort_callback.assert_not_called()
        events.append("cancelled-before-copy-ready")

    event.synchronize.side_effect = cancel_while_waiting
    publish_and_resolve(scheduler, batch, result)

    assert events == ["copy", "record", "cancelled-before-copy-ready"]
    assert scheduler.outbox.qsize() == len(original_requests) - 1
    assert request_data.speculative_emitted_tokens == 0
    assert request_data.speculative_first_token_id is None
    assert not request.output_ids
    scheduler.release_request_kv_cache.assert_not_called()
    scheduler.run_abort_callback.assert_not_called()


@pytest.mark.parametrize("publication", [3], indirect=True)
def test_partial_batch_publication_failure_drains_before_cleanup(
    publication, monkeypatch: pytest.MonkeyPatch
):
    scheduler, batch, result, _, _, _ = publication
    adapter, forward_batch, calls = make_speculative_target_probe()
    adapter.thinker_runner.prepare_and_forward = Mock(
        return_value=SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=torch.ones(3, 20)),
            next_token_ids=result.next_token_ids,
        )
    )
    adapter.prefill_token_publisher = scheduler.publish_prefill_first_token
    monkeypatch.setattr(ForwardBatch, "init_new", lambda *args, **kwargs: forward_batch)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: calls.append("drain")),
    )
    stream_builder = scheduler.stream_output_builder
    publication_error = ValueError("second row publication failed")

    def fail_second_row(request_id, request_data, output):
        if request_id == batch.reqs[1].rid:
            calls.append("publication_failed")
            raise publication_error
        else:
            return stream_builder(request_id, request_data, output)

    scheduler.stream_output_builder = fail_second_row
    scheduler.running = True

    def forward_then_resolve(*args):
        adapter.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        calls.append("draft_submitted")
        scheduler.outbox.get_nowait().resolve()
        scheduler.wait_prefill_publication()

    scheduler._run_batch = Mock(
        side_effect=forward_then_resolve
    )  # noqa: leading-underscore

    scheduler.handle_batch_failure = Mock(
        side_effect=lambda *args: calls.append("cleanup")
    )
    scheduler.run_batch(batch)
    assert calls[-3:] == ["publication_failed", "drain", "cleanup"]
    scheduler.handle_batch_failure.assert_called_once_with(batch, publication_error)
    assert scheduler.outbox.qsize() == 1
    assert scheduler.outbox.get_nowait().request_id == batch.reqs[0].rid
    assert [request.omni_data.speculative_emitted_tokens for request in batch.reqs] == [
        1,
        0,
        0,
    ]
    assert all(not request.output_ids for request in batch.reqs)


@pytest.mark.parametrize("publication", [3], indirect=True)
def test_callback_returns_before_copy_resolution(publication):
    scheduler, batch, result, _, event, events = publication
    scheduler.publish_prefill_first_token(batch, result)
    assert events == ["copy", "record"]
    event.synchronize.assert_not_called()
    assert all(
        request.omni_data.speculative_emitted_tokens == 0 for request in batch.reqs
    )
    envelope = scheduler.outbox.get_nowait()
    envelope.resolve()
    scheduler.wait_prefill_publication()
    assert events == ["copy", "record", "ready"]
    assert scheduler.outbox.qsize() == 3


@pytest.mark.parametrize("publication", [3], indirect=True)
def test_native_return_waits_for_ack_before_cache_publication(publication, monkeypatch):
    scheduler, batch, result, _, event, events = publication
    copy_entered = threading.Event()
    copy_release = threading.Event()

    def wait_copy():
        copy_entered.set()
        assert copy_release.wait(2)
        events.append("ready")

    event.synchronize.side_effect = wait_copy
    original_requests = tuple(batch.reqs)
    draft_submitted = threading.Event()
    worker_errors = []
    returned = threading.Event()
    native_result = SimpleNamespace(
        next_draft_input=None, new_seq_lens=None, has_sampled_token_ids=False
    )
    scheduler.emit_prefill_start_for_batch = Mock()
    scheduler.stamp_batch_launch = Mock()
    scheduler.emit_prefill_end_for_batch = Mock()
    scheduler.future_map = None
    scheduler.device_module = SimpleNamespace(Event=Mock(return_value=object()))
    scheduler._forward_isolation = (
        lambda *args, **kwargs: nullcontext()
    )  # noqa: leading-underscore
    monkeypatch.setattr(scheduler_module, "resolve_forward_inputs", lambda *args: None)

    def forward(batch):
        scheduler.publish_prefill_first_token(batch, result)
        assert events == ["copy", "record"]
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.reqs = []
        events.append("draft-submitted")
        draft_submitted.set()
        return native_result

    def update_cache(*args):
        assert all(
            request.omni_data.speculative_emitted_tokens == 1
            for request in original_requests
        )
        assert scheduler.pending_prefill_publication is None
        events.append("cache-publication")

    scheduler.draft_worker = SimpleNamespace(forward_batch_generation=forward)
    scheduler.update_cache_from_scheduler = Mock(side_effect=update_cache)

    def run():
        try:
            assert scheduler.run_batch(batch) is native_result
        except BaseException as error:
            worker_errors.append(error)
        finally:
            returned.set()

    worker = threading.Thread(target=run)
    worker.start()
    assert draft_submitted.wait(2)
    assert copy_entered.wait(2)
    scheduler.update_cache_from_scheduler.assert_not_called()
    assert not returned.is_set()
    envelope = scheduler.outbox.get_nowait()
    envelope.resolve()
    copy_release.set()
    worker.join(timeout=2)
    assert not worker.is_alive() and not worker_errors
    assert events == ["copy", "record", "draft-submitted", "ready", "cache-publication"]
    assert [scheduler.outbox.get_nowait().request_id for _ in range(3)] == [
        request.rid for request in original_requests
    ]


def test_stop_before_callback_retains_resources_and_rejects_new_pending(publication):
    scheduler, batch, result, _, _, events = publication
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.shutdown_resources = Mock()
    scheduler.discard_pending_request_admissions = Mock()
    assert scheduler.stop() is True
    scheduler.publish_prefill_first_token(batch, result)
    scheduler.shutdown_resources.assert_not_called()
    scheduler.discard_pending_request_admissions.assert_not_called()
    assert not events and scheduler.outbox.empty()
    assert scheduler.pending_prefill_publication is None


def test_stop_during_pending_copy_suppresses_early_output(publication):
    scheduler, batch, result, _, _, events = publication
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.shutdown_resources = Mock()
    scheduler.publish_prefill_first_token(batch, result)
    assert scheduler.stop() is True
    scheduler.outbox.get_nowait().resolve()
    with pytest.raises(RuntimeError, match="consumer stopped"):
        scheduler.wait_prefill_publication()
    scheduler.close_prefill_publication(RuntimeError("consumer stopped"))
    assert events == ["copy", "record", "drain"]
    assert scheduler.outbox.empty()
    scheduler.shutdown_resources.assert_not_called()


def test_worker_failure_before_resolution_drains_and_closes_envelope(publication):
    scheduler, batch, result, _, _, events = publication
    scheduler.publish_prefill_first_token(batch, result)
    envelope = scheduler.outbox.get_nowait()
    scheduler.close_prefill_publication(ValueError("draft submission failed"))
    envelope.resolve()
    assert events == ["copy", "record", "drain"]
    assert scheduler.pending_prefill_publication is None
    assert scheduler.outbox.empty()


def test_failed_gpu_drain_never_enters_recoverable_cleanup(publication, monkeypatch):
    scheduler, batch, result, _, _, events = publication
    producer = SimpleNamespace(
        synchronize=Mock(side_effect=RuntimeError("device lost"))
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: producer)
    monkeypatch.setattr(scheduler_module.os, "write", Mock())
    monkeypatch.setattr(scheduler_module.os, "_exit", Mock(side_effect=SystemExit(1)))
    scheduler.publish_prefill_first_token(batch, result)
    with pytest.raises(SystemExit, match="1"):
        scheduler.close_prefill_publication(ValueError("draft submission failed"))
    assert scheduler.pending_prefill_publication is not None
    assert not scheduler.pending_prefill_publication.completion.done()
    assert events == ["copy", "record"]


def test_stop_before_scheduler_start_cannot_restart_publication(publication):
    scheduler, _, _, _, _, _ = publication
    scheduler.shutdown_resources = Mock()
    scheduler.discard_pending_request_admissions = Mock()
    scheduler.emit_remaining_model_path_ends = Mock()
    scheduler.shutdown_request_build_executor = Mock()
    scheduler.event_loop_normal = Mock()
    scheduler.enable_async_decode = False
    scheduler.enable_overlap = False
    assert scheduler.stop() is False
    scheduler.start()
    scheduler.event_loop_normal.assert_not_called()
    assert not scheduler.running and scheduler.scheduler_thread_id is None


def test_consumer_failure_before_pending_cannot_deadlock_stop(publication):
    async def run():
        scheduler, batch, result, _, _, events = publication
        entered = threading.Event()
        native_return_release = threading.Event()
        scheduler.shutdown_resources = Mock(
            side_effect=lambda: events.append("resource-close")
        )
        scheduler.discard_pending_request_admissions = Mock()
        scheduler.emit_remaining_model_path_ends = Mock()
        scheduler.shutdown_request_build_executor = Mock()
        scheduler.handle_batch_failure = Mock(
            side_effect=lambda *args: events.append("cleanup")
        )
        scheduler.enable_async_decode = False
        scheduler.enable_overlap = False

        def forward(*args):
            scheduler.publish_prefill_first_token(batch, result)
            entered.set()
            assert native_return_release.wait(2)
            scheduler.wait_prefill_publication()

        scheduler._run_batch = forward  # noqa: leading-underscore
        scheduler.event_loop_normal = lambda: scheduler.run_batch(batch)
        scheduler.outbox.put(
            OutgoingMessage(request_id="request", type="stream", target="decode")
        )
        stage = make_stage(scheduler=scheduler)
        stage.active_requests.add("request")
        stage.send_stream_to_target = AsyncMock(
            side_effect=RuntimeError("ordinary send failed")
        )
        await stage.start()
        assert await asyncio.to_thread(entered.wait, 1)
        with pytest.raises(RuntimeError, match="ordinary send failed"):
            await stage.drain_outbox_external()
        stopping = asyncio.create_task(stage.stop())
        await asyncio.sleep(0)
        assert not scheduler.running
        native_return_release.set()
        await asyncio.wait_for(stopping, 2)
        assert stage.scheduler_thread is None
        assert scheduler.pending_prefill_publication is None
        assert events == ["copy", "record", "drain", "cleanup", "resource-close"]
        scheduler.handle_batch_failure.assert_called_once()
        assert batch.reqs[0].omni_data.speculative_emitted_tokens == 0

    asyncio.run(run())


@pytest.mark.parametrize("publication", [3], indirect=True)
def test_producer_claim_does_not_wait_for_prior_network_send(publication):
    async def run():
        scheduler, batch, result, _, event, _ = publication
        stage = make_stage(
            scheduler=scheduler, endpoints={"decode": "inproc://producer-claim"}
        )
        stage.running = True
        stage.active_requests.update(
            {"earlier", *(request.rid for request in batch.reqs)}
        )
        send_entered = asyncio.Event()
        send_release = asyncio.Event()
        producer_completed = threading.Event()
        failures = []
        sent_request_ids = []

        async def send_to_stage(target, endpoint, message):
            sent_request_ids.append(message.request_id)
            if message.request_id == "earlier":
                send_entered.set()
                await send_release.wait()
            else:
                pass

        stage.control_plane.send_to_stage = send_to_stage
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="earlier",
                type="stream",
                target="decode",
                data=torch.tensor([1]),
            )
        )
        drain = asyncio.create_task(stage.drain_outbox_external())
        await asyncio.wait_for(send_entered.wait(), 2)

        def producer():
            try:
                scheduler.publish_prefill_first_token(batch, result)
                scheduler.wait_prefill_publication()
                producer_completed.set()
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=producer)
        thread.start()
        try:
            assert await asyncio.to_thread(producer_completed.wait, 2)
            assert not send_release.is_set()
            assert not failures and scheduler.pending_prefill_publication is None
            assert event.synchronize.call_count == 1
            assert all(
                request.omni_data.speculative_emitted_tokens == 1
                for request in batch.reqs
            )
            assert all(not request.output_ids for request in batch.reqs)
            assert sent_request_ids == ["earlier"]
        finally:
            send_release.set()
            await asyncio.to_thread(thread.join, 2)
            stage.running = False
            await asyncio.wait_for(drain, 2)
        assert sent_request_ids == ["earlier", *(request.rid for request in batch.reqs)]

    asyncio.run(run())


@pytest.mark.parametrize("first_owner", ["consumer", "producer"])
def test_competing_resolvers_claim_once(publication, first_owner):
    scheduler, batch, result, _, event, _ = publication
    copy_entered = threading.Event()
    copy_release = threading.Event()
    consumer_completed = threading.Event()
    producer_completed = threading.Event()
    failures = []

    def wait_copy():
        copy_entered.set()
        assert copy_release.wait(2)

    event.synchronize.side_effect = wait_copy
    scheduler.publish_prefill_first_token(batch, result)
    envelope = scheduler.outbox.get_nowait()

    def consumer():
        try:
            envelope.resolve()
        except BaseException as error:
            failures.append(error)
        finally:
            consumer_completed.set()

    def producer():
        try:
            scheduler.wait_prefill_publication()
        except BaseException as error:
            failures.append(error)
        finally:
            producer_completed.set()

    consumer_thread = threading.Thread(target=consumer)
    producer_thread = threading.Thread(target=producer)
    first, second = (
        (consumer_thread, producer_thread)
        if first_owner == "consumer"
        else (producer_thread, consumer_thread)
    )
    first.start()
    assert copy_entered.wait(2)
    second.start()
    try:
        assert not producer_completed.is_set()
        if first_owner == "producer":
            assert consumer_completed.wait(1)
        else:
            pass
        assert not batch.reqs[0].output_ids
    finally:
        copy_release.set()
        first.join(2)
        second.join(2)
    assert not first.is_alive() and not second.is_alive() and not failures
    assert event.synchronize.call_count == 1
    assert scheduler.outbox.qsize() == 1
    assert batch.reqs[0].omni_data.speculative_emitted_tokens == 1
    assert scheduler.pending_prefill_publication is None
    envelope.resolve()
    assert event.synchronize.call_count == 1 and scheduler.outbox.qsize() == 1


def test_queued_weak_envelope_releases_retired_payloads(publication):
    scheduler, batch, result, _, event, _ = publication
    scheduler.publish_prefill_first_token(batch, result)
    pending = scheduler.pending_prefill_publication
    pending.source_token_ids = torch.tensor([7])
    pending.host_token_ids = torch.tensor([7])
    batch.reqs[0].omni_data.prefill_input_embeds = torch.ones((2, 4))
    references = [
        weakref.ref(pending),
        weakref.ref(pending.source_token_ids),
        weakref.ref(pending.host_token_ids),
        weakref.ref(batch.reqs[0]),
        weakref.ref(batch.reqs[0].omni_data),
        weakref.ref(batch.reqs[0].omni_data.prefill_input_embeds),
    ]
    pending = None
    gc.collect()
    assert all(reference() is not None for reference in references)
    scheduler.wait_prefill_publication()
    assert event.synchronize.call_count == 1
    assert scheduler.outbox.qsize() == 2
    scheduler.find_request_data = lambda request_id: None
    batch.reqs.clear()
    gc.collect()
    assert all(reference() is None for reference in references)
    envelope = scheduler.outbox.get_nowait()
    replacement = SimpleNamespace(completion=Mock())
    scheduler.pending_prefill_publication = replacement
    envelope.resolve()
    assert scheduler.pending_prefill_publication is replacement
    replacement.completion.assert_not_called()
    scheduler.pending_prefill_publication = None
    assert event.synchronize.call_count == 1


def test_error_drain_retains_strong_payload_before_retirement(publication):
    scheduler, batch, result, _, _, events = publication
    scheduler.publish_prefill_first_token(batch, result)
    pending = scheduler.pending_prefill_publication
    pending_reference = weakref.ref(pending)
    pending.source_token_ids = torch.tensor([7])
    source_reference = weakref.ref(pending.source_token_ids)
    drain_entered = threading.Event()
    drain_release = threading.Event()
    failures = []

    def drain_writer():
        drain_entered.set()
        assert pending_reference() is not None and source_reference() is not None
        assert drain_release.wait(2)
        events.append("drain")

    pending.producer_stream.synchronize = drain_writer
    pending = None

    def close_publication():
        try:
            scheduler.close_prefill_publication(RuntimeError("native submit failed"))
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=close_publication)
    thread.start()
    assert drain_entered.wait(2)
    gc.collect()
    assert pending_reference() is not None and source_reference() is not None
    drain_release.set()
    thread.join(2)
    assert not thread.is_alive() and not failures
    assert scheduler.pending_prefill_publication is None
    gc.collect()
    assert pending_reference() is None and source_reference() is None
    scheduler.outbox.get_nowait().resolve()
    assert not batch.reqs[0].output_ids
    assert batch.reqs[0].omni_data.speculative_emitted_tokens == 0


@pytest.mark.parametrize("first_owner", ["consumer", "producer"])
def test_stop_after_claim_suppresses_publication_and_retains_resources(
    publication, first_owner
):
    scheduler, batch, result, _, event, _ = publication
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.shutdown_resources = Mock()
    copy_entered = threading.Event()
    copy_release = threading.Event()
    errors = []

    def wait_copy():
        copy_entered.set()
        assert copy_release.wait(2)

    event.synchronize.side_effect = wait_copy
    scheduler.publish_prefill_first_token(batch, result)
    envelope = scheduler.outbox.get_nowait()
    pending_reference = weakref.ref(scheduler.pending_prefill_publication)

    def run_owner():
        try:
            if first_owner == "consumer":
                envelope.resolve()
            else:
                scheduler.wait_prefill_publication()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_owner)
    thread.start()
    assert copy_entered.wait(2)
    try:
        assert scheduler.stop() is True
        assert pending_reference() is not None and pending_reference().closed
        scheduler.shutdown_resources.assert_not_called()
        assert not batch.reqs[0].output_ids
    finally:
        copy_release.set()
        thread.join(2)
    assert not thread.is_alive() and not errors
    scheduler.wait_prefill_publication()
    envelope.resolve()
    assert event.synchronize.call_count == 1
    assert scheduler.outbox.empty()
    assert batch.reqs[0].omni_data.speculative_emitted_tokens == 0
    scheduler.shutdown_resources.assert_not_called()


@pytest.mark.parametrize("flag", ["enable_early_first_token"])
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
