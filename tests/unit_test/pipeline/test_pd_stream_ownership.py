# SPDX-License-Identifier: Apache-2.0
"""The Decode stage owns first-sample streaming and request-local failures."""

import threading
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.runtime_context import get_context

from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.pd_scheduler import OmniDecodeScheduler
from sglang_omni.scheduling.pd_state import PDDecodeBatchResultProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput
from tests.unit_test.pipeline.test_pd_lifecycle import decode_scheduler
from tests.unit_test.pipeline.test_pd_utils import prefill_req, rebuilt_req


def build_stream(
    request_id: str, request_data: SGLangARRequestData, output: RequestOutput
) -> list[OutgoingMessage]:
    if not request_data.stage_payload.request.params["stream"]:
        return []
    else:
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=torch.tensor([output.data]),
                target="decode",
            )
        ]


@pytest.fixture
def stream_scheduler() -> Iterator[OmniDecodeScheduler]:
    scheduler = decode_scheduler()
    scheduler.emit_transferred_token = True
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.completed_request_ids = {}
    scheduler.pending_stream_ingress = {}
    scheduler.request_admission_lock = threading.RLock()
    scheduler.is_entry_rank = True
    scheduler.model_runner = None
    scheduler.request_finished_callback = None
    scheduler.abort_callback = Mock()
    scheduler.result_adapter = lambda request_data: request_data.stage_payload
    scheduler.stream_output_builder = build_stream
    with get_context().override_server_args(weight_version=None):
        yield scheduler


@pytest.mark.parametrize(
    "mode, expected_types",
    [
        ("stream", ["stream"]),
        ("aborted", []),
        ("abort_reason", []),
        ("nonstream", []),
        ("skip", []),
        ("disabled", []),
        ("max_tokens", ["stream", "result"]),
        ("stop_token", ["stream", "result"]),
    ],
)
def test_first_sample_streams_once_before_terminal(
    stream_scheduler: OmniDecodeScheduler, mode: str, expected_types: list[str]
) -> None:
    request = rebuilt_req(prefill_req(max_new_tokens=1 if mode == "max_tokens" else 16))
    request.omni_data.stage_payload.request.params["stream"] = mode != "nonstream"
    if mode == "abort_reason":
        request.finished_reason = FINISH_ABORT("cancelled")
    elif mode == "stop_token":
        request.sampling_params.stop_token_ids = {42}
    else:
        pass
    request.update_finish_state()
    stream_scheduler.emit_transferred_token = mode != "disabled"
    stream_scheduler.aborted_request_ids = {request.rid} if mode == "aborted" else set()

    for _ in range(2):
        stream_scheduler.stream_output(
            [request], skip_req=request if mode == "skip" else None
        )
    messages = list(stream_scheduler.outbox.queue)
    assert [message.type for message in messages] == expected_types
    if expected_types:
        assert messages[0].target == "decode"
        assert messages[0].data.tolist() == [42]
    else:
        pass


@pytest.mark.parametrize("partial_output", [False, True])
def test_first_sample_failure_preserves_other_requests(
    stream_scheduler: OmniDecodeScheduler, partial_output: bool
) -> None:
    """A failed callback or iterator drops only its request's messages and KV."""
    requests = [rebuilt_req(prefill_req()) for _ in range(3)]
    for index, request in enumerate(requests):
        request.rid = f"request-{index}"
    failed_request = requests[1]
    error = RuntimeError("stream failed")

    def incomplete_stream(message: OutgoingMessage) -> Iterator[OutgoingMessage]:
        yield message
        raise error

    def build_or_fail(
        request_id: str, request_data: SGLangARRequestData, output: RequestOutput
    ) -> list[OutgoingMessage] | Iterator[OutgoingMessage]:
        messages = build_stream(request_id, request_data, output)
        if request_id != failed_request.rid:
            return messages
        elif partial_output:
            return incomplete_stream(messages[0])
        else:
            raise error

    stream_scheduler.stream_output_builder = build_or_fail
    stream_scheduler.release_request_kv_cache = Mock()
    stream_scheduler.stream_output(requests)

    messages = list(stream_scheduler.outbox.queue)
    assert [(message.type, message.request_id) for message in messages] == [
        ("error", failed_request.rid),
        ("stream", requests[0].rid),
        ("stream", requests[2].rid),
    ]
    assert messages[0].data is error
    assert isinstance(failed_request.finished_reason, FINISH_ABORT)
    assert failed_request.omni_data is None
    assert failed_request.rid in stream_scheduler.completed_request_ids
    stream_scheduler.release_request_kv_cache.assert_called_once_with(failed_request)
    stream_scheduler.abort_callback.assert_called_once_with(failed_request.rid)


@pytest.mark.parametrize("max_new_tokens", [1, 4])
def test_main_prebuilt_emits_transferred_sample_before_decode(
    stream_scheduler, max_new_tokens
):
    """Main's PREBUILT callback omits unfinished requests unless Omni publishes them."""
    request = rebuilt_req(prefill_req(max_new_tokens=max_new_tokens))
    processor = PDDecodeBatchResultProcessor(
        SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=DisaggregationMode.DECODE,
            token_to_kv_pool_allocator=Mock(),
            request_retirer=Mock(),
            output_streamer=SimpleNamespace(
                stream_output=stream_scheduler.stream_output
            ),
            model_config=None,
            tree_cache=None,
            hisparse_coordinator=None,
            req_to_token_pool=None,
            kv_weight_version_tracker=None,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=None,
            draft_worker=None,
            model_worker=None,
            logprob_result_processor=None,
            beam_coordinator=None,
            abort_request=None,
            still_in_batch=None,
            prefill_launched_reqs=None,
        )
    )
    batch = ScheduleBatch(reqs=[request])
    processor.process_batch_result_prebuilt(batch)
    messages = list(stream_scheduler.outbox.queue)
    assert [message.type for message in messages] == (
        ["stream", "result"] if max_new_tokens == 1 else ["stream"]
    )
    assert messages[0].data.tolist() == [42]
    assert request.finished() == (max_new_tokens == 1)
    if max_new_tokens == 1:
        processor.request_retirer.release.assert_called_once_with(
            request, checkpoint=True
        )
    else:
        processor.request_retirer.release.assert_not_called()
