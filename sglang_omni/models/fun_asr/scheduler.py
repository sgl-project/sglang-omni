# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU preparation for complete Fun-ASR transcription requests."""

from __future__ import annotations

import threading
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from functools import partial
from typing import Unpack
from weakref import ReferenceType, ref

from sglang_omni.models.fun_asr.request_builders import (
    FunASRRequestBuilder,
    FunASRRequestData,
    PreparedFunASRInput,
)
from sglang_omni.profiler.event_recorder import emit, get_active_stage
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SESSION_METADATA_KEY
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, OmniSchedulerArguments


def snapshot_cpu_payload(payload: StagePayload) -> StagePayload | None:
    """Snapshot only owned bytes and simple parameters; other inputs stay lazy."""
    parameters = payload.request.params or {}
    if (
        SESSION_METADATA_KEY in payload.request.metadata
        or parameters.get("_asr_streaming") is True
        or payload.prefetched_chunks
        or payload.prefetched_stream_done
    ):
        return None
    else:
        pass
    inputs = payload.request.inputs
    if isinstance(inputs, bytes):
        copied_inputs = inputs
    elif isinstance(inputs, dict):
        source = next(
            (
                inputs[key]
                for key in ("audio_bytes", "bytes", "file")
                if inputs.get(key) is not None
            ),
            None,
        )
        if not isinstance(source, bytes):
            return None
        else:
            copied_inputs = dict(inputs)
    else:
        return None
    copied_parameters: dict[str, object] = {}
    for name, value in parameters.items():
        if value is None or type(value) in (str, int, float, bool):
            copied_parameters[name] = value
        elif isinstance(value, list) and all(type(word) is str for word in value):
            copied_parameters[name] = list(value)
        else:
            return None
    return StagePayload(
        request_id=payload.request_id,
        request=OmniRequest(inputs=copied_inputs, params=copied_parameters),
        data=None,
    )


@dataclass(kw_only=True)
class CPUPreparation:
    payload: StagePayload
    snapshot: StagePayload
    cpu_future: Future[PreparedFunASRInput]
    active_stage: str | None
    build_future: Future[FunASRRequestData] | None = None
    finish_future: Future[FunASRRequestData] | None = None
    cancelled: bool = False


class FunASRScheduler(OmniScheduler[FunASRRequestData]):
    """Prepare at most one unadmitted request using the existing worker pool."""

    def __init__(
        self,
        *,
        cpu_input_builder: FunASRRequestBuilder,
        **scheduler_arguments: Unpack[OmniSchedulerArguments[FunASRRequestData]],
    ) -> None:
        self.cpu_input_builder = cpu_input_builder
        self.cpu_preparation: CPUPreparation | None = None
        self.cpu_preparation_closed = False
        self.is_processing_inputs = False
        self.has_preparation_slot_credit = False
        self.preparation_shutdown_lock = threading.Lock()
        super().__init__(**scheduler_arguments)

    def enqueue(self, message: IncomingMessage) -> None:
        with self.request_admission_lock:
            executor = self.request_build_executor
            if (
                message.type == "new_request"
                and isinstance(message.data, StagePayload)
                and executor is not None
                and not self.cpu_preparation_closed
                and not self.stop_requested
                and not self.is_processing_inputs
                and self.cpu_preparation is None
                and self.tp_size == 1
                and not self.enable_overlap
                and self.session_bridge is None
                and self.stream_chunk_handler is None
                and self.stream_done_handler is None
                and message.request_id not in self.aborted_request_ids
                and message.request_id not in self.pending_stream_ingress
                and not self.backlogged_request_build_payloads
                and not self.deferred_request_payloads
                and not self.pending_request_admissions
                and self.inbox.empty()
                and not self.waiting_queue_is_full()
                and not self.pending_request_builds
                and self.request_build_max_pending > 1
                and self.request_build_slots_in_use() < self.request_build_max_pending
            ):
                snapshot = snapshot_cpu_payload(message.data)
            else:
                snapshot = None
            if snapshot is not None:
                active_stage = get_active_stage()
                try:
                    future = executor.submit(
                        self.prepare_cpu_input, snapshot, active_stage
                    )
                except RuntimeError:
                    # note (Codex): a concurrently closed executor must not reject an otherwise valid inbox message.
                    pass
                else:
                    preparation = CPUPreparation(
                        payload=message.data,
                        snapshot=snapshot,
                        cpu_future=future,
                        active_stage=active_stage,
                    )
                    self.cpu_preparation = preparation
                    future.add_done_callback(
                        partial(self.cpu_prepared, ref(preparation))
                    )
            else:
                pass
            self.inbox.put(message)

    def prepare_cpu_input(
        self, payload: StagePayload, active_stage: str | None
    ) -> PreparedFunASRInput:
        with self.request_admission_lock:
            preparation = self.cpu_preparation
            if (
                preparation is None
                or preparation.snapshot is not payload
                or preparation.cancelled
                or self.cpu_preparation_closed
            ):
                raise CancelledError()
            else:
                pass
        emit(
            request_id=payload.request_id,
            stage=active_stage,
            event_name="scheduler_cpu_prepare_start",
        )
        try:
            return self.cpu_input_builder.prepare_cpu_input(payload)
        finally:
            emit(
                request_id=payload.request_id,
                stage=active_stage,
                event_name="scheduler_cpu_prepare_end",
            )

    def request_build_slots_in_use(self) -> int:
        preparation = self.cpu_preparation
        pending = self.pending_request_builds
        if (
            preparation is not None
            and not preparation.cancelled
            and not self.has_preparation_slot_credit
            and (
                preparation.payload.request_id not in pending
                or pending[preparation.payload.request_id][2]
                is not preparation.build_future
            )
        ):
            return len(pending) + 1
        else:
            return len(pending)

    def recv_requests(self) -> list[StagePayload]:
        with self.request_admission_lock:
            self.is_processing_inputs = True
        try:
            return super().recv_requests()
        except Exception:
            with self.request_admission_lock:
                self.is_processing_inputs = False
            raise

    def process_input_requests(self, recv_reqs: list[StagePayload]) -> None:
        with self.preparation_shutdown_lock:
            if self.cpu_preparation_closed:
                return
            else:
                pass
            with self.request_admission_lock:
                self.is_processing_inputs = True
            try:
                super().process_input_requests(recv_reqs)
            finally:
                with self.request_admission_lock:
                    self.is_processing_inputs = False

    def sleep_during_idle(self) -> None:
        # note (Codex): idle receive owns the next input before publishing idle_wait_message; keep ingress reserved until that input pass finishes.
        with self.request_admission_lock:
            self.is_processing_inputs = True
        super().sleep_during_idle()

    def stage_request_build_payloads(
        self, recv_reqs: list[StagePayload]
    ) -> tuple[list[StagePayload], list[StagePayload]]:
        with self.request_admission_lock:
            preparation = self.cpu_preparation
            # note (Codex): the prepared head already owns a slot; counting it twice would reject a request accepted by the ordinary path.
            self.has_preparation_slot_credit = bool(
                preparation is not None
                and not preparation.cancelled
                and recv_reqs
                and recv_reqs[0] is preparation.payload
                and not self.pending_request_builds
                and not self.backlogged_request_build_payloads
            )
            try:
                return super().stage_request_build_payloads(recv_reqs)
            finally:
                self.has_preparation_slot_credit = False

    def submit_request_build(
        self, payload: StagePayload, active_stage: str | None
    ) -> Future[FunASRRequestData]:
        preparation = self.cpu_preparation
        if (
            preparation is not None
            and preparation.payload is payload
            and not preparation.cancelled
        ):
            assert preparation.build_future is None
            preparation.active_stage = active_stage
            preparation.build_future = Future()
            emit(
                request_id=payload.request_id,
                stage=active_stage,
                event_name="scheduler_cpu_prepare_claimed",
            )
            if preparation.cpu_future.done():
                self.cpu_prepared(ref(preparation), preparation.cpu_future)
            else:
                pass
            return preparation.build_future
        else:
            return super().submit_request_build(payload, active_stage)

    def cpu_prepared(
        self,
        preparation_reference: ReferenceType[CPUPreparation],
        future: Future[PreparedFunASRInput],
    ) -> None:
        preparation = preparation_reference()
        if preparation is None:
            return
        else:
            pass
        with self.request_admission_lock:
            if preparation.cancelled or self.cpu_preparation_closed:
                self.release_cancelled_preparation(preparation)
                return
            elif (
                preparation.build_future is None
                or preparation.build_future.done()
                or preparation.finish_future is not None
            ):
                return
            else:
                pass
            executor = self.request_build_executor
            assert executor is not None
            try:
                prepared = future.result()
                preparation.finish_future = executor.submit(
                    self.finish_prepared_request, preparation, prepared
                )
            except Exception as error:
                preparation.build_future.set_exception(error)
                if self.cpu_preparation is preparation:
                    self.cpu_preparation = None
                else:
                    pass
            else:
                preparation.finish_future.add_done_callback(
                    partial(self.request_build_finished, ref(preparation))
                )

    def finish_prepared_request(
        self, preparation: CPUPreparation, prepared: PreparedFunASRInput
    ) -> FunASRRequestData:
        with self.request_admission_lock:
            if preparation.cancelled or self.cpu_preparation_closed:
                raise CancelledError()
            else:
                pass
        emit(
            request_id=preparation.payload.request_id,
            stage=preparation.active_stage,
            event_name="scheduler_request_build_start",
        )
        try:
            return self.cpu_input_builder.finish_request(preparation.payload, prepared)
        finally:
            emit(
                request_id=preparation.payload.request_id,
                stage=preparation.active_stage,
                event_name="scheduler_request_build_end",
            )

    def request_build_finished(
        self,
        preparation_reference: ReferenceType[CPUPreparation],
        future: Future[FunASRRequestData],
    ) -> None:
        preparation = preparation_reference()
        if preparation is None:
            return
        else:
            pass
        with self.request_admission_lock:
            result_future = preparation.build_future
            assert result_future is not None
            if not result_future.cancelled():
                try:
                    result = future.result()
                except Exception as error:
                    result_future.set_exception(error)
                else:
                    result_future.set_result(result)
            else:
                pass
            if self.cpu_preparation is preparation:
                self.cpu_preparation = None
            else:
                pass

    def release_cancelled_preparation(self, preparation: CPUPreparation) -> None:
        if (
            self.cpu_preparation is preparation
            and preparation.cpu_future.done()
            and (preparation.finish_future is None or preparation.finish_future.done())
        ):
            self.cpu_preparation = None
        else:
            pass

    def cancel_cpu_preparation(self, preparation: CPUPreparation) -> None:
        preparation.cancelled = True
        if preparation.build_future is not None:
            preparation.build_future.cancel()
        else:
            pass
        # note (Codex): keep the early slot until cancelled work leaves the executor. Logical admission capacity is released as in ordinary aborts.
        self.release_cancelled_preparation(preparation)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        with self.request_admission_lock:
            preparation = self.cpu_preparation
            if preparation is not None and preparation.payload.request_id == request_id:
                self.cancel_cpu_preparation(preparation)
            else:
                pass
            super().abort(request_id, defer_running_cleanup=defer_running_cleanup)

    def shutdown_request_build_executor(self) -> None:
        with self.preparation_shutdown_lock:
            with self.request_admission_lock:
                self.cpu_preparation_closed = True
                preparation = self.cpu_preparation
                if preparation is not None:
                    self.cancel_cpu_preparation(preparation)
                else:
                    pass
                executor = self.request_build_executor
                self.request_build_executor = None
            if executor is not None:
                # note (Codex): the encoder must outlive every already running build, including ordinary fallbacks.
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                pass

    def shutdown_resources(self) -> None:
        self.shutdown_request_build_executor()
        super().shutdown_resources()
