# SPDX-License-Identifier: Apache-2.0
"""Deferred writer lifecycle for request-level profiling events."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from io import TextIOWrapper
from pathlib import Path

from sglang_omni.profiler.event_recorder_types import (
    DEFAULT_EVENT_QUEUE_CAPACITY,
    EVENT_FILE_BUFFER_BYTES,
    MAX_EVENT_TEXT_LENGTH,
    MAX_PRODUCER_BUFFERS,
    WRITER_POLL_INTERVAL_SECONDS,
    ProducerBuffer,
    QueuedRequestEvent,
    RecorderRunState,
    RequestEvent,
    RequestEventFinalization,
    get_active_stage,
    json_default,
    snapshot_metadata,
)

logger = logging.getLogger(__name__)


class RequestEventRecorder:
    """Record into producer-private buffers and write only after stop.

    The active path owns no writer thread, file handle, JSON serialization, or
    shared queue. Each OS thread gets one bounded buffer per recording run.
    The post-stop writer waits for producer quiescence, drains those buffers,
    and publishes one process-local status sidecar.
    """

    def __init__(self, queue_capacity: int = DEFAULT_EVENT_QUEUE_CAPACITY) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be greater than zero")
        else:
            pass
        self.queue_capacity = queue_capacity
        self._control_lock = threading.Lock()
        self._producer_local = threading.local()
        self.active_state: RecorderRunState | None = None
        self.pending_states: dict[int, RecorderRunState] = {}
        self.writer_thread: threading.Thread | None = None
        self.writer_failed = False
        # note(chenye): Individual handles remain authoritative for each run;
        # this aggregate keeps shutdown from forgetting a failed publish.
        self.finalization_failed = False
        self.writer_start_count = 0
        self._in_flight_registrations = 0
        self.pid = os.getpid()
        self._path_generation = 0

    def is_active(self) -> bool:
        return self.active_state is not None

    def active_run_id(self) -> str | None:
        active_state = self.active_state
        if active_state is None:
            return None
        else:
            pass
        return active_state.run_id

    def capture_active_run_timestamp_ns(self) -> tuple[str, int] | None:
        """Capture the active run ID and wall-clock timestamp together."""
        active_state = self.active_state
        if active_state is None:
            return None
        else:
            pass
        return active_state.run_id, time.time_ns()

    def active_path(self) -> str | None:
        active_state = self.active_state
        if active_state is None:
            return None
        else:
            pass
        return str(active_state.path)

    def emit(
        self,
        *,
        request_id: str,
        stage: str | None,
        event_name: str,
        metadata: Mapping[str, object] | None = None,
        timestamp_ns: int | None = None,
        expected_run_id: str | None = None,
    ) -> None:
        """Capture one event without waiting for a writer or storage."""
        self.emit_record(
            request_id=request_id,
            stage=stage,
            event_name=event_name,
            metadata=metadata,
            timestamp_ns=timestamp_ns,
            expected_run_id=expected_run_id,
        )

    def emit_record(
        self,
        *,
        request_id: str,
        stage: str | None,
        event_name: str,
        metadata: Mapping[str, object] | None = None,
        timestamp_ns: int | None = None,
        expected_run_id: str | None = None,
    ) -> bool:
        """Capture one event without allowing recorder failures to escape."""
        try:
            return self._emit_record(
                request_id=request_id,
                stage=stage,
                event_name=event_name,
                metadata=metadata,
                timestamp_ns=timestamp_ns,
                expected_run_id=expected_run_id,
            )
        except Exception:
            run_id = expected_run_id or self.active_run_id()
            if run_id is not None:
                try:
                    self.record_external_loss(run_id, 1, "producer_emit_failure")
                except Exception:
                    pass
            else:
                pass
            return False

    def _emit_record(
        self,
        *,
        request_id: str,
        stage: str | None,
        event_name: str,
        metadata: Mapping[str, object] | None = None,
        timestamp_ns: int | None = None,
        expected_run_id: str | None = None,
    ) -> bool:
        """Capture one event and report whether it was retained.

        This return value is for deferred internal buffers. The public
        emit facade intentionally keeps its historical None return.
        """
        cached_buffer = self._cached_producer_buffer()
        if cached_buffer is not None:
            cached_buffer.in_flight += 1
        else:
            pass
        try:
            run_state = self.active_state
            if run_state is None:
                return False
            else:
                pass
            if expected_run_id is not None and run_state.run_id != expected_run_id:
                return False
            else:
                pass

            if cached_buffer is not None and cached_buffer.run_state is run_state:
                return self._emit_to_buffer(
                    run_state,
                    cached_buffer,
                    request_id=request_id,
                    stage=stage,
                    event_name=event_name,
                    metadata=metadata,
                    timestamp_ns=timestamp_ns,
                    expected_run_id=expected_run_id,
                )
            else:
                pass

            with self._control_lock:
                if self.active_state is not run_state:
                    return False
                else:
                    pass
                self._in_flight_registrations += 1
            try:
                producer_buffer = self._producer_buffer_for(run_state)
                if producer_buffer is None:
                    return False
                else:
                    pass
                return self._emit_from_buffer(
                    run_state,
                    producer_buffer,
                    request_id=request_id,
                    stage=stage,
                    event_name=event_name,
                    metadata=metadata,
                    timestamp_ns=timestamp_ns,
                    expected_run_id=expected_run_id,
                )
            finally:
                with self._control_lock:
                    self._in_flight_registrations -= 1
        finally:
            if cached_buffer is not None:
                cached_buffer.in_flight -= 1
            else:
                pass

    def _cached_producer_buffer(self) -> ProducerBuffer | None:
        handles = getattr(self._producer_local, "handles", None)
        if handles is None:
            return None
        else:
            pass
        return handles.get(id(self))

    def _emit_from_buffer(
        self,
        run_state: RecorderRunState,
        producer_buffer: ProducerBuffer,
        *,
        request_id: str,
        stage: str | None,
        event_name: str,
        metadata: Mapping[str, object] | None,
        timestamp_ns: int | None,
        expected_run_id: str | None,
    ) -> bool:
        producer_buffer.in_flight += 1
        try:
            return self._emit_to_buffer(
                run_state,
                producer_buffer,
                request_id=request_id,
                stage=stage,
                event_name=event_name,
                metadata=metadata,
                timestamp_ns=timestamp_ns,
                expected_run_id=expected_run_id,
            )
        finally:
            producer_buffer.in_flight -= 1

    def _emit_to_buffer(
        self,
        run_state: RecorderRunState,
        producer_buffer: ProducerBuffer,
        *,
        request_id: str,
        stage: str | None,
        event_name: str,
        metadata: Mapping[str, object] | None,
        timestamp_ns: int | None,
        expected_run_id: str | None,
    ) -> bool:
        if self.active_state is not run_state:
            return False
        else:
            pass
        if expected_run_id is not None and run_state.run_id != expected_run_id:
            return False
        else:
            pass

        if len(producer_buffer.events) >= producer_buffer.capacity:
            producer_buffer.dropped_events += 1
            producer_buffer.incomplete_reasons.add("producer_buffer_overflow")
            return False
        else:
            pass

        resolved_stage = stage
        if resolved_stage is None:
            resolved_stage = get_active_stage() or run_state.stage or "unknown"
        else:
            pass
        captured_timestamp_ns = (
            timestamp_ns if timestamp_ns is not None else time.time_ns()
        )

        try:
            captured_metadata = snapshot_metadata(metadata)
        except Exception:
            producer_buffer.metadata_snapshot_failures += 1
            producer_buffer.dropped_events += 1
            producer_buffer.incomplete_reasons.add("metadata_snapshot_failure")
            return False
        else:
            pass

        metadata_truncated = "__metadata_truncated__" in captured_metadata
        if metadata_truncated:
            producer_buffer.metadata_truncations += 1
            producer_buffer.incomplete_reasons.add("metadata_truncated")
        else:
            pass

        if not self._event_text_is_bounded(
            request_id, resolved_stage, event_name, run_state.run_id
        ):
            producer_buffer.dropped_events += 1
            producer_buffer.incomplete_reasons.add("event_payload_too_large")
            return False
        else:
            pass

        if self.active_state is not run_state:
            return False
        else:
            pass
        if expected_run_id is not None and run_state.run_id != expected_run_id:
            return False
        else:
            pass
        event = RequestEvent(
            request_id=request_id,
            stage=resolved_stage,
            event_name=event_name,
            timestamp_ns=captured_timestamp_ns,
            run_id=run_state.run_id,
            pid=self.pid,
            metadata=captured_metadata,
        )
        producer_buffer.events.append(
            QueuedRequestEvent(run_state=run_state, event=event)
        )
        producer_buffer.accepted_events += 1
        return True

    def start(self, run_id: str, event_dir: str, stage: str) -> str:
        """Start or join one process-local recording run.

        Starting a different run while one is active or draining is rejected.
        This keeps measured windows explicit and prevents a new capture from
        sharing a writer with a previous window.
        """
        with self._control_lock:
            active_state = self.active_state
            if active_state is not None:
                if active_state.run_id == run_id:
                    active_state.stages.add(stage)
                    active_state.active_stages.add(stage)
                    return str(active_state.path)
                else:
                    raise RuntimeError(
                        f"request event run {active_state.run_id} is already active"
                    )
            else:
                pass
            if self.pending_states:
                raise RuntimeError(
                    "request event recorder is still draining the previous run"
                )
            else:
                pass
            if self.writer_thread is not None:
                raise RuntimeError(
                    "request event recorder writer is still shutting down"
                )
            else:
                pass

            directory = Path(event_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            directory = directory.resolve()
            path, status_path = self._allocate_paths_unlocked(directory, stage)
            finalization = RequestEventFinalization(
                path=str(path),
                status_path=str(status_path),
                completion_event=threading.Event(),
            )
            run_state = RecorderRunState(
                run_id=run_id,
                stage=stage,
                stages={stage},
                active_stages={stage},
                path=path,
                status_path=status_path,
                finalization=finalization,
            )
            self.active_state = run_state
            return str(path)

    def _allocate_paths_unlocked(
        self, directory: Path, stage: str
    ) -> tuple[Path, Path]:
        base_path = directory / f"events_{stage}_{self.pid}.jsonl"
        base_status_path = directory / f"run_status_{self.pid}.json"
        if not base_path.exists() and not base_status_path.exists():
            return base_path, base_status_path
        else:
            pass
        self._path_generation += 1
        suffix = f"_{self._path_generation}"
        return (
            directory / f"events_{stage}_{self.pid}{suffix}.jsonl",
            directory / f"run_status_{self.pid}{suffix}.json",
        )

    def validate_start_unlocked(self) -> None:
        """Validate a new run while the control lock is held."""
        if self.active_state is not None:
            raise RuntimeError("request event recorder is active")
        else:
            pass
        if self.pending_states:
            raise RuntimeError(
                "request event recorder is still draining the previous run"
            )
        else:
            pass
        self.writer_failed = False

    def _producer_buffer_for(
        self, run_state: RecorderRunState
    ) -> ProducerBuffer | None:
        handles = getattr(self._producer_local, "handles", None)
        if handles is None:
            handles: dict[int, ProducerBuffer] = {}
            self._producer_local.handles = handles
        else:
            pass
        recorder_key = id(self)
        cached_buffer = handles.get(recorder_key)
        if cached_buffer is not None and cached_buffer.run_state is run_state:
            return cached_buffer
        else:
            pass
        with self._control_lock:
            if self.active_state is not run_state:
                if id(run_state) in self.pending_states:
                    run_state.dropped_events += 1
                    run_state.incomplete_reasons.add(
                        "producer_buffer_registration_after_stop"
                    )
                else:
                    pass
                return None
            else:
                pass
            if len(run_state.producer_buffers) >= MAX_PRODUCER_BUFFERS:
                run_state.dropped_events += 1
                run_state.incomplete_reasons.add("producer_buffer_registration_limit")
                return None
            else:
                pass
            producer_buffer = ProducerBuffer(
                run_state=run_state,
                thread_id=threading.get_ident(),
                capacity=self.queue_capacity,
            )
            run_state.producer_buffers.append(producer_buffer)
            handles[recorder_key] = producer_buffer
            return producer_buffer

    @staticmethod
    def _event_text_is_bounded(
        request_id: str,
        stage: str,
        event_name: str,
        run_id: str,
    ) -> bool:
        return all(
            len(text) <= MAX_EVENT_TEXT_LENGTH
            for text in (request_id, stage, event_name, run_id)
        )

    def register_deferred_buffer(
        self, run_id: str, finalizer: Callable[[], int]
    ) -> bool:
        """Register one bounded RequestEventBuffer state for finalization."""
        with self._control_lock:
            active_state = self.active_state
            if active_state is None or active_state.run_id != run_id:
                return False
            else:
                pass
            if len(active_state.deferred_buffer_finalizers) >= MAX_PRODUCER_BUFFERS:
                active_state.dropped_events += 1
                active_state.incomplete_reasons.add(
                    "deferred_buffer_registration_limit"
                )
                return False
            else:
                pass
            active_state.deferred_buffer_finalizers.append(finalizer)
            return True

    def record_external_loss(self, run_id: str, count: int, reason: str) -> bool:
        """Account for bounded deferred records that could not be emitted."""
        if count <= 0:
            return True
        else:
            pass
        with self._control_lock:
            run_state = self.active_state
            if run_state is None or run_state.run_id != run_id:
                run_state = next(
                    (
                        candidate
                        for candidate in self.pending_states.values()
                        if candidate.run_id == run_id
                    ),
                    None,
                )
            else:
                pass
            if run_state is None:
                return False
            else:
                pass
            run_state.dropped_events += count
            run_state.incomplete_reasons.add(reason)
            return True

    def record_external_metadata_snapshot_failure(
        self, run_id: str, count: int
    ) -> bool:
        """Account for deferred records rejected during metadata snapshotting."""
        if count <= 0:
            return True
        else:
            pass
        with self._control_lock:
            run_state = self.active_state
            if run_state is None or run_state.run_id != run_id:
                run_state = next(
                    (
                        candidate
                        for candidate in self.pending_states.values()
                        if candidate.run_id == run_id
                    ),
                    None,
                )
            else:
                pass
            if run_state is None:
                return False
            else:
                pass
            run_state.metadata_snapshot_failures += count
            run_state.dropped_events += count
            run_state.incomplete_reasons.add(
                "deferred_buffer_metadata_snapshot_failure"
            )
            return True

    def _retire_active_state_unlocked(self) -> RecorderRunState | None:
        run_state = self.active_state
        if run_state is None:
            return None
        else:
            pass
        self.active_state = None
        run_state.active_stages.clear()
        self.pending_states[id(run_state)] = run_state
        self._start_writer_unlocked()
        return run_state

    def release_stage(self, stage: str) -> RequestEventFinalization | None:
        """Release one co-located stage without stopping its shared run."""
        with self._control_lock:
            active_state = self.active_state
            if active_state is None or stage not in active_state.active_stages:
                return None
            else:
                pass
            active_state.active_stages.remove(stage)
            if active_state.active_stages:
                return None
            else:
                pass
            retired_state = self._retire_active_state_unlocked()
            if retired_state is None:
                return None
            else:
                pass
            return retired_state.finalization

    def deactivate(
        self, *, run_id: str | None = None
    ) -> RequestEventFinalization | None:
        """Deactivate a run without waiting for file finalization."""
        with self._control_lock:
            active_state = self.active_state
            if active_state is None:
                return None
            else:
                pass
            if run_id is not None and active_state.run_id != run_id:
                return None
            else:
                pass
            retired_state = self._retire_active_state_unlocked()
            if retired_state is None:
                return None
            else:
                pass
            return retired_state.finalization

    def stop(
        self,
        *,
        run_id: str | None = None,
        wait: bool = False,
    ) -> str | None:
        """Deactivate the active run; waiting is explicit and administrative."""
        finalization = self.deactivate(run_id=run_id)
        if finalization is None:
            return None
        else:
            pass
        if wait:
            if not finalization.wait():
                return None
            else:
                pass
        else:
            pass
        return finalization.path

    def wait_for_finalizations(self, timeout_seconds: float | None = None) -> bool:
        """Wait for all retired runs currently known to this recorder."""
        with self._control_lock:
            finalizations = [
                state.finalization for state in self.pending_states.values()
            ]
            failed = self.finalization_failed
        if timeout_seconds is None:
            completed = all(finalization.wait() for finalization in finalizations)
        else:
            deadline = time.monotonic() + timeout_seconds
            completed = True
            for finalization in finalizations:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not finalization.wait(remaining):
                    completed = False
                    break
                else:
                    pass
        with self._control_lock:
            return completed and not self.finalization_failed and not failed

    def _start_writer_unlocked(self) -> None:
        if self.writer_thread is not None or not self.pending_states:
            return
        else:
            pass
        writer = threading.Thread(
            target=self.writer_loop,
            name=f"request-event-writer-{self.pid}",
            daemon=True,
        )
        self.writer_thread = writer
        self.writer_start_count += 1
        try:
            writer.start()
        except Exception:
            self.writer_thread = None
            self.writer_failed = True
            states = list(self.pending_states.values())
            for state in states:
                state.finalization_failures += 1
                state.incomplete_reasons.add("writer_thread_start_failure")
                self._complete_state_unlocked(state, successful=False)
            logger.warning("RequestEventRecorder could not start its writer thread")
        else:
            pass

    def writer_loop(self) -> None:
        """Drain all retired runs; this method is never entered while active."""
        self._set_writer_scheduling()
        try:
            while True:
                with self._control_lock:
                    states = list(self.pending_states.values())
                    if not states:
                        if self.writer_thread is threading.current_thread():
                            self.writer_thread = None
                        else:
                            pass
                        return
                    else:
                        pass
                for run_state in states:
                    self._finalize_state(run_state)
        except Exception:
            logger.exception("RequestEventRecorder writer thread crashed")
            self.writer_failed = True
            with self._control_lock:
                states = list(self.pending_states.values())
            for run_state in states:
                self._fail_state(run_state)
        finally:
            with self._control_lock:
                if self.writer_thread is threading.current_thread():
                    self.writer_thread = None
                else:
                    pass

    def _set_writer_scheduling(self) -> None:
        scheduling = "default"
        if hasattr(os, "sched_setscheduler") and hasattr(os, "SCHED_IDLE"):
            try:
                os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
            except Exception:
                logger.info(
                    "RequestEventRecorder SCHED_IDLE unavailable; using default scheduling"
                )
            else:
                scheduling = "SCHED_IDLE"
        else:
            logger.info(
                "RequestEventRecorder SCHED_IDLE unsupported; using default scheduling"
            )
        with self._control_lock:
            for run_state in self.pending_states.values():
                run_state.writer_scheduling = scheduling

    def _wait_for_producer_quiescence(self, run_state: RecorderRunState) -> None:
        while True:
            with self._control_lock:
                registrations = self._in_flight_registrations
                buffers = tuple(run_state.producer_buffers)
            if registrations == 0 and all(buffer.in_flight == 0 for buffer in buffers):
                return
            else:
                pass
            time.sleep(WRITER_POLL_INTERVAL_SECONDS)

    def _seal_deferred_buffers(self, run_state: RecorderRunState) -> None:
        with self._control_lock:
            finalizers = tuple(run_state.deferred_buffer_finalizers)
        for finalizer in finalizers:
            try:
                pending_count = finalizer()
            except Exception:
                run_state.finalization_failures += 1
                run_state.incomplete_reasons.add("deferred_buffer_finalization_failure")
            else:
                if pending_count > 0:
                    run_state.dropped_events += pending_count
                    run_state.incomplete_reasons.add("deferred_buffer_unflushed")
                else:
                    pass

    def _aggregate_producer_buffers(self, run_state: RecorderRunState) -> None:
        if run_state.producer_counters_aggregated:
            return
        else:
            pass
        for producer_buffer in tuple(run_state.producer_buffers):
            run_state.accepted_events += producer_buffer.accepted_events
            run_state.dropped_events += producer_buffer.dropped_events
            run_state.metadata_snapshot_failures += (
                producer_buffer.metadata_snapshot_failures
            )
            run_state.metadata_truncations += producer_buffer.metadata_truncations
            run_state.incomplete_reasons.update(producer_buffer.incomplete_reasons)
        run_state.producer_counters_aggregated = True

    def _finalize_state(self, run_state: RecorderRunState) -> bool:
        with self._control_lock:
            if id(run_state) not in self.pending_states:
                return False
            else:
                pass

        self._wait_for_producer_quiescence(run_state)
        self._seal_deferred_buffers(run_state)
        self._aggregate_producer_buffers(run_state)
        file_handle: TextIOWrapper | None = None
        status_published = False
        try:
            file_handle = self.open_event_file(run_state)
            for producer_buffer in tuple(run_state.producer_buffers):
                for queued_event in tuple(producer_buffer.events):
                    run_state.processed_events += 1
                    try:
                        file_handle = self.write_queued_event(queued_event, file_handle)
                    except Exception:
                        run_state.finalization_failures += 1
                        run_state.incomplete_reasons.add("writer_callback_failure")
                        self._flush_and_close(run_state, file_handle)
                        file_handle = None
                    else:
                        pass
            file_handle = self._flush_and_close(run_state, file_handle)
            status_published = self._publish_status(run_state)
        except Exception:
            run_state.finalization_failures += 1
            run_state.incomplete_reasons.add("finalization_failure")
            logger.warning(
                "RequestEventRecorder failed while finalizing %s",
                run_state.run_id,
                exc_info=True,
            )
            self._flush_and_close(run_state, file_handle)
            status_published = self._publish_status(run_state)
        finally:
            with self._control_lock:
                if not run_state.finalization.completion_event.is_set():
                    finalization_successful = (
                        status_published
                        and not run_state.incomplete_reasons
                        and run_state.serialization_failures == 0
                        and run_state.write_failures == 0
                        and run_state.finalization_failures == 0
                    )
                    self._complete_state_unlocked(
                        run_state, successful=finalization_successful
                    )
                else:
                    pass
        return run_state.finalization.successful

    def open_event_file(self, run_state: RecorderRunState) -> TextIOWrapper | None:
        try:
            return run_state.path.open(
                "x",
                buffering=EVENT_FILE_BUFFER_BYTES,
                encoding="utf-8",
            )
        except Exception:
            run_state.event_file_open_failures += 1
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_open_failure")
            return None

    def write_queued_event(
        self,
        queued_event: QueuedRequestEvent,
        file_handle: TextIOWrapper | None,
    ) -> TextIOWrapper | None:
        """Serialize and write one event; only the writer calls this."""
        run_state = queued_event.run_state
        if file_handle is None:
            return None
        else:
            pass
        try:
            serialized_event = json.dumps(
                queued_event.event.to_dict(),
                default=json_default,
            )
        except Exception:
            run_state.serialization_failures += 1
            run_state.incomplete_reasons.add("serialization_failure")
            return file_handle
        else:
            pass
        try:
            file_handle.write(serialized_event)
            file_handle.write("\n")
        except Exception:
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_write_failure")
            try:
                file_handle.close()
            except Exception:
                pass
            return None
        else:
            pass
        run_state.written_events += 1
        return file_handle

    def _flush_and_close(
        self,
        run_state: RecorderRunState,
        file_handle: TextIOWrapper | None,
    ) -> None:
        if file_handle is None:
            return
        else:
            pass
        try:
            file_handle.flush()
        except Exception:
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_flush_failure")
        else:
            pass
        try:
            file_handle.close()
        except Exception:
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_close_failure")
        else:
            pass

    def _publish_status(self, run_state: RecorderRunState) -> bool:
        run_state.unwritten_events = max(
            run_state.accepted_events - run_state.written_events, 0
        )
        if run_state.written_events + run_state.unwritten_events != (
            run_state.accepted_events
        ):
            run_state.incomplete_reasons.add("event_count_mismatch")
        else:
            pass
        if run_state.processed_events != run_state.accepted_events:
            run_state.incomplete_reasons.add("processing_count_mismatch")
        else:
            pass
        status = {
            "run_id": run_state.run_id,
            "pid": self.pid,
            "event_path": str(run_state.path),
            "stages": sorted(run_state.stages),
            "accepted_events": run_state.accepted_events,
            "processed_events": run_state.processed_events,
            "written_events": run_state.written_events,
            "unwritten_events": run_state.unwritten_events,
            "dropped_events": run_state.dropped_events,
            "metadata_snapshot_failures": run_state.metadata_snapshot_failures,
            "metadata_truncations": run_state.metadata_truncations,
            "serialization_failures": run_state.serialization_failures,
            "event_file_open_failures": run_state.event_file_open_failures,
            "write_failures": run_state.write_failures,
            "finalization_failures": run_state.finalization_failures,
            "incomplete_reasons": sorted(run_state.incomplete_reasons),
            "complete": not run_state.incomplete_reasons,
            "completeness_scope": "event_storage",
            "writer_scheduling": run_state.writer_scheduling,
        }
        temporary_path = run_state.status_path.with_name(
            f".{run_state.status_path.name}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(status, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary_path, run_state.status_path)
        except Exception:
            run_state.finalization_failures += 1
            run_state.incomplete_reasons.add("status_write_failure")
            self.finalization_failed = True
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        else:
            pass
        return True

    def _complete_state_unlocked(
        self, run_state: RecorderRunState, *, successful: bool
    ) -> None:
        if not successful:
            self.finalization_failed = True
        else:
            pass
        run_state.finalization.successful = successful
        run_state.finalization.completion_event.set()
        self.pending_states.pop(id(run_state), None)

    def _fail_state(self, run_state: RecorderRunState) -> None:
        self._wait_for_producer_quiescence(run_state)
        self._seal_deferred_buffers(run_state)
        self._aggregate_producer_buffers(run_state)
        run_state.finalization_failures += 1
        run_state.incomplete_reasons.add("writer_thread_failure")
        run_state.unwritten_events = max(
            run_state.accepted_events - run_state.written_events, 0
        )
        try:
            self._publish_status(run_state)
        except Exception:
            self.finalization_failed = True
            logger.warning(
                "RequestEventRecorder could not publish writer failure status for %s",
                run_state.run_id,
                exc_info=True,
            )
        with self._control_lock:
            if id(run_state) in self.pending_states:
                self._complete_state_unlocked(run_state, successful=False)
            else:
                pass

    def writer_should_exit(self) -> bool:
        """Return whether this writer has no retired work left."""
        with self._control_lock:
            should_exit = (
                self.active_state is None
                and not self.pending_states
                and self.writer_thread is threading.current_thread()
            )
        return should_exit

    def handle_writer_idle(
        self,
        open_state: RecorderRunState | None,
        file_handle: TextIOWrapper | None,
    ) -> tuple[RecorderRunState | None, TextIOWrapper | None]:
        """Compatibility shim for callers that used the old live-writer hook."""
        return open_state, file_handle

    def finalize_retired_states(self) -> None:
        """Compatibility helper used by administrative tests."""
        with self._control_lock:
            states = tuple(self.pending_states.values())
        for run_state in states:
            self._finalize_state(run_state)

    def fail_writer(
        self,
        open_state: RecorderRunState | None,
        file_handle: TextIOWrapper | None,
    ) -> None:
        """Mark all retired work incomplete after an unexpected writer error."""
        del open_state
        if file_handle is not None:
            with self._control_lock:
                states = tuple(self.pending_states.values())
            if states:
                self._flush_and_close(states[0], file_handle)
            else:
                pass
        else:
            pass
        self.writer_failed = True
        with self._control_lock:
            states = tuple(self.pending_states.values())
        for run_state in states:
            self._fail_state(run_state)
