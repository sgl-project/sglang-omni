# SPDX-License-Identifier: Apache-2.0
"""Process-local request event capture and post-stop writer."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from io import TextIOWrapper
from pathlib import Path

from sglang_omni.profiler.event_recorder_types import (
    DEFAULT_EVENT_BUFFER_CAPACITY,
    EVENT_FILE_BUFFER_BYTES,
    MAX_EVENT_TEXT_LENGTH,
    MAX_PRODUCER_BUFFERS,
    WRITER_POLL_INTERVAL_SECONDS,
    ProducerBuffer,
    ProducerDropCounter,
    RecorderRunState,
    RequestEvent,
    RequestEventFinalization,
    get_active_stage,
    json_default,
    snapshot_metadata,
)

logger = logging.getLogger(__name__)


class RequestEventRecorder:
    """Capture into bounded producer buffers and write only after stop."""

    def __init__(self, queue_capacity: int = DEFAULT_EVENT_BUFFER_CAPACITY) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be greater than zero")
        else:
            pass
        self.event_capacity = queue_capacity
        self.control_lock = threading.Lock()
        self.producer_local = threading.local()
        self.active_state: RecorderRunState | None = None
        self.draining_state: RecorderRunState | None = None
        self.writer_thread: threading.Thread | None = None
        self.pid = os.getpid()

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

    def begin_buffer_operation(self) -> ProducerBuffer | None:
        """Reserve one producer operation before stop can freeze its buffer."""
        observed_state = self.active_state
        if observed_state is None:
            return None
        else:
            pass

        producer_buffer = getattr(self.producer_local, "buffer", None)
        if producer_buffer is not None and producer_buffer.run_state is observed_state:
            producer_buffer.in_flight += 1
            if self.active_state is observed_state:
                return producer_buffer
            else:
                producer_buffer.in_flight -= 1
                return None
        else:
            pass

        producer_drop_counter = getattr(self.producer_local, "drop_counter", None)
        if (
            producer_drop_counter is not None
            and producer_drop_counter.run_state is observed_state
        ):
            producer_drop_counter.in_flight += 1
            if self.active_state is observed_state:
                producer_drop_counter.dropped_events += 1
            else:
                pass
            producer_drop_counter.in_flight -= 1
            return None
        else:
            pass

        with self.control_lock:
            if self.active_state is not observed_state:
                return None
            else:
                pass
            producer_buffer = getattr(self.producer_local, "buffer", None)
            if (
                producer_buffer is not None
                and producer_buffer.run_state is observed_state
            ):
                producer_buffer.in_flight += 1
                return producer_buffer
            else:
                pass
            if len(observed_state.producer_buffers) >= MAX_PRODUCER_BUFFERS:
                producer_drop_counter = ProducerDropCounter(
                    run_state=observed_state,
                    dropped_events=1,
                )
                observed_state.producer_drop_counters.append(producer_drop_counter)
                self.producer_local.drop_counter = producer_drop_counter
                observed_state.incomplete_reasons.add(
                    "producer_buffer_registration_limit"
                )
                return None
            else:
                pass
            producer_buffer = ProducerBuffer(
                run_state=observed_state,
                capacity=self.event_capacity,
                in_flight=1,
            )
            observed_state.producer_buffers.append(producer_buffer)
            self.producer_local.buffer = producer_buffer
            return producer_buffer

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
        """Append one event without waiting for storage or another producer."""
        try:
            producer_buffer = self.begin_buffer_operation()
        except Exception:
            return
        else:
            pass
        if producer_buffer is None:
            return
        else:
            pass

        run_state = producer_buffer.run_state
        try:
            if expected_run_id is not None and expected_run_id != run_state.run_id:
                return
            else:
                pass
            if (
                len(producer_buffer.events) + producer_buffer.pending_event_count
                >= producer_buffer.capacity
            ):
                producer_buffer.dropped_events += 1
                producer_buffer.incomplete_reasons.add("producer_buffer_overflow")
                return
            else:
                pass

            resolved_stage = stage or get_active_stage() or run_state.stage or "unknown"
            if not self.event_text_is_bounded(
                request_id, resolved_stage, event_name, run_state.run_id
            ):
                producer_buffer.dropped_events += 1
                producer_buffer.incomplete_reasons.add("event_payload_too_large")
                return
            else:
                pass
            try:
                captured_metadata = snapshot_metadata(metadata)
            except Exception:
                producer_buffer.metadata_snapshot_failures += 1
                producer_buffer.dropped_events += 1
                producer_buffer.incomplete_reasons.add("metadata_snapshot_failure")
                return
            else:
                pass
            if "__metadata_truncated__" in captured_metadata:
                producer_buffer.metadata_truncations += 1
                producer_buffer.incomplete_reasons.add("metadata_truncated")
            else:
                pass
            request_event = RequestEvent(
                request_id=request_id,
                stage=resolved_stage,
                event_name=event_name,
                timestamp_ns=(
                    timestamp_ns if timestamp_ns is not None else time.time_ns()
                ),
                run_id=run_state.run_id,
                pid=self.pid,
                metadata=captured_metadata,
            )
            producer_buffer.events.append(request_event)
            producer_buffer.accepted_events += 1
        except Exception:
            producer_buffer.dropped_events += 1
            producer_buffer.incomplete_reasons.add("producer_emit_failure")
        finally:
            producer_buffer.in_flight -= 1

    @staticmethod
    def event_text_is_bounded(
        request_id: str,
        stage: str,
        event_name: str,
        run_id: str,
    ) -> bool:
        return all(
            len(text) <= MAX_EVENT_TEXT_LENGTH
            for text in (request_id, stage, event_name, run_id)
        )

    def start(self, run_id: str, event_dir: str, stage: str) -> str:
        """Start or join one process-local run using a fresh output directory."""
        with self.control_lock:
            active_state = self.active_state
            if active_state is not None:
                if active_state.run_id != run_id:
                    raise RuntimeError(
                        f"request event run {active_state.run_id} is already active"
                    )
                else:
                    pass
                directory = Path(event_dir).expanduser().resolve()
                if active_state.path.parent != directory:
                    raise RuntimeError("co-located stages must use the same event_dir")
                else:
                    pass
                active_state.stages.add(stage)
                active_state.active_stages.add(stage)
                return str(active_state.path)
            else:
                pass

            if self.writer_thread is not None:
                raise RuntimeError(
                    "request event recorder is still draining the previous run"
                )
            else:
                pass
            if self.draining_state is not None:
                if not self.draining_state.finalization.completion_event.is_set():
                    raise RuntimeError(
                        "request event recorder is still draining the previous run"
                    )
                else:
                    self.draining_state = None
            else:
                pass

            directory = Path(event_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            directory = directory.resolve()
            event_path = directory / f"events_{stage}_{self.pid}.jsonl"
            status_path = directory / f"run_status_{self.pid}.json"
            if event_path.exists() or status_path.exists():
                raise FileExistsError(
                    f"request event output already exists in {directory}; use a fresh event_dir"
                )
            else:
                pass
            finalization = RequestEventFinalization(
                path=str(event_path),
                status_path=str(status_path),
                completion_event=threading.Event(),
            )
            self.active_state = RecorderRunState(
                run_id=run_id,
                stage=stage,
                stages={stage},
                active_stages={stage},
                path=event_path,
                status_path=status_path,
                finalization=finalization,
            )
            return str(event_path)

    def retire_active_state(self, run_state: RecorderRunState) -> None:
        """Deactivate one run and start its only post-stop writer."""
        self.active_state = None
        run_state.active_stages.clear()
        self.draining_state = run_state
        writer_thread = threading.Thread(
            target=self.writer_loop,
            args=(run_state,),
            name=f"request-event-writer-{self.pid}",
            daemon=True,
        )
        self.writer_thread = writer_thread
        try:
            writer_thread.start()
        except Exception:
            self.writer_thread = None
            run_state.finalization_failures += 1
            run_state.incomplete_reasons.add("writer_thread_start_failure")
            run_state.finalization.completion_event.set()
            logger.warning("RequestEventRecorder could not start its writer thread")
        else:
            pass

    def release_stage(self, stage: str) -> RequestEventFinalization | None:
        """Release one co-located stage and stop after the last local release."""
        with self.control_lock:
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
            self.retire_active_state(active_state)
            return active_state.finalization

    def deactivate(
        self, *, run_id: str | None = None
    ) -> RequestEventFinalization | None:
        """Deactivate the active run without waiting for its writer."""
        with self.control_lock:
            active_state = self.active_state
            if active_state is None:
                return None
            else:
                pass
            if run_id is not None and active_state.run_id != run_id:
                return None
            else:
                pass
            self.retire_active_state(active_state)
            return active_state.finalization

    def stop(
        self,
        *,
        run_id: str | None = None,
        wait: bool = False,
    ) -> str | None:
        """Deactivate the run; optional waiting is for tests and administration."""
        finalization = self.deactivate(run_id=run_id)
        if finalization is None:
            return None
        else:
            pass
        if wait and not finalization.wait():
            return None
        else:
            pass
        return finalization.path

    def wait_for_finalizations(
        self,
        timeout_seconds: float | None = None,
    ) -> bool:
        """Wait for the one draining run, if present."""
        with self.control_lock:
            if self.active_state is not None:
                return False
            else:
                pass
            draining_state = self.draining_state
        if draining_state is None:
            return True
        else:
            pass
        return draining_state.finalization.wait(timeout_seconds)

    def writer_loop(self, run_state: RecorderRunState) -> None:
        """Wait for reserved producers, then write and publish one run."""
        file_handle: TextIOWrapper | None = None
        status_published = False
        try:
            self.set_writer_scheduling(run_state)
            while any(
                producer_buffer.in_flight > 0
                for producer_buffer in run_state.producer_buffers
            ) or any(
                drop_counter.in_flight > 0
                for drop_counter in run_state.producer_drop_counters
            ):
                time.sleep(WRITER_POLL_INTERVAL_SECONDS)

            for drop_counter in run_state.producer_drop_counters:
                run_state.dropped_events += drop_counter.dropped_events

            for producer_buffer in run_state.producer_buffers:
                run_state.accepted_events += producer_buffer.accepted_events
                run_state.dropped_events += producer_buffer.dropped_events
                run_state.metadata_snapshot_failures += (
                    producer_buffer.metadata_snapshot_failures
                )
                run_state.metadata_truncations += producer_buffer.metadata_truncations
                run_state.incomplete_reasons.update(producer_buffer.incomplete_reasons)
                pending_event_count = sum(
                    len(pending_events)
                    for pending_events in producer_buffer.deferred_event_buffers
                )
                if pending_event_count:
                    producer_buffer.dropped_events += pending_event_count
                    run_state.dropped_events += pending_event_count
                    run_state.incomplete_reasons.add("deferred_buffer_unflushed")
                else:
                    pass
                for pending_events in producer_buffer.deferred_event_buffers:
                    pending_events.clear()
                producer_buffer.pending_event_count = 0

            file_handle = self.open_event_file(run_state)
            for producer_buffer in run_state.producer_buffers:
                for request_event in producer_buffer.events:
                    run_state.processed_events += 1
                    if file_handle is not None:
                        try:
                            self.write_event(request_event, file_handle, run_state)
                        except Exception:
                            run_state.finalization_failures += 1
                            run_state.incomplete_reasons.add("writer_failure")
                            file_handle = self.close_event_file(run_state, file_handle)
                        else:
                            pass
                        if file_handle is not None and file_handle.closed:
                            file_handle = None
                        else:
                            pass
                    else:
                        pass
                producer_buffer.events.clear()
            file_handle = self.close_event_file(run_state, file_handle)
            status_published = self.publish_status(run_state)
        except Exception:
            run_state.finalization_failures += 1
            run_state.incomplete_reasons.add("writer_failure")
            logger.exception(
                "RequestEventRecorder failed while finalizing %s", run_state.run_id
            )
            file_handle = self.close_event_file(run_state, file_handle)
            status_published = self.publish_status(run_state)
        finally:
            run_state.finalization.successful = status_published
            with self.control_lock:
                if self.writer_thread is threading.current_thread():
                    self.writer_thread = None
                else:
                    pass
            run_state.finalization.completion_event.set()

    def set_writer_scheduling(self, run_state: RecorderRunState) -> None:
        """Try SCHED_IDLE in the writer thread; record the actual outcome."""
        if hasattr(os, "sched_setscheduler") and hasattr(os, "SCHED_IDLE"):
            try:
                os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
            except OSError:
                run_state.writer_scheduling = "default"
                logger.info("RequestEventRecorder could not apply SCHED_IDLE")
            else:
                run_state.writer_scheduling = "SCHED_IDLE"
        else:
            run_state.writer_scheduling = "default"
            logger.info("RequestEventRecorder SCHED_IDLE is unsupported")

    def open_event_file(self, run_state: RecorderRunState) -> TextIOWrapper | None:
        try:
            return run_state.path.open(
                "x",
                buffering=EVENT_FILE_BUFFER_BYTES,
                encoding="utf-8",
            )
        except OSError:
            run_state.event_file_open_failures += 1
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_open_failure")
            return None

    def write_event(
        self,
        request_event: RequestEvent,
        file_handle: TextIOWrapper,
        run_state: RecorderRunState,
    ) -> None:
        try:
            serialized_event = json.dumps(
                request_event.to_dict(),
                default=json_default,
            )
        except (TypeError, ValueError):
            run_state.serialization_failures += 1
            run_state.incomplete_reasons.add("serialization_failure")
            return
        else:
            pass
        try:
            file_handle.write(serialized_event)
            file_handle.write("\n")
        except OSError:
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_write_failure")
            try:
                file_handle.close()
            except OSError:
                run_state.write_failures += 1
                run_state.incomplete_reasons.add("event_file_close_failure")
            else:
                pass
            return
        else:
            pass
        run_state.written_events += 1

    def close_event_file(
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
        except OSError:
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_flush_failure")
        else:
            pass
        try:
            file_handle.close()
        except OSError:
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_close_failure")
        else:
            pass

    def publish_status(self, run_state: RecorderRunState) -> bool:
        run_state.unwritten_events = max(
            run_state.accepted_events - run_state.written_events, 0
        )
        if run_state.processed_events != run_state.accepted_events:
            run_state.incomplete_reasons.add("processing_count_mismatch")
        else:
            pass
        if (
            run_state.written_events + run_state.unwritten_events
            != run_state.accepted_events
        ):
            run_state.incomplete_reasons.add("write_count_mismatch")
        else:
            pass
        if run_state.dropped_events:
            run_state.incomplete_reasons.add("events_dropped")
        else:
            pass
        status_payload = {
            "run_id": run_state.run_id,
            "pid": self.pid,
            "event_path": str(run_state.path),
            "stages": sorted(run_state.stages),
            "accepted_events": run_state.accepted_events,
            "producer_buffer_count": len(run_state.producer_buffers),
            "producer_buffer_event_counts": [
                producer_buffer.accepted_events + producer_buffer.dropped_events
                for producer_buffer in run_state.producer_buffers
            ],
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
                json.dumps(status_payload, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary_path, run_state.status_path)
        except OSError:
            run_state.finalization_failures += 1
            run_state.incomplete_reasons.add("status_write_failure")
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        else:
            pass
        return True
