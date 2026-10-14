# SPDX-License-Identifier: Apache-2.0
"""Asynchronous writer lifecycle for request-level profiling events."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from io import TextIOWrapper
from pathlib import Path
from queue import Empty, Full, Queue

from sglang_omni.profiler.event_recorder_types import (
    DEFAULT_EVENT_QUEUE_CAPACITY,
    EVENT_FILE_BUFFER_BYTES,
    MAX_PENDING_FINALIZATIONS,
    WRITER_POLL_INTERVAL_SECONDS,
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
    """Process-local bounded event queue with one asynchronous writer thread."""

    def __init__(self, queue_capacity: int = DEFAULT_EVENT_QUEUE_CAPACITY) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be greater than zero")
        else:
            pass
        self.lock = threading.Lock()
        self.queue_capacity = queue_capacity
        self.event_queue: Queue[QueuedRequestEvent] = Queue(maxsize=queue_capacity)
        self.active_state: RecorderRunState | None = None
        self.pending_states: dict[int, RecorderRunState] = {}
        self.next_sequence_number = 0
        self.writer_thread: threading.Thread | None = None
        self.pid = os.getpid()

    def is_active(self) -> bool:
        with self.lock:
            return self.active_state is not None

    def active_run_id(self) -> str | None:
        with self.lock:
            if self.active_state is None:
                return None
            else:
                pass
            return self.active_state.run_id

    def capture_active_run_timestamp_ns(self) -> tuple[str, int] | None:
        """Capture the active run ID and wall-clock timestamp together."""
        with self.lock:
            if self.active_state is None:
                return None
            else:
                pass
            return self.active_state.run_id, time.time_ns()

    def active_path(self) -> str | None:
        with self.lock:
            if self.active_state is None:
                return None
            else:
                pass
            return str(self.active_state.path)

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
        """Capture and enqueue one event without waiting for the writer."""
        with self.lock:
            run_state = self.active_state
            if run_state is None:
                return
            else:
                pass
            if expected_run_id is not None and run_state.run_id != expected_run_id:
                return
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
            with self.lock:
                if self.active_state is run_state:
                    run_state.metadata_snapshot_failures += 1
                    run_state.incomplete_reasons.add("metadata_snapshot_failure")
                else:
                    pass
            return
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
        queued_event = QueuedRequestEvent(run_state=run_state, event=event)

        with self.lock:
            if self.active_state is not run_state:
                return
            else:
                pass
            if expected_run_id is not None and run_state.run_id != expected_run_id:
                return
            else:
                pass
            try:
                self.event_queue.put_nowait(queued_event)
            except Full:
                run_state.dropped_events += 1
                run_state.incomplete_reasons.add("queue_overflow")
            else:
                run_state.accepted_events += 1

    def start(self, run_id: str, event_dir: str, stage: str) -> str:
        """Start or join a per-process JSONL recording run.

        Co-located stages share one file per process. A different run ID
        retires the old run without waiting for its writer.
        """
        with self.lock:
            if self.active_state is not None and self.active_state.run_id == run_id:
                self.active_state.stages.add(stage)
                return str(self.active_state.path)
            else:
                pass
            if len(self.pending_states) >= MAX_PENDING_FINALIZATIONS:
                raise RuntimeError(
                    "too many profiler runs are waiting for finalization"
                )
            else:
                pass

        directory = Path(event_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"events_{stage}_{self.pid}.jsonl"
        run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
        status_path = directory / f"run_status_{run_digest}_{self.pid}.json"

        with self.lock:
            if self.active_state is not None and self.active_state.run_id == run_id:
                self.active_state.stages.add(stage)
                return str(self.active_state.path)
            else:
                pass
            if len(self.pending_states) >= MAX_PENDING_FINALIZATIONS:
                raise RuntimeError(
                    "too many profiler runs are waiting for finalization"
                )
            else:
                pass

            previous_state = self.retire_active_state_unlocked()
            if previous_state is not None:
                logger.warning(
                    f"RequestEventRecorder rotating from run_id={previous_state.run_id} "
                    f"to run_id={run_id}"
                )
            else:
                pass

            completion_event = threading.Event()
            finalization = RequestEventFinalization(
                path=str(path),
                status_path=str(status_path),
                completion_event=completion_event,
            )
            run_state = RecorderRunState(
                run_id=run_id,
                stage=stage,
                stages={stage},
                path=path,
                status_path=status_path,
                sequence_number=self.next_sequence_number,
                finalization=finalization,
            )
            self.next_sequence_number += 1
            self.active_state = run_state
            if self.writer_thread is None or not self.writer_thread.is_alive():
                self.writer_thread = threading.Thread(
                    target=self.writer_loop,
                    name=f"request-event-writer-{self.pid}",
                    daemon=True,
                )
                self.writer_thread.start()
            else:
                pass
            return str(path)

    def retire_active_state_unlocked(self) -> RecorderRunState | None:
        """Deactivate the current state; the caller must hold ``lock``."""
        run_state = self.active_state
        if run_state is None:
            return None
        else:
            pass
        self.active_state = None
        run_state.stop_requested.set()
        self.pending_states[run_state.sequence_number] = run_state
        return run_state

    def deactivate(
        self, *, run_id: str | None = None
    ) -> RequestEventFinalization | None:
        """Deactivate a run without waiting for file finalization."""
        with self.lock:
            active_state = self.active_state
            if active_state is None:
                return None
            else:
                pass
            if run_id is not None and active_state.run_id != run_id:
                logger.warning(
                    f"Ignoring RequestEventRecorder stop for run_id={run_id}; "
                    f"active run_id={active_state.run_id}"
                )
                return None
            else:
                pass
            retired_state = self.retire_active_state_unlocked()
            assert retired_state is not None
            return retired_state.finalization

    def stop(
        self,
        *,
        run_id: str | None = None,
        wait: bool = True,
    ) -> str | None:
        """Deactivate the active run and optionally wait for finalization."""
        finalization = self.deactivate(run_id=run_id)
        if finalization is None:
            return None
        else:
            pass
        if wait:
            finalization.wait()
        else:
            pass
        return finalization.path

    def wait_for_finalizations(self, timeout_seconds: float | None = None) -> bool:
        """Wait for all retired runs currently known to this recorder."""
        with self.lock:
            finalizations = [
                run_state.finalization for run_state in self.pending_states.values()
            ]
        if timeout_seconds is None:
            return all(finalization.wait() for finalization in finalizations)
        else:
            pass
        deadline = time.monotonic() + timeout_seconds
        for finalization in finalizations:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return False
            else:
                pass
            if not finalization.wait(remaining_seconds):
                return False
            else:
                pass
        return True

    def writer_loop(self) -> None:
        """Drain accepted events and finalize retired runs off the producer path."""
        open_state: RecorderRunState | None = None
        file_handle: TextIOWrapper | None = None
        try:
            while True:
                try:
                    queued_event = self.event_queue.get(
                        timeout=WRITER_POLL_INTERVAL_SECONDS
                    )
                except Empty:
                    open_state, file_handle = self.handle_writer_idle(
                        open_state, file_handle
                    )
                    with self.lock:
                        should_exit = (
                            self.active_state is None
                            and not self.pending_states
                            and self.event_queue.empty()
                            and open_state is None
                        )
                    if should_exit:
                        return
                    else:
                        pass
                    continue

                try:
                    target_state = queued_event.run_state
                    if open_state is not target_state:
                        if open_state is not None:
                            self.finalize_run(open_state, file_handle)
                        else:
                            pass
                        self.finalize_retired_states_before(
                            target_state.sequence_number
                        )
                        open_state = target_state
                        file_handle = self.open_event_file(open_state)
                    else:
                        pass
                    file_handle = self.write_queued_event(queued_event, file_handle)
                finally:
                    self.event_queue.task_done()
        except Exception:
            logger.exception("RequestEventRecorder writer thread crashed")
            if open_state is not None:
                open_state.write_failures += 1
                open_state.incomplete_reasons.add("writer_thread_failure")
            else:
                pass
        finally:
            if open_state is not None:
                self.finalize_run(open_state, file_handle)
            else:
                pass
            with self.lock:
                if self.writer_thread is threading.current_thread():
                    self.writer_thread = None
                else:
                    pass

    def handle_writer_idle(
        self,
        open_state: RecorderRunState | None,
        file_handle: TextIOWrapper | None,
    ) -> tuple[RecorderRunState | None, TextIOWrapper | None]:
        """Open active files and finalize retired runs when the queue is idle."""
        with self.lock:
            active_state = self.active_state
            queue_is_empty = self.event_queue.empty()
        if open_state is not None and open_state is not active_state:
            self.finalize_run(open_state, file_handle)
            open_state = None
            file_handle = None
        else:
            pass

        if queue_is_empty:
            self.finalize_retired_states_before(None)
            with self.lock:
                active_state = self.active_state
            if open_state is None and active_state is not None:
                open_state = active_state
                file_handle = self.open_event_file(open_state)
            else:
                pass
        else:
            pass
        return open_state, file_handle

    def finalize_retired_states_before(self, sequence_number: int | None) -> None:
        with self.lock:
            retired_states = sorted(
                self.pending_states.values(),
                key=lambda run_state: run_state.sequence_number,
            )
        for run_state in retired_states:
            if sequence_number is None or run_state.sequence_number < sequence_number:
                self.finalize_run(run_state, None)
            else:
                pass

    def open_event_file(self, run_state: RecorderRunState) -> TextIOWrapper | None:
        try:
            return run_state.path.open(
                "a",
                buffering=EVENT_FILE_BUFFER_BYTES,
                encoding="utf-8",
            )
        except Exception:
            run_state.event_file_open_failures += 1
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_open_failure")
            logger.warning(
                f"RequestEventRecorder failed to open {run_state.path}",
                exc_info=True,
            )
            return None

    def write_queued_event(
        self,
        queued_event: QueuedRequestEvent,
        file_handle: TextIOWrapper | None,
    ) -> TextIOWrapper | None:
        """Serialize and write one event; the caller is the writer thread."""
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
            if run_state.serialization_failures == 1:
                logger.warning(
                    f"RequestEventRecorder failed to serialize event "
                    f"{queued_event.event.event_name} for "
                    f"{queued_event.event.request_id}",
                    exc_info=True,
                )
            else:
                pass
            return file_handle
        else:
            pass
        try:
            file_handle.write(serialized_event)
            file_handle.write("\n")
        except Exception:
            run_state.write_failures += 1
            run_state.incomplete_reasons.add("event_file_write_failure")
            if run_state.write_failures == 1:
                logger.warning(
                    f"RequestEventRecorder failed to write event "
                    f"{queued_event.event.event_name} for "
                    f"{queued_event.event.request_id}",
                    exc_info=True,
                )
            else:
                pass
            try:
                file_handle.close()
            except Exception:
                logger.warning(
                    f"RequestEventRecorder failed to close broken file "
                    f"{run_state.path}",
                    exc_info=True,
                )
            return None
        else:
            pass
        run_state.written_events += 1
        return file_handle

    def finalize_run(
        self,
        run_state: RecorderRunState,
        file_handle: TextIOWrapper | None,
    ) -> None:
        """Flush, close, and account for one retired run on the writer thread."""
        with self.lock:
            if run_state.sequence_number not in self.pending_states:
                return
            else:
                pass

        if file_handle is None and run_state.event_file_open_failures == 0:
            file_handle = self.open_event_file(run_state)
        else:
            pass
        if file_handle is not None:
            try:
                file_handle.flush()
            except Exception:
                run_state.write_failures += 1
                run_state.incomplete_reasons.add("event_file_flush_failure")
                logger.warning(
                    f"RequestEventRecorder failed to flush {run_state.path}",
                    exc_info=True,
                )
            else:
                pass
            try:
                file_handle.close()
            except Exception:
                run_state.write_failures += 1
                run_state.incomplete_reasons.add("event_file_close_failure")
                logger.warning(
                    f"RequestEventRecorder failed to close {run_state.path}",
                    exc_info=True,
                )
            else:
                pass
        else:
            pass

        if run_state.written_events != run_state.accepted_events:
            run_state.incomplete_reasons.add("event_count_mismatch")
        else:
            pass
        status = {
            "run_id": run_state.run_id,
            "pid": self.pid,
            "event_path": str(run_state.path),
            "accepted_events": run_state.accepted_events,
            "written_events": run_state.written_events,
            "dropped_events": run_state.dropped_events,
            "metadata_snapshot_failures": run_state.metadata_snapshot_failures,
            "serialization_failures": run_state.serialization_failures,
            "event_file_open_failures": run_state.event_file_open_failures,
            "write_failures": run_state.write_failures,
            "finalization_failures": run_state.finalization_failures,
            "incomplete_reasons": sorted(run_state.incomplete_reasons),
            "complete": not run_state.incomplete_reasons,
            "completeness_scope": "event_storage",
            "stages": sorted(run_state.stages),
        }
        try:
            run_state.status_path.write_text(
                json.dumps(status, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            run_state.finalization_failures += 1
            run_state.incomplete_reasons.add("status_write_failure")
            logger.warning(
                f"RequestEventRecorder failed to write {run_state.status_path}",
                exc_info=True,
            )
        else:
            pass
        run_state.finalization.completion_event.set()
        with self.lock:
            self.pending_states.pop(run_state.sequence_number, None)
