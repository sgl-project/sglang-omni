# SPDX-License-Identifier: Apache-2.0
"""Asynchronous writer lifecycle for request-level profiling events."""

from __future__ import annotations

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
        self.pending_states: dict[str, RecorderRunState] = {}
        self.writer_thread: threading.Thread | None = None
        self.writer_failed = False
        # Sticky aggregate signal: individual handles remain authoritative for
        # each run, while shutdown must not forget a completed failed publish.
        self.finalization_failed = False
        self.pid = os.getpid()

    def is_active(self) -> bool:
        return self.active_state is not None

    def active_run_id(self) -> str | None:
        if self.active_state is None:
            return None
        else:
            pass
        with self.lock:
            active_state = self.active_state
            if active_state is None:
                return None
            else:
                pass
            return active_state.run_id

    def capture_active_run_timestamp_ns(self) -> tuple[str, int] | None:
        """Capture the active run ID and wall-clock timestamp together."""
        if self.active_state is None:
            return None
        else:
            pass
        with self.lock:
            active_state = self.active_state
            if active_state is None:
                return None
            else:
                pass
            return active_state.run_id, time.time_ns()

    def active_path(self) -> str | None:
        if self.active_state is None:
            return None
        else:
            pass
        with self.lock:
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
        """Capture and enqueue one event without waiting for the writer."""
        if self.active_state is None:
            return
        else:
            pass
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
                if (
                    self.active_state is run_state
                    and not run_state.finalization.completion_event.is_set()
                ):
                    run_state.metadata_snapshot_failures += 1
                    run_state.incomplete_reasons.add("metadata_snapshot_failure")
                else:
                    pass
            return
        else:
            pass

        if "__metadata_truncated__" in captured_metadata:
            metadata_truncated = True
        else:
            metadata_truncated = False
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
            if metadata_truncated:
                run_state.metadata_truncations += 1
                run_state.incomplete_reasons.add("metadata_truncated")
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
        retires the old run without waiting for its writer. Each recording is
        expected to use a unique run ID and a fresh event directory.
        """
        with self.lock:
            active_state = self.active_state
            if active_state is not None and active_state.run_id == run_id:
                active_state.stages.add(stage)
                active_state.active_stages.add(stage)
                return str(active_state.path)
            else:
                pass
            self.validate_start_unlocked()

        directory = Path(event_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)

        with self.lock:
            active_state = self.active_state
            if active_state is not None and active_state.run_id == run_id:
                active_state.stages.add(stage)
                active_state.active_stages.add(stage)
                return str(active_state.path)
            else:
                pass
            self.validate_start_unlocked()

            path = directory / f"events_{stage}_{self.pid}.jsonl"
            status_path = directory / f"run_status_{self.pid}.json"

            previous_state = self.retire_active_state_unlocked()
            if previous_state is not None:
                logger.warning(
                    f"RequestEventRecorder rotating from run_id={previous_state.run_id} "
                    f"to run_id={run_id}"
                )
            else:
                pass

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
            if self.writer_thread is None:
                self.writer_thread = threading.Thread(
                    target=self.writer_loop,
                    name=f"request-event-writer-{self.pid}",
                    daemon=True,
                )
                self.writer_thread.start()
            else:
                pass
            return str(path)

    def validate_start_unlocked(self) -> None:
        """Validate a new run; the caller must hold ``lock``."""
        if self.writer_failed and self.pending_states:
            raise RuntimeError("request event writer failed")
        else:
            pass
        if self.writer_failed:
            self.writer_failed = False
        else:
            pass
        if len(self.pending_states) >= MAX_PENDING_FINALIZATIONS:
            raise RuntimeError("too many profiler runs are waiting for finalization")
        else:
            pass

    def retire_active_state_unlocked(self) -> RecorderRunState | None:
        """Deactivate the current state; the caller must hold ``lock``."""
        run_state = self.active_state
        if run_state is None:
            return None
        else:
            pass
        self.active_state = None
        self.pending_states[run_state.run_id] = run_state
        return run_state

    def release_stage(self, stage: str) -> RequestEventFinalization | None:
        """Release one co-located stage without stopping its shared run."""
        with self.lock:
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
            retired_state = self.retire_active_state_unlocked()
            assert retired_state is not None
            return retired_state.finalization

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
            if not finalization.wait():
                return None
            else:
                pass
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
            completed = True
            for finalization in finalizations:
                if not finalization.wait():
                    completed = False
                else:
                    pass
            with self.lock:
                return completed and not self.finalization_failed
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
        with self.lock:
            return not self.finalization_failed and all(
                finalization.successful for finalization in finalizations
            )

    def writer_should_exit(self) -> bool:
        """Relinquish writer ownership after all accepted events are accounted."""
        with self.lock:
            should_exit = self.active_state is None and not self.pending_states
            if not should_exit:
                return False
            else:
                pass
            if self.writer_thread is threading.current_thread():
                self.writer_thread = None
                return True
            else:
                pass
            return False

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
                    if self.writer_should_exit():
                        return
                    else:
                        pass
                    continue

                try:
                    target_state = queued_event.run_state
                    if open_state is not target_state:
                        if open_state is not None:
                            if not self.finalize_run(open_state, file_handle):
                                raise RuntimeError(
                                    "request event writer switched runs before "
                                    "the previous run was eligible for finalization"
                                )
                            else:
                                pass
                        else:
                            pass
                        self.finalize_retired_states()
                        open_state = target_state
                        file_handle = self.open_event_file(open_state)
                    else:
                        pass
                    try:
                        file_handle = self.write_queued_event(queued_event, file_handle)
                    finally:
                        # Writer-owned accounting avoids producer lock contention.
                        queued_event.run_state.processed_events += 1
                finally:
                    self.event_queue.task_done()
        except Exception:
            logger.exception("RequestEventRecorder writer thread crashed")
            self.fail_writer(open_state, file_handle)
        finally:
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
        """Finalize only runs proven inactive and fully processed."""
        if open_state is not None and self.can_finalize_run(open_state):
            self.finalize_run(open_state, file_handle)
            open_state = None
            file_handle = None
        else:
            pass

        self.finalize_retired_states()
        return open_state, file_handle

    def finalize_retired_states(self) -> None:
        with self.lock:
            retired_states = list(self.pending_states.values())
        for run_state in retired_states:
            self.finalize_run(run_state, None)

    def open_event_file(self, run_state: RecorderRunState) -> TextIOWrapper | None:
        try:
            return run_state.path.open(
                "x",
                buffering=EVENT_FILE_BUFFER_BYTES,
                encoding="utf-8",
            )
        except Exception:
            with self.lock:
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
            serialized_event = json.dumps(queued_event.event.to_dict())
        except Exception:
            with self.lock:
                run_state.serialization_failures += 1
                run_state.incomplete_reasons.add("serialization_failure")
                first_failure = run_state.serialization_failures == 1
            if first_failure:
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
            with self.lock:
                run_state.write_failures += 1
                run_state.incomplete_reasons.add("event_file_write_failure")
                first_failure = run_state.write_failures == 1
            if first_failure:
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

    def can_finalize_run(self, run_state: RecorderRunState) -> bool:
        """Return whether a normal finalization has a synchronized drain proof."""
        with self.lock:
            return (
                run_state.run_id in self.pending_states
                and self.active_state is not run_state
                and run_state.accepted_events == run_state.processed_events
            )

    def fail_writer(
        self,
        open_state: RecorderRunState | None,
        file_handle: TextIOWrapper | None,
    ) -> None:
        """Terminally fail the writer and complete every affected handle."""
        states: dict[str, RecorderRunState] = {}
        with self.lock:
            self.writer_failed = True
            retired_state = self.retire_active_state_unlocked()
            if retired_state is not None:
                states[retired_state.run_id] = retired_state
            else:
                pass
            for run_state in self.pending_states.values():
                states[run_state.run_id] = run_state
            if open_state is not None:
                states[open_state.run_id] = open_state
            else:
                pass
            if self.writer_thread is threading.current_thread():
                self.writer_thread = None
            else:
                pass

        while True:
            try:
                queued_event = self.event_queue.get_nowait()
            except Empty:
                break
            else:
                states[queued_event.run_state.run_id] = queued_event.run_state
                self.event_queue.task_done()

        for run_state in states.values():
            with self.lock:
                run_state.incomplete_reasons.add("writer_thread_failure")
                run_state.unwritten_events = max(
                    run_state.accepted_events - run_state.written_events,
                    0,
                )
                if not run_state.finalization.completion_event.is_set():
                    self.pending_states[run_state.run_id] = run_state
                else:
                    pass

        for run_state in states.values():
            if run_state.finalization.completion_event.is_set():
                continue
            else:
                pass
            state_file = file_handle if run_state is open_state else None
            self.finalize_run(run_state, state_file, force=True)

    def finalize_run(
        self,
        run_state: RecorderRunState,
        file_handle: TextIOWrapper | None,
        *,
        force: bool = False,
    ) -> bool:
        """Flush, close, and atomically publish one retired run's status."""
        with self.lock:
            if run_state.run_id not in self.pending_states:
                return False
            else:
                pass
            if not force and (
                self.active_state is run_state
                or run_state.accepted_events != run_state.processed_events
            ):
                return False
            else:
                pass

        status_published = False
        try:
            if (
                file_handle is None
                and run_state.event_file_open_failures == 0
                and run_state.write_failures == 0
            ):
                file_handle = self.open_event_file(run_state)
            else:
                pass
            if file_handle is not None:
                try:
                    file_handle.flush()
                except Exception:
                    with self.lock:
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
                    with self.lock:
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

            with self.lock:
                run_state.unwritten_events = max(
                    run_state.accepted_events - run_state.written_events,
                    0,
                )
                if run_state.unwritten_events:
                    run_state.incomplete_reasons.add("event_count_mismatch")
                else:
                    pass
                if run_state.accepted_events != run_state.processed_events:
                    run_state.incomplete_reasons.add("processing_count_mismatch")
                else:
                    pass
                status = {
                    "run_id": run_state.run_id,
                    "pid": self.pid,
                    "event_path": str(run_state.path),
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
                    "stages": sorted(run_state.stages),
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
                status_published = True
            except Exception:
                with self.lock:
                    run_state.finalization_failures += 1
                    run_state.incomplete_reasons.add("status_write_failure")
                logger.warning(
                    f"RequestEventRecorder failed to publish {run_state.status_path}",
                    exc_info=True,
                )
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                pass
        except Exception:
            with self.lock:
                run_state.finalization_failures += 1
                run_state.incomplete_reasons.add("finalization_failure")
            logger.warning(
                f"RequestEventRecorder failed to finalize {run_state.path}",
                exc_info=True,
            )
        finally:
            with self.lock:
                if not status_published:
                    self.finalization_failed = True
                else:
                    pass
                self.pending_states.pop(run_state.run_id, None)
                run_state.finalization.successful = status_published
                run_state.finalization.completion_event.set()
            return status_published
