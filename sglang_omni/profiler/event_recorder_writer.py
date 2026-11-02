# SPDX-License-Identifier: Apache-2.0
"""Process-local request event capture and writer."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Mapping
from io import TextIOWrapper
from pathlib import Path

from sglang_omni.profiler.event_recorder_types import (
    DEFAULT_EVENT_BUFFER_CAPACITY,
    DEFAULT_MAX_EVENT_FILE_BYTES,
    EVENT_FILE_BUFFER_BYTES,
    MAX_EVENT_TEXT_LENGTH,
    MAX_PRODUCER_BUFFERS,
    RUNTIME_WRITER_MAX_EVENTS_PER_ROUND,
    RUNTIME_WRITER_POLL_INTERVAL_SECONDS,
    WRITER_POLL_INTERVAL_SECONDS,
    ProducerBuffer,
    ProducerDropCounter,
    ProfileMode,
    RecorderRunState,
    RequestEvent,
    RequestEventFinalization,
    SamplingUnit,
    get_active_stage,
    json_default,
    snapshot_metadata,
)
from sglang_omni.proto.session import SESSION_METADATA_KEY

logger = logging.getLogger(__name__)


class RequestEventRecorder:
    """Capture into bounded producer buffers and write through one process-local
    writer.
    """

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

    def is_runtime_active(self) -> bool:
        active_state = self.active_state
        return active_state is not None and active_state.mode == "runtime"

    @staticmethod
    def session_identity_from_metadata(
        metadata: Mapping[str, object] | None,
    ) -> tuple[str, int] | None:
        if metadata is None:
            return None
        else:
            pass
        session_id = metadata.get("session_id")
        session_open_index = metadata.get("session_open_index")
        if isinstance(session_id, str) and type(session_open_index) is int:
            return session_id, session_open_index
        else:
            pass
        session_operation = metadata.get(SESSION_METADATA_KEY)
        if not isinstance(session_operation, Mapping):
            return None
        else:
            pass
        session_identity = session_operation.get("session_identity")
        if not isinstance(session_identity, Mapping):
            return None
        else:
            pass
        session_id = session_identity.get("id")
        session_open_index = session_identity.get("open_index")
        if isinstance(session_id, str) and type(session_open_index) is int:
            return session_id, session_open_index
        else:
            return None

    @staticmethod
    def deterministic_sampling_decision(
        run_state: RecorderRunState,
        sampling_identity: str | tuple[str, int],
    ) -> bool:
        assert run_state.sample_rate is not None
        assert run_state.sampling_unit is not None
        digest = hashlib.blake2b(digest_size=8, person=b"sgl-omni-prof")
        sampling_parts = [run_state.run_id, run_state.sampling_unit]
        if isinstance(sampling_identity, str):
            sampling_parts.append(sampling_identity)
        else:
            sampling_parts.extend((sampling_identity[0], str(sampling_identity[1])))
        for sampling_part in sampling_parts:
            encoded_part = sampling_part.encode("utf-8")
            digest.update(len(encoded_part).to_bytes(8, "big"))
            digest.update(encoded_part)
        else:
            pass
        sampling_value = int.from_bytes(digest.digest(), "big")
        sampling_limit = int(run_state.sample_rate * (1 << 64))
        return sampling_value < sampling_limit

    def session_sampling_decision(
        self,
        run_state: RecorderRunState,
        session_identity: tuple[str, int],
    ) -> bool:
        cached_decision = run_state.session_sampling_decisions.get(session_identity)
        if cached_decision is not None:
            return cached_decision
        else:
            pass
        decision = self.deterministic_sampling_decision(run_state, session_identity)
        return run_state.session_sampling_decisions.setdefault(
            session_identity, decision
        )

    def session_sampling_identity_for_request(
        self, request_id: str
    ) -> tuple[str, int] | None:
        """Return a selected request's cached session identity for stream transport."""
        run_state = self.active_state
        if (
            run_state is None
            or run_state.mode != "runtime"
            or run_state.sampling_unit != "session"
            or run_state.request_sampling_decisions.get(request_id) is not True
        ):
            return None
        else:
            pass
        return run_state.request_session_identities.get(request_id)

    def register_session_sampling_context(
        self,
        request_id: str,
        session_identity: tuple[str, int],
        *,
        stage_name: str,
    ) -> None:
        """Cache transported session sampling context before a payload arrives."""
        run_state = self.active_state
        if (
            run_state is None
            or run_state.mode != "runtime"
            or run_state.sampling_unit != "session"
        ):
            return
        else:
            pass
        has_registered_session_sampling_context = (
            request_id in run_state.request_sampling_decisions
            and request_id in run_state.request_session_identities
        )
        request_stage_names = run_state.request_sampling_stage_names.get(request_id)
        if (
            has_registered_session_sampling_context
            and request_stage_names is not None
            and stage_name in request_stage_names
        ):
            return
        else:
            pass
        with self.control_lock:
            if (
                self.active_state is not run_state
                or run_state.mode != "runtime"
                or run_state.sampling_unit != "session"
            ):
                return
            else:
                pass
            has_registered_session_sampling_context = (
                request_id in run_state.request_sampling_decisions
                and request_id in run_state.request_session_identities
            )
            if has_registered_session_sampling_context:
                run_state.request_sampling_stage_names.setdefault(
                    request_id, set()
                ).add(stage_name)
                return
            else:
                pass
            run_state.request_session_identities[request_id] = session_identity
            run_state.request_sampling_decisions[request_id] = (
                self.session_sampling_decision(run_state, session_identity)
            )
            run_state.request_sampling_stage_names.setdefault(request_id, set()).add(
                stage_name
            )

    def record_request_admission(
        self,
        request_id: str,
        request_metadata: Mapping[str, object] | None,
        *,
        stage_name: str,
    ) -> None:
        with self.control_lock:
            run_state = self.active_state
            if run_state is None or run_state.mode != "runtime":
                return
            else:
                pass
            session_identity = None
            if run_state.sampling_unit == "session":
                session_identity = self.session_identity_from_metadata(request_metadata)
                if session_identity is not None:
                    run_state.request_session_identities.setdefault(
                        request_id, session_identity
                    )
                    session_operation = (
                        request_metadata.get(SESSION_METADATA_KEY)
                        if request_metadata is not None
                        else None
                    )
                    if isinstance(session_operation, Mapping):
                        operation = session_operation.get("operation")
                        if isinstance(operation, str):
                            run_state.request_session_operations[request_id] = operation
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            else:
                pass
            decision = run_state.request_sampling_decisions.get(request_id)
            if decision is None:
                if run_state.sampling_unit == "request":
                    decision = self.deterministic_sampling_decision(
                        run_state, request_id
                    )
                else:
                    if session_identity is None:
                        decision = False
                    else:
                        decision = self.session_sampling_decision(
                            run_state, session_identity
                        )
            else:
                pass
            run_state.request_sampling_decisions[request_id] = decision
            run_state.request_sampling_stage_names.setdefault(request_id, set()).add(
                stage_name
            )

    def should_sample_request(
        self,
        run_state: RecorderRunState,
        request_id: str,
        *,
        event_metadata: Mapping[str, object] | None = None,
        request_metadata: Mapping[str, object] | None = None,
        sampling_session_identity: tuple[str, int] | None = None,
    ) -> bool:
        if run_state.mode != "runtime":
            return True
        else:
            pass
        cached_decision = run_state.request_sampling_decisions.get(request_id)
        if cached_decision is not None:
            return cached_decision
        elif run_state.sampling_unit == "request":
            return self.deterministic_sampling_decision(run_state, request_id)
        else:
            pass
        session_identity = sampling_session_identity
        if session_identity is None:
            session_identity = self.session_identity_from_metadata(event_metadata)
        else:
            pass
        if session_identity is None:
            session_identity = self.session_identity_from_metadata(request_metadata)
        else:
            pass
        if session_identity is None:
            return False
        else:
            return self.session_sampling_decision(run_state, session_identity)

    def clear_request_sampling(
        self,
        request_id: str,
        *,
        stage_name: str,
        clear_session: bool = False,
    ) -> None:
        with self.control_lock:
            run_state = self.active_state
            if run_state is None or run_state.mode != "runtime":
                return
            else:
                pass
            stage_names = run_state.request_sampling_stage_names.get(request_id)
            if stage_names is not None:
                stage_names.discard(stage_name)
                if stage_names:
                    return
                else:
                    run_state.request_sampling_stage_names.pop(request_id, None)
            else:
                pass
            run_state.request_sampling_decisions.pop(request_id, None)
            session_identity = run_state.request_session_identities.pop(
                request_id, None
            )
            session_operation = run_state.request_session_operations.pop(
                request_id, None
            )
            if session_identity is not None:
                if clear_session or session_operation in ("close", "read"):
                    self.clear_session_sampling_if_idle(run_state, session_identity)
                elif session_identity in run_state.session_sampling_clear_when_idle:
                    self.clear_session_sampling_if_idle(run_state, session_identity)
                else:
                    pass
            else:
                pass

    def clear_session_sampling(self, session_id: str, session_open_index: int) -> None:
        with self.control_lock:
            run_state = self.active_state
            if run_state is None or run_state.mode != "runtime":
                return
            else:
                pass
            self.clear_session_sampling_if_idle(
                run_state, (session_id, session_open_index)
            )

    @staticmethod
    def clear_session_sampling_if_idle(
        run_state: RecorderRunState,
        session_identity: tuple[str, int],
    ) -> None:
        if session_identity in run_state.request_session_identities.values():
            run_state.session_sampling_clear_when_idle.add(session_identity)
        else:
            run_state.session_sampling_decisions.pop(session_identity, None)
            run_state.session_sampling_clear_when_idle.discard(session_identity)

    def begin_buffer_operation(
        self,
        *,
        request_id: str | None = None,
        event_metadata: Mapping[str, object] | None = None,
        expected_run_id: str | None = None,
        expected_run_state: RecorderRunState | None = None,
        sampling_session_identity: tuple[str, int] | None = None,
    ) -> ProducerBuffer | None:
        """Reserve one producer operation before stop can freeze its buffer."""
        observed_state = self.active_state
        if observed_state is None:
            return None
        else:
            pass
        if observed_state.mode == "runtime":
            if (
                expected_run_state is not None
                and observed_state is not expected_run_state
            ):
                return None
            elif (
                expected_run_id is not None and expected_run_id != observed_state.run_id
            ):
                return None
            else:
                pass
            if observed_state.file_size_limit_reached:
                return None
            else:
                pass
            if request_id is not None and not self.should_sample_request(
                observed_state,
                request_id,
                event_metadata=event_metadata,
                sampling_session_identity=sampling_session_identity,
            ):
                return None
            else:
                pass
            if observed_state.file_size_limit_reached:
                return None
            else:
                pass
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
                capacity=observed_state.event_buffer_capacity,
                in_flight=1,
                events=([] if observed_state.mode == "performance" else deque()),
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
        sampling_session_identity: tuple[str, int] | None = None,
    ) -> None:
        """Append one event without waiting for storage or another producer."""
        try:
            producer_buffer = self.begin_buffer_operation(
                request_id=request_id,
                event_metadata=metadata,
                expected_run_id=expected_run_id,
                sampling_session_identity=sampling_session_identity,
            )
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
            if run_state.mode == "runtime" and run_state.file_size_limit_reached:
                return
            else:
                pass
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
            if run_state.mode == "runtime" and run_state.file_size_limit_reached:
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
            if run_state.mode == "runtime" and run_state.file_size_limit_reached:
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

    def start(
        self,
        run_id: str,
        event_dir: str,
        stage: str,
        *,
        event_buffer_capacity: int | None = None,
        mode: ProfileMode = "performance",
        sampling_unit: SamplingUnit | None = None,
        sample_rate: float | None = None,
        max_event_file_bytes: int | None = None,
    ) -> str:
        """Start or join one process-local run using a fresh output directory."""
        if event_buffer_capacity is not None and (
            type(event_buffer_capacity) is not int or event_buffer_capacity <= 0
        ):
            raise ValueError("event_buffer_capacity must be a positive integer")
        else:
            pass
        if mode not in ("performance", "runtime"):
            raise ValueError("mode must be 'performance' or 'runtime'")
        elif mode == "runtime":
            if sampling_unit not in ("request", "session"):
                raise ValueError("runtime mode requires a sampling_unit")
            elif (
                sample_rate is None
                or type(sample_rate) not in (int, float)
                or not math.isfinite(sample_rate)
                or not (0 < sample_rate <= 1)
            ):
                raise ValueError("runtime mode requires 0 < sample_rate <= 1")
            elif max_event_file_bytes is not None and (
                type(max_event_file_bytes) is not int or max_event_file_bytes <= 0
            ):
                raise ValueError("max_event_file_bytes must be a positive integer")
            else:
                if max_event_file_bytes is None:
                    max_event_file_bytes = DEFAULT_MAX_EVENT_FILE_BYTES
                else:
                    pass
        elif (
            sampling_unit is not None
            or sample_rate is not None
            or max_event_file_bytes is not None
        ):
            raise ValueError("runtime sampling and file limits require runtime mode")
        else:
            pass
        with self.control_lock:
            active_state = self.active_state
            if active_state is not None:
                if active_state.run_id != run_id:
                    raise RuntimeError(
                        f"request event run {active_state.run_id} is already active"
                    )
                else:
                    pass
                if (
                    event_buffer_capacity is not None
                    and event_buffer_capacity != active_state.event_buffer_capacity
                ):
                    raise RuntimeError(
                        "co-located stages must use the same event_buffer_capacity"
                    )
                else:
                    pass
                if (
                    mode != active_state.mode
                    or sampling_unit != active_state.sampling_unit
                    or sample_rate != active_state.sample_rate
                    or max_event_file_bytes != active_state.max_event_file_bytes
                ):
                    raise RuntimeError(
                        "co-located stages must use the same request event "
                        "profile configuration"
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
                    f"request event output already exists in {directory}; "
                    "use a fresh event_dir"
                )
            else:
                pass
            finalization = RequestEventFinalization(
                path=str(event_path),
                status_path=str(status_path),
                completion_event=threading.Event(),
            )
            run_state = RecorderRunState(
                run_id=run_id,
                stage=stage,
                stages={stage},
                active_stages={stage},
                path=event_path,
                status_path=status_path,
                finalization=finalization,
                event_buffer_capacity=(
                    self.event_capacity
                    if event_buffer_capacity is None
                    else event_buffer_capacity
                ),
                mode=mode,
                sampling_unit=sampling_unit,
                sample_rate=sample_rate,
                max_event_file_bytes=max_event_file_bytes,
            )
            self.active_state = run_state
            if mode == "runtime":
                writer_thread = threading.Thread(
                    target=self.writer_loop,
                    args=(run_state,),
                    name=f"request-event-writer-{self.pid}",
                    daemon=True,
                )
                self.writer_thread = writer_thread
                try:
                    writer_thread.start()
                except Exception as exc:
                    self.active_state = None
                    self.writer_thread = None
                    raise RuntimeError(
                        "request event runtime writer could not start"
                    ) from exc
                else:
                    pass
            else:
                pass
            return str(event_path)

    def retire_active_state(self, run_state: RecorderRunState) -> None:
        """Deactivate one run and notify its process-local writer."""
        self.active_state = None
        run_state.active_stages.clear()
        self.draining_state = run_state
        if run_state.mode == "runtime":
            run_state.stop_requested.set()
            return
        else:
            pass
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
            if run_state.mode == "runtime":
                try:
                    self.set_writer_scheduling(run_state)
                    file_handle = self.open_event_file(run_state)
                    while not run_state.stop_requested.is_set():
                        self.drain_runtime_buffers(run_state, file_handle)
                        run_state.stop_requested.wait(
                            RUNTIME_WRITER_POLL_INTERVAL_SECONDS
                        )
                except Exception:
                    run_state.writer_failed = True
                    run_state.finalization_failures += 1
                    run_state.incomplete_reasons.add("writer_failure")
                    logger.exception(
                        "RequestEventRecorder runtime writer failed for %s",
                        run_state.run_id,
                    )
                    run_state.stop_requested.wait()
                else:
                    pass
                while any(
                    producer_buffer.in_flight > 0
                    for producer_buffer in run_state.producer_buffers
                ) or any(
                    drop_counter.in_flight > 0
                    for drop_counter in run_state.producer_drop_counters
                ):
                    time.sleep(WRITER_POLL_INTERVAL_SECONDS)
                else:
                    pass
                self.collect_runtime_status(run_state)
                while self.drain_runtime_buffers(
                    run_state,
                    None if run_state.writer_failed else file_handle,
                ):
                    if any(
                        producer_buffer.events
                        for producer_buffer in run_state.producer_buffers
                    ):
                        time.sleep(RUNTIME_WRITER_POLL_INTERVAL_SECONDS)
                    else:
                        pass
                else:
                    pass
                file_handle = self.close_event_file(run_state, file_handle)
                status_published = self.publish_status(run_state)
            else:
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
                    run_state.metadata_truncations += (
                        producer_buffer.metadata_truncations
                    )
                    run_state.incomplete_reasons.update(
                        producer_buffer.incomplete_reasons
                    )
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
                                file_handle = self.close_event_file(
                                    run_state, file_handle
                                )
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
                run_state.request_sampling_decisions.clear()
                run_state.request_sampling_stage_names.clear()
                run_state.request_session_identities.clear()
                run_state.request_session_operations.clear()
                run_state.session_sampling_decisions.clear()
            run_state.finalization.completion_event.set()

    def collect_runtime_status(self, run_state: RecorderRunState) -> None:
        for drop_counter in run_state.producer_drop_counters:
            run_state.dropped_events += drop_counter.dropped_events
        else:
            pass
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
        else:
            pass

    def drain_runtime_buffers(
        self,
        run_state: RecorderRunState,
        file_handle: TextIOWrapper | None,
    ) -> int:
        producer_buffers = tuple(run_state.producer_buffers)
        producer_buffer_count = len(producer_buffers)
        if producer_buffer_count == 0:
            return 0
        else:
            pass
        start_index = run_state.writer_buffer_index % producer_buffer_count
        ordered_buffers = (
            producer_buffers[start_index:] + producer_buffers[:start_index]
        )
        active_buffers = [
            producer_buffer
            for producer_buffer in ordered_buffers
            if producer_buffer.events
        ]
        run_state.writer_buffer_index = (start_index + 1) % producer_buffer_count
        processed_event_count = 0
        while (
            active_buffers
            and processed_event_count < RUNTIME_WRITER_MAX_EVENTS_PER_ROUND
        ):
            remaining_buffers: list[ProducerBuffer] = []
            for producer_buffer in active_buffers:
                try:
                    request_event = producer_buffer.events.popleft()
                except IndexError:
                    pass
                else:
                    run_state.processed_events += 1
                    processed_event_count += 1
                    if (
                        file_handle is not None
                        and not file_handle.closed
                        and not run_state.writer_failed
                    ):
                        self.write_event(request_event, file_handle, run_state)
                        if file_handle.closed:
                            run_state.writer_failed = True
                        else:
                            pass
                    else:
                        pass
                    if producer_buffer.events:
                        remaining_buffers.append(producer_buffer)
                    else:
                        pass
                if processed_event_count >= RUNTIME_WRITER_MAX_EVENTS_PER_ROUND:
                    break
                else:
                    pass
            active_buffers = remaining_buffers
        else:
            pass
        return processed_event_count

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
        if run_state.mode == "runtime" and run_state.file_size_limit_reached:
            return
        else:
            pass
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
        if run_state.mode == "runtime":
            serialized_record = f"{serialized_event}\n"
            serialized_record_bytes = len(serialized_record.encode("utf-8"))
            assert run_state.max_event_file_bytes is not None
            if (
                run_state.event_file_bytes_written + serialized_record_bytes
                > run_state.max_event_file_bytes
            ):
                run_state.file_size_limit_reached = True
                run_state.incomplete_reasons.add("file_size_limit_reached")
                return
            else:
                pass
            try:
                file_handle.write(serialized_record)
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
            run_state.event_file_bytes_written += serialized_record_bytes
            run_state.written_events += 1
        else:
            pass
        if run_state.mode == "runtime":
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
            "event_buffer_capacity": run_state.event_buffer_capacity,
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
        if run_state.mode == "runtime":
            status_payload.update(
                {
                    "mode": run_state.mode,
                    "sampling_unit": run_state.sampling_unit,
                    "sample_rate": run_state.sample_rate,
                    "completeness_scope": "sampled_event_storage",
                    "max_event_file_bytes": run_state.max_event_file_bytes,
                    "event_file_bytes_written": run_state.event_file_bytes_written,
                }
            )
        else:
            pass
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
