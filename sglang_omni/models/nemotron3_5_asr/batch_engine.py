# SPDX-License-Identifier: Apache-2.0
"""One model owner for offline requests and persistent ASR sessions."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Literal

from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRPreparedChunk,
)
from sglang_omni.models.nemotron3_5_asr.request_builders import (
    Nemotron3_5ASRRequest,
    make_nemotron3_5_asr_request_builder,
    normalize_nemotron_language,
    validate_nemotron_greedy_params,
)
from sglang_omni.models.nemotron3_5_asr.streaming import (
    AppendResult,
    Nemotron3_5ASRStreamingChunkSpec,
    Nemotron3_5ASRStreamState,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ModelTask:
    kind: Literal["open", "append", "offline"]
    payload: StagePayload
    identity: SessionIdentity | None = None
    chunk: TimedChunk | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    future: Future[AppendResult | StagePayload | None] = field(default_factory=Future)
    ready_since: float = field(default_factory=time.monotonic)
    previous_text: str = ""
    is_first_output: bool = False


class NemotronBatchEngine:
    def __init__(
        self,
        runner: Nemotron3_5ASRModelRunner,
        *,
        max_batch_size: int,
        max_batch_wait_ms: float,
        max_pending_tasks: int,
        max_open_sessions: int,
        max_state_bytes: int,
        max_pcm_bytes: int,
        max_history_tokens: int,
        max_text_bytes: int,
    ) -> None:
        self.runner = runner
        self.spec = Nemotron3_5ASRStreamingChunkSpec(**runner.streaming_chunk_spec)
        self.build_request = make_nemotron3_5_asr_request_builder(
            prompt_dictionary=runner.prompt_dictionary
        )
        self.max_batch_size = max_batch_size
        self.max_batch_wait_s = max_batch_wait_ms / 1000
        self.max_pending_tasks = max_pending_tasks
        self.max_open_sessions = max_open_sessions
        self.max_state_bytes = max_state_bytes
        self.max_pcm_bytes = max_pcm_bytes
        self.max_history_tokens = max_history_tokens
        self.max_text_bytes = max_text_bytes
        self.session_reservation = (
            runner.streaming_state_budget_bytes
            + max_pcm_bytes + max_history_tokens * 96 + max_text_bytes * 12
        )
        self.condition = threading.Condition()
        self.commands: deque[ModelTask] = deque()
        self.tasks: dict[Future[AppendResult | StagePayload | None], ModelTask] = {}
        self.closes: dict[SessionIdentity, Future[None]] = {}
        self.states: dict[SessionIdentity, Nemotron3_5ASRStreamState] = {}
        self.ready: OrderedDict[SessionIdentity, ModelTask] = OrderedDict()
        self.offline: deque[ModelTask] = deque()
        self.usage_snapshots: dict[SessionIdentity, ResourceUsage] = {}
        self.is_stopping = False
        self.is_finalizing = False
        self.failure: BaseException | None = None
        self.last_kind: Literal["stream", "offline"] = "offline"
        self.thread = threading.Thread(target=self.run, name="nemotron-model")
        self.thread.start()

    def submit(self, task: ModelTask) -> Future[AppendResult | StagePayload | None]:
        with self.condition:
            if self.failure is not None:
                task.future.set_exception(RuntimeError(f"Nemotron engine failed: {self.failure}"))
            elif self.is_stopping:
                task.future.set_exception(RuntimeError("Nemotron engine is stopping"))
            elif len(self.tasks) >= self.max_pending_tasks:
                task.future.set_exception(RuntimeError("Nemotron accepted operation exceeds task budget"))
            else:
                self.tasks[task.future] = task
                self.commands.append(task)
                self.condition.notify()
        return task.future

    def open(self, identity: SessionIdentity, request: OmniRequest) -> Future[AppendResult | StagePayload | None]:
        return self.submit(ModelTask(
            kind="open", identity=identity,
            payload=StagePayload(request_id=identity.id, request=request, data=None),
        ))

    def append(self, identity: SessionIdentity, chunk: TimedChunk, payload: StagePayload,
               cancelled: threading.Event) -> Future[AppendResult | StagePayload | None]:
        return self.submit(ModelTask(kind="append", identity=identity, chunk=chunk,
                                     payload=payload, cancelled=cancelled))

    def submit_offline(self, payload: StagePayload, cancelled: threading.Event) -> Future[AppendResult | StagePayload | None]:
        return self.submit(ModelTask(kind="offline", payload=payload, cancelled=cancelled))

    def close(self, identity: SessionIdentity) -> Future[None]:
        with self.condition:
            if self.is_finalizing or self.failure is not None:
                result: Future[None] = Future()
                result.set_result(None)
                return result
            else:
                result = self.closes.setdefault(identity, Future())
                for task in self.tasks.values():
                    if task.identity == identity:
                        task.cancelled.set()
                    else:
                        pass
                self.condition.notify()
                return result

    def usage(self, identity: SessionIdentity) -> ResourceUsage:
        with self.condition:
            return self.usage_snapshots.get(identity, ResourceUsage())

    def begin_shutdown(self) -> None:
        with self.condition:
            self.is_stopping = True
            for task in self.tasks.values():
                task.cancelled.set()
                if not task.future.done():
                    task.future.set_exception(RuntimeError("Nemotron engine is stopping"))
                else:
                    pass
            self.condition.notify_all()

    def shutdown(self) -> None:
        self.begin_shutdown()
        with self.condition:
            self.is_finalizing = True
            self.condition.notify_all()
        self.thread.join()

    def finish(self, task: ModelTask, result: AppendResult | StagePayload | None = None,
               error: BaseException | None = None) -> None:
        with self.condition:
            self.tasks.pop(task.future, None)
            if task.future.done():
                return
            elif error is not None:
                task.future.set_exception(error)
            elif task.cancelled.is_set():
                task.future.set_exception(RuntimeError("Nemotron operation cancelled"))
            else:
                task.future.set_result(result)

    def release(self, identity: SessionIdentity, error: BaseException | None = None) -> None:
        task = self.ready.pop(identity, None)
        self.states.pop(identity, None)
        with self.condition:
            self.usage_snapshots.pop(identity, None)
        if task is not None:
            self.finish(task, error=error or RuntimeError("Nemotron session closed"))
        else:
            pass

    def accept(self, task: ModelTask) -> None:
        if task.cancelled.is_set() or task.future.done():
            self.finish(task)
            return
        elif task.kind == "offline":
            self.offline.append(task)
            return
        else:
            identity = task.identity
            assert identity is not None
        try:
            if task.kind == "open":
                if identity in self.states:
                    raise ValueError("Nemotron session already opened")
                elif len(self.states) >= self.max_open_sessions or (
                    (len(self.states) + 1) * self.session_reservation > self.max_state_bytes
                ):
                    raise RuntimeError("Nemotron session state reservation exhausted")
                else:
                    params = task.payload.request.params or {}
                    token_limit = validate_nemotron_greedy_params(params)
                    language = normalize_nemotron_language(params.get("language"), self.runner.prompt_dictionary)
                    state = Nemotron3_5ASRStreamState(
                        request_id=identity.id, payload=task.payload, language=language,
                        spec=self.spec, decode=self.runner.new_streaming_decode_state(),
                        max_new_tokens=token_limit,
                    )
                    self.states[identity] = state
                    self.publish_usage(identity, state)
                    self.finish(task)
            else:
                state = self.states[identity]
                chunk = task.chunk
                assert chunk is not None
                if identity in self.ready:
                    raise RuntimeError("concurrent append for one Nemotron session")
                else:
                    task.previous_text = state.clean_text
                    task.is_first_output = not state.clean_text
                    state.append_chunk(chunk, self.max_pcm_bytes)
                    task.ready_since = time.monotonic()
                    self.ready[identity] = task
                    self.complete_if_drained(identity, task, state)
        except Exception as exc:
            if task.kind != "open" or identity in self.states:
                self.release(identity, exc)
            else:
                pass
            self.finish(task, error=exc)

    def publish_usage(self, identity: SessionIdentity, state: Nemotron3_5ASRStreamState) -> None:
        with self.condition:
            self.usage_snapshots[identity] = state.usage(self.session_reservation)

    def complete_if_drained(self, identity: SessionIdentity, task: ModelTask,
                            state: Nemotron3_5ASRStreamState) -> None:
        if task.cancelled.is_set() or task.future.done():
            self.release(identity)
        elif state.has_ready_window():
            self.publish_usage(identity, state)
        else:
            state.trim_pcm()
            self.publish_usage(identity, state)
            self.ready.pop(identity, None)
            self.finish(task, state.append_result(task.payload, task.previous_text, task.is_first_output))

    def run_stream_batch(self) -> None:
        lanes: list[tuple[SessionIdentity, ModelTask, Nemotron3_5ASRStreamState]] = []
        chunks: list[Nemotron3_5ASRPreparedChunk] = []
        for identity, task in list(self.ready.items())[:self.max_batch_size]:
            state = self.states[identity]
            try:
                if task.cancelled.is_set() or task.future.done():
                    self.release(identity)
                    continue
                elif len(state.decode.tokens) + self.spec.subsequent_frames * 11 >= self.max_history_tokens:
                    raise RuntimeError("Nemotron token history budget exhausted")
                else:
                    window = state.pop_ready_window()
                    prepared = self.runner.prepare_streaming_chunk(
                        window.waveform, language=state.language, is_first=window.is_first)
                    lanes.append((identity, task, state))
                    chunks.append(prepared)
                    self.ready.move_to_end(identity)
            except Exception as exc:
                self.release(identity, exc)
        if not lanes:
            return
        else:
            pass
        try:
            result = self.runner.run_streaming_batch(
                [state.decode for _, _, state in lanes], chunks,
                requested_languages=[state.language for _, _, state in lanes],
                max_new_tokens=[state.max_new_tokens for _, _, state in lanes],
            )
        except Exception as exc:
            for identity, _, _ in lanes:
                self.release(identity, exc)
            return
        for index, (identity, task, state) in enumerate(lanes):
            try:
                if result.errors is not None and result.errors[index] is not None:
                    raise result.errors[index]
                else:
                    pass
                text = result.clean_texts[index]
                if not text.startswith(state.clean_text):
                    raise RuntimeError("Nemotron streaming transcript changed a previously emitted prefix")
                elif len(text.encode()) + len(result.raw_texts[index].encode()) > self.max_text_bytes:
                    raise RuntimeError("Nemotron transcript budget exhausted")
                else:
                    state.raw_text = result.raw_texts[index]
                    state.clean_text = text
                    state.detected_language = result.languages[index]
                    state.model_compute_s += result.elapsed_s / len(lanes)
                    task.ready_since = time.monotonic()
                    self.complete_if_drained(identity, task, state)
            except Exception as exc:
                self.release(identity, exc)

    def run_offline_batch(self) -> None:
        lanes: list[ModelTask] = []
        requests: list[Nemotron3_5ASRRequest] = []
        for _ in range(min(len(self.offline), self.max_batch_size)):
            task = self.offline.popleft()
            try:
                if task.cancelled.is_set() or task.future.done():
                    self.finish(task)
                else:
                    requests.append(self.build_request(task.payload))
                    lanes.append(task)
            except Exception as exc:
                self.finish(task, error=exc)
        if not lanes:
            return
        else:
            pass
        try:
            results = self.runner.run_batch(requests)
        except Exception as exc:
            for task in lanes:
                self.finish(task, error=exc)
        else:
            for task, result in zip(lanes, results, strict=True):
                self.finish(task, result)

    def run(self) -> None:
        try:
            while True:
                with self.condition:
                    closes = dict(self.closes)
                    self.closes.clear()
                    commands = list(self.commands)
                    self.commands.clear()
                    finalizing = self.is_finalizing
                    stopping = self.is_stopping
                for identity, future in closes.items():
                    self.release(identity)
                    future.set_result(None)
                for task in commands:
                    self.accept(task)
                if stopping:
                    for identity in list(self.states):
                        self.release(identity)
                    while self.offline:
                        self.finish(self.offline.popleft())
                else:
                    pass
                if finalizing:
                    break
                else:
                    pass
                for identity, task in list(self.ready.items()):
                    if task.cancelled.is_set() or task.future.done():
                        self.release(identity)
                    else:
                        pass
                streaming = bool(self.ready)
                offline = bool(self.offline)
                if streaming or offline:
                    kind = "stream" if streaming and (not offline or self.last_kind == "offline") else "offline"
                    tasks = list(self.ready.values()) if kind == "stream" else list(self.offline)
                    remaining = min(task.ready_since for task in tasks) + self.max_batch_wait_s - time.monotonic()
                    if len(tasks) >= self.max_batch_size or remaining <= 0:
                        self.last_kind = kind
                        if kind == "stream":
                            self.run_stream_batch()
                        else:
                            self.run_offline_batch()
                        continue
                    else:
                        timeout = remaining
                else:
                    timeout = None
                with self.condition:
                    if not self.commands and not self.closes and not self.is_finalizing:
                        self.condition.wait(timeout)
                    else:
                        pass
        except BaseException as exc:
            logger.exception("Nemotron model owner failed")
            with self.condition:
                self.failure = exc
        finally:
            with self.condition:
                self.is_finalizing = True
                for task in list(self.tasks.values()):
                    self.finish(task, error=RuntimeError(f"Nemotron engine stopped: {self.failure}"))
                for future in self.closes.values():
                    future.set_result(None)
                self.closes.clear()
                self.commands.clear()
                self.usage_snapshots.clear()
            self.ready.clear()
            self.offline.clear()
            self.states.clear()
            self.runner.close()
