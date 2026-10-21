# SPDX-License-Identifier: Apache-2.0
"""Multiprocess stage fixture with synthetic session hooks.

Linear topologies are reused across tests. A test that stops a worker starts its own.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import threading
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, TypedDict

import pytest

from sglang_omni.audio.vad import VADConfig
from sglang_omni.config.schema import (
    REPLICA_SEPARATOR,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.vad.session import VADHooks
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.pipeline.sessions import Session
from sglang_omni.pipeline.stage_workers import StageLaunchConfig
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    SessionOperationDict,
    SessionOperationKind,
    TimedChunk,
)
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.session import (
    ChunkEmitter,
    SessionContext,
    SessionHooks,
    SessionInbox,
    SessionScheduler,
)
from tests.unit_test.fixtures.session_outputs import control_chunk

REPLICA_COUNT = 2
EVENT_POLL_TIMEOUT_S = 1
OwnerEvent = tuple[Literal["open", "close", "finished", "cancelled"], str, str]
AppendEvent = tuple[Literal["append"], str, str, int]
ControlEvent = tuple[Literal["control"], str, str, str]
StageEvent = OwnerEvent | AppendEvent | ControlEvent


class SessionMetadata(TypedDict):
    omni_session: SessionOperationDict


@dataclass
class HookState:
    session_id: str
    count: int
    open_delay_s: float
    fail_open_stage: str | None
    emit_every: int
    cadence: int
    should_ignore_cancel: bool
    delay_s: float
    fail_close_once_stage: str | None
    emit_format: str | None
    emit_control: str | None
    # Seconds after an append completes at which the control is emitted on the session's output stream.
    emit_after_s: float
    reply_ms: float


def number_param(params: Mapping[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)


def integer_param(params: Mapping[str, object], key: str, default: int) -> int:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def text_param(params: Mapping[str, object], key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def read_hook_state(
    session_identity: SessionIdentity, request: OmniRequest
) -> HookState:
    params: dict[str, object] = {}
    for key, value in request.params.items():
        if not isinstance(key, str):
            raise TypeError("param names must be strings")
        params[key] = value
    should_ignore_cancel = params.get("ignore_cancel", False)
    if not isinstance(should_ignore_cancel, bool):
        raise TypeError("ignore_cancel must be a boolean")
    return HookState(
        session_id=session_identity.id,
        count=0,
        open_delay_s=number_param(params, "open_delay", 0),
        fail_open_stage=text_param(params, "fail_open"),
        emit_every=integer_param(params, "emit_every", 1),
        cadence=integer_param(params, "cadence", 1),
        should_ignore_cancel=should_ignore_cancel,
        delay_s=number_param(params, "delay", 0),
        fail_close_once_stage=text_param(params, "fail_close_once"),
        emit_format=text_param(params, "emit_format"),
        emit_control=text_param(params, "emit_control"),
        emit_after_s=number_param(params, "emit_after", 0),
        reply_ms=number_param(params, "emit_reply_ms", 0),
    )


REPLY_UNIT_MS = 100


def reply_chunks(reply_ms: float) -> list[TimedChunk]:
    """Turn 1 with `reply_ms` of 16 kHz pcm16 audio in 100 ms chunks, all at once."""
    units = int(reply_ms // REPLY_UNIT_MS)
    return [
        control_chunk("turn.started", turn=1),
        *(
            TimedChunk(
                "audio",
                index * REPLY_UNIT_MS,
                REPLY_UNIT_MS,
                0,
                bytes(REPLY_UNIT_MS * 32),
                "pcm16",
                turn=1,
            )
            for index in range(units)
        ),
        control_chunk("turn.done", units * REPLY_UNIT_MS, turn=1),
    ]


class Hooks(SessionHooks):
    def __init__(self, name: str, events: Queue[StageEvent]) -> None:
        self.name = name
        self.events = events
        self.states: dict[SessionIdentity, HookState] = {}
        self.emitters: dict[SessionIdentity, ChunkEmitter] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        state = read_hook_state(session_identity, request)
        self.events.put(("open", self.name, session_identity.id))
        time.sleep(state.open_delay_s)
        if state.fail_open_stage == self.name:
            raise RuntimeError("open failed")
        self.states[session_identity] = state

    def attach_output_stream(
        self, session_identity: SessionIdentity, emit: ChunkEmitter
    ) -> None:
        self.emitters[session_identity] = emit

    def append(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        import torch

        state = self.states[context.session_identity]
        self.events.put(("append", self.name, state.session_id, chunk.seq))
        state.count += 1
        logical_name = self.name.partition(REPLICA_SEPARATOR)[0]
        if logical_name == "source":
            payload.data = {"tensor": torch.tensor([state.count])}
        elif logical_name == "middle":
            assert payload.data["tensor"].item() == state.count
        else:
            if logical_name == "sink":
                assert payload.data["tensor"].item() == state.count
            else:
                pass
            if state.reply_ms and state.count == 1:
                for emitted in reply_chunks(state.reply_ms):
                    context.emit(emitted)
                payload.data = {"count": state.count}
                return payload
            if state.count % state.emit_every and not chunk.eos:
                payload.data = {"count": state.count}
                return payload
            if state.emit_after_s:
                late = control_chunk(
                    state.emit_control, chunk.t_start_ms, turn=state.count
                )
                threading.Timer(
                    state.emit_after_s,
                    self.emitters[context.session_identity],
                    args=(late,),
                ).start()
                payload.data = {"count": state.count}
                return payload
            for index in range(state.cadence):
                if state.should_ignore_cancel:
                    time.sleep(state.delay_s)
                    self.events.put(("finished", self.name, state.session_id))
                elif context.cancelled.wait(state.delay_s):
                    self.events.put(("cancelled", self.name, state.session_id))
                    break
                if state.emit_control is not None:
                    emitted = control_chunk(
                        state.emit_control,
                        chunk.t_start_ms,
                        seq=index,
                        turn=state.count,
                    )
                else:
                    emitted = TimedChunk(
                        "text",
                        chunk.t_start_ms,
                        0,
                        index,
                        (
                            {"count": state.count, "index": index}
                            if state.emit_format is None
                            else {
                                "count": state.count,
                                "index": index,
                                "stage": logical_name,
                                "input": (
                                    chunk.payload if chunk.modality == "text" else None
                                ),
                            }
                        ),
                        format=state.emit_format,
                        eos=chunk.eos and index == state.cadence - 1,
                    )
                context.emit(emitted)
            payload.data = {"count": state.count}
        return payload

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        state = self.states[session_identity]
        self.events.put(("control", self.name, state.session_id, event.payload["type"]))
        logical_name = self.name.partition(REPLICA_SEPARATOR)[0]
        if (
            state.reply_ms
            and logical_name == "sink"
            and event.payload["type"] == "interrupt"
        ):
            self.emitters[session_identity](
                control_chunk("turn.interrupted", turn=event.turn)
            )
        else:
            pass

    def close(self, session_identity: SessionIdentity) -> None:
        state = self.states.get(session_identity)
        assert state is not None
        self.events.put(("close", self.name, state.session_id))
        if state.fail_close_once_stage == self.name:
            state.fail_close_once_stage = None
            raise RuntimeError("close rejected")
        del self.states[session_identity]
        del self.emitters[session_identity]

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return ResourceUsage(bytes=self.states[session_identity].count)


def make_session_scheduler(name: str, events: Queue[StageEvent]) -> SessionScheduler:
    return SessionScheduler(Hooks(name, events))


class LoudnessDetector:
    """A speech detector for the VAD stage: speech is any non-zero byte, boundaries fall on chunk starts."""

    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self.is_speech = False
        self.last_speech_offset = 0
        self.samples = 0

    def boundaries(self, pcm: bytes) -> list[tuple[bool, int]]:
        loud = any(pcm)
        found = [] if loud == self.is_speech else [(loud, self.samples)]
        self.is_speech = loud
        self.samples += len(pcm) // 2
        if loud:
            self.last_speech_offset = self.samples
        else:
            pass
        return found

    def reset(self) -> None:
        self.is_speech = False
        self.last_speech_offset = 0
        self.samples = 0


def make_vad_scheduler(name: str, events: Queue[StageEvent]) -> SessionScheduler:
    return SessionScheduler(VADHooks(VADConfig(), LoudnessDetector))


VAD_FACTORY = f"{__name__}.make_vad_scheduler"


def worker(spec: StageLaunchConfig, ready: Event) -> None:
    from sglang_omni.pipeline.stage_workers import construct_stage

    async def run() -> None:
        stage = construct_stage(spec, logging.getLogger(__name__))
        await stage.start()
        ready.set()
        await stage.run()
        assert not stage.scheduler.open_sessions

    asyncio.run(run())


@asynccontextmanager
async def pipeline(
    tmp_path: Path,
    *,
    stage_count: int = 2,
    replicated: bool = False,
    replicate_entry: bool = False,
    list_next: bool = False,
    names: list[str] | None = None,
    factories: Mapping[str, str] | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    other_terminals: tuple[str, ...] = (),
) -> AsyncIterator[tuple[Coordinator, Queue[StageEvent], list[SpawnProcess]]]:
    """A linear route of synthetic stages, or with `names`, unchained terminal stages.

    `factories` replaces the synthetic hooks of the named stages with another
    factory taking `name`, `events` and `factory_kwargs`; `other_terminals`
    are terminal stages beside the route, as other models of the pipeline.
    """
    factories = factories or {}
    factory_kwargs = factory_kwargs or {}
    ctx = multiprocessing.get_context("spawn")
    is_chained = names is None
    names = names or (
        ["source", "middle", "sink"] if stage_count == 3 else ["source", "sink"]
    )
    stages = []
    for index, name in enumerate([*names, *other_terminals]):
        target = names[index + 1] if is_chained and index + 1 < len(names) else None
        stages.append(
            StageConfig(
                name=name,
                process=name,
                terminal=target is None,
                next=[target] if list_next and target else target,
                factory_path=factories.get(name, f"{__name__}.make_session_scheduler"),
            )
        )
    process_configs: dict[str, ProcessConfig] = {}
    if replicated:
        process_configs["sink"] = ProcessConfig(num_replicas=REPLICA_COUNT)
    if replicate_entry:
        process_configs["source"] = ProcessConfig(num_replicas=REPLICA_COUNT)
    config = PipelineConfig(
        model_path="mock",
        entry_stage=names[0],
        stages=stages,
        processes=process_configs,
    )
    plan, stages = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(stages, plan)
    endpoints = {stage.name: f"ipc://{tmp_path}/{stage.name}" for stage in expanded}
    completion, abort = f"ipc://{tmp_path}/done", f"ipc://{tmp_path}/abort"
    coordinator = Coordinator(
        completion,
        abort,
        names[0],
        [stage.name for stage in stages if stage.terminal],
        logical_process_plan=plan,
        replica_topology=topology,
        session_route=(lambda: names) if is_chained else None,
    )
    events = ctx.Queue()
    processes = []
    await coordinator.start()
    loop = asyncio.create_task(coordinator.run_completion_loop())
    try:
        for stage in expanded:
            ready = ctx.Event()
            spec = StageLaunchConfig(
                stage_name=stage.name,
                factory=stage.factory_path,
                factory_kwargs={"name": stage.name, "events": events, **factory_kwargs},
                next_stages=stage.next,
                is_terminal=stage.terminal,
                recv_endpoint=endpoints[stage.name],
                coordinator_endpoint=completion,
                abort_endpoint=abort,
                stage_endpoints=endpoints,
                replica_topology=topology.to_dict(),
            )
            process = ctx.Process(target=worker, args=(spec, ready))
            process.expected_exitcode = 0
            process.start()
            processes.append(process)
            assert await asyncio.to_thread(ready.wait, 30)
            coordinator.register_stage(stage.name, endpoints[stage.name])
        # Note (Junnan Li): ZMQ connects SUB sockets asynchronously; an abort published before that is dropped.
        await asyncio.sleep(0.1)
        yield coordinator, events, processes
    finally:
        await coordinator.shutdown_stages()
        await coordinator.stop()
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        for process in processes:
            await asyncio.to_thread(process.join, 10)
            if process.is_alive():
                process.kill()
                process.join()
            assert process.exitcode == process.expected_exitcode
        events.close()


def operation_metadata(
    operation: SessionOperationKind,
    session_identity: SessionIdentity,
    chunk: TimedChunk | None = None,
    *,
    should_preempt: bool = False,
) -> SessionMetadata:
    session_operation = SessionOperation(
        operation=operation,
        session_identity=session_identity,
        stages=("source",),
        chunk=chunk,
        should_preempt=should_preempt,
    )
    return {SESSION_METADATA_KEY: session_operation.to_dict()}


def chunk(seq: int, eos: bool = False) -> TimedChunk:
    return TimedChunk("audio", seq * 20, 20, seq, b"pcm", eos=eos)


PipelineResources: TypeAlias = (
    "tuple[Coordinator, Queue[StageEvent], list[SpawnProcess]]"
)


def event_log(events: Queue[StageEvent]) -> list[StageEvent]:
    stage_events: list[StageEvent] = []
    while not events.empty():
        stage_events.append(events.get(timeout=EVENT_POLL_TIMEOUT_S))
    return stage_events


class Condition(Protocol):
    def __call__(self) -> bool: ...


class RegisteredScheduler(Protocol):
    inbox: SessionInbox

    def compute(self, payload: StagePayload) -> StagePayload: ...


def block_session_cleanup(
    monkeypatch: pytest.MonkeyPatch, coordinator: Coordinator
) -> tuple[asyncio.Event, asyncio.Event]:
    entered, release = asyncio.Event(), asyncio.Event()
    cleanup_session = coordinator.cleanup_session

    async def blocked(session: Session) -> None:
        entered.set()
        await release.wait()
        await cleanup_session(session)

    monkeypatch.setattr(coordinator, "cleanup_session", blocked)
    return entered, release


def block_request_abort(
    monkeypatch: pytest.MonkeyPatch, coordinator: Coordinator
) -> tuple[asyncio.Event, asyncio.Event]:
    entered, release = asyncio.Event(), asyncio.Event()
    abort_request = coordinator.abort

    async def blocked(request_id: str) -> bool:
        entered.set()
        await release.wait()
        return await abort_request(request_id)

    monkeypatch.setattr(coordinator, "abort", blocked)
    return entered, release


async def wait_until(condition: Condition, timeout: float = 5) -> None:
    """Poll until condition() holds; asyncio.timeout needs Python 3.11."""

    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def compute_registered(
    scheduler: RegisteredScheduler, payload: StagePayload
) -> StagePayload:
    """Run one session operation on an unstarted scheduler through its inbox registration."""
    scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))
    message = scheduler.inbox.get_nowait()
    return scheduler.compute(message.data)
