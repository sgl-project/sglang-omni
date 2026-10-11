# SPDX-License-Identifier: Apache-2.0
"""Multiprocess stage fixture with synthetic session hooks.

Linear topologies are reused across tests. A test that stops a worker starts its own.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import pytest

from sglang_omni.config.schema import (
    REPLICA_SEPARATOR,
    PipelineConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.config.topology import compile_logical_processes
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
    TimedChunk,
)
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionInbox,
    SessionScheduler,
)

REPLICA_COUNT = 2
EVENT_POLL_TIMEOUT_S = 1
OwnerEvent = tuple[Literal["open", "close", "finished", "cancelled"], str, str]
AppendEvent = tuple[Literal["append"], str, str, int]
StageEvent = OwnerEvent | AppendEvent


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
    )


class Hooks(SessionHooks):
    def __init__(self, name: str, events: Queue[StageEvent]) -> None:
        self.name = name
        self.events = events
        self.states: dict[SessionIdentity, HookState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        state = read_hook_state(session_identity, request)
        self.events.put(("open", self.name, session_identity.id))
        time.sleep(state.open_delay_s)
        if state.fail_open_stage == self.name:
            raise RuntimeError("open failed")
        self.states[session_identity] = state

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
            assert payload.data["tensor"].item() == state.count
            if state.count % state.emit_every and not chunk.eos:
                payload.data = {"count": state.count}
                return payload
            for index in range(state.cadence):
                if state.should_ignore_cancel:
                    time.sleep(state.delay_s)
                    self.events.put(("finished", self.name, state.session_id))
                elif context.cancelled.wait(state.delay_s):
                    self.events.put(("cancelled", self.name, state.session_id))
                    break
                context.emit(
                    TimedChunk(
                        "text",
                        chunk.t_start_ms,
                        0,
                        index,
                        {"count": state.count, "index": index},
                        eos=chunk.eos and index == state.cadence - 1,
                    )
                )
            payload.data = {"count": state.count}
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        state = self.states.get(session_identity)
        assert state is not None
        self.events.put(("close", self.name, state.session_id))
        if state.fail_close_once_stage == self.name:
            state.fail_close_once_stage = None
            raise RuntimeError("close rejected")
        del self.states[session_identity]

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return ResourceUsage(bytes=self.states[session_identity].count)


def make_session_scheduler(name: str, events: Queue[StageEvent]) -> SessionScheduler:
    return SessionScheduler(Hooks(name, events))


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
) -> AsyncIterator[tuple[Coordinator, Queue[StageEvent], list[SpawnProcess]]]:
    ctx = multiprocessing.get_context("spawn")
    names = ["source", "middle", "sink"] if stage_count == 3 else ["source", "sink"]
    stages = []
    for index, name in enumerate(names):
        target = names[index + 1] if index + 1 < len(names) else None
        stages.append(
            StageConfig(
                name=name,
                process=name,
                terminal=target is None,
                next=[target] if list_next and target else target,
                factory_path=f"{__name__}.make_session_scheduler",
            )
        )
    process_configs: dict[str, ProcessConfig] = {}
    if replicated:
        process_configs["sink"] = ProcessConfig(num_replicas=REPLICA_COUNT)
    if replicate_entry:
        process_configs["source"] = ProcessConfig(num_replicas=REPLICA_COUNT)
    config = PipelineConfig(
        model_path="mock",
        entry_stage="source",
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
        "source",
        ["sink"],
        logical_process_plan=plan,
        replica_topology=topology,
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
                factory_kwargs={"name": stage.name, "events": events},
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
    operation: Literal["open", "append", "close"],
    session_identity: SessionIdentity,
    chunk: TimedChunk | None = None,
) -> SessionMetadata:
    session_operation = SessionOperation(
        operation=operation,
        session_identity=session_identity,
        stages=("source",),
        chunk=chunk,
    )
    return {SESSION_METADATA_KEY: session_operation.to_dict()}


def chunk(seq: int, eos: bool = False) -> TimedChunk:
    return TimedChunk("audio", seq * 20, 20, seq, b"pcm", eos=eos)


PipelineResources = tuple[Coordinator, Queue[StageEvent], list[SpawnProcess]]


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
