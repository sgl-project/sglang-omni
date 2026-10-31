# SPDX-License-Identifier: Apache-2.0
"""The async_tools kind: tool calls answered at once and background tasks reported as they end."""

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.models.streaming_cascade.base import ComponentConfig, StreamComponent
from sglang_omni.models.tools.runner import ToolRunner
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import INTERRUPT, TOOL_CALL, control_type
from sglang_omni.proto.tools import (
    FollowUp,
    TaskArtifact,
    TaskResult,
    TaskStatus,
    TaskUpdate,
    ToolCall,
    ToolResponse,
    task_result_chunk,
    task_update_chunk,
    tool_calls_of,
    tool_response_chunk,
)
from sglang_omni.scheduling.session import ChunkEmitter
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)

CANCEL_TOOL_NAME = "cancel_task"
# Seconds a closing session waits for its cancelled tasks; below the component timeout, which bounds the whole close.
CLOSE_GRACE_S = 2.0


class ToolOutcome(BaseModel):
    """What a background tool returns: a short text the model reads and what the client shows."""

    summary: str
    artifact: TaskArtifact | None = None


class BackgroundTool(Protocol):
    async def __call__(
        self, arguments: Mapping[str, object], options: Mapping[str, object]
    ) -> ToolOutcome: ...


class AsyncToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implementation: str = Field(min_length=1)
    mode: Literal["sync", "async"] = "async"
    timeout_s: float = Field(default=180, gt=0)
    on_interrupt: Literal["keep", "cancel"] = "cancel"
    follow_up: FollowUp = "speak"
    options: dict[str, object] = Field(default_factory=dict)


class AsyncToolsConfig(ComponentConfig):
    """Tools by function name; a sync tool answers in its tool response, an async one starts a task the response names."""

    kind: Literal["async_tools"] = "async_tools"
    tools: dict[str, AsyncToolConfig] = Field(min_length=1)
    sync_timeout_s: float = Field(default=5, gt=0)
    max_jobs_per_session: int = Field(default=2, gt=0)
    # Raw artifact bytes; base64 grows them by a third on the wire.
    max_artifact_bytes: int = Field(default=3 * 1024 * 1024, gt=0)
    cancel_tool: bool = True

    def tool_names(self) -> set[str]:
        return set(self.tools) | ({CANCEL_TOOL_NAME} if self.cancel_tool else set())


@dataclass(kw_only=True)
class Job:
    task_id: str
    call: ToolCall
    origin_turn: int
    config: AsyncToolConfig
    status: TaskStatus = "created"
    cancel_reason: str = ""
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class AsyncTools(StreamComponent):
    """One Session's tool calls and background tasks.

    Every tool_call chunk is answered with one tool_response of its turn
    before the next chunk is read, so the node never holds the Session on a
    slow tool. An async tool's task reports created, running and its end on
    task_update and, unless the Session closes, task_result; neither carries
    a turn, so a task outlives the reply that started it. An interrupt of
    that reply cancels the tasks of tools whose on_interrupt is cancel.
    """

    def __init__(self, config: AsyncToolsConfig) -> None:
        self.config = config
        self.emit: ChunkEmitter | None = None
        self.jobs: dict[str, Job] = {}
        self.task_count = 0
        self.is_closing = False
        # note (zengjiahua): the graph fences a turn's queued chunks at its interrupt, but a call already handed to this node passed the fence.
        self.interrupted_turns: set[int] = set()
        self.sync_runner = ToolRunner.from_paths(
            {
                name: tool.implementation
                for name, tool in config.tools.items()
                if tool.mode == "sync"
            }
        )
        self.background_tools: dict[str, BackgroundTool] = {
            name: import_string(tool.implementation)
            for name, tool in config.tools.items()
            if tool.mode == "async"
        }

    def can_replace(self) -> bool:
        return not self.running_jobs()

    def running_jobs(self) -> list[Job]:
        return [
            job for job in self.jobs.values() if job.status in ("created", "running")
        ]

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        self.emit = emit
        async for chunk in inputs:
            if chunk.format == TOOL_CALL:
                responses = [
                    await self.respond(call, chunk.turn)
                    for call in tool_calls_of(chunk)
                ]
                emit(tool_response_chunk(responses, chunk.turn))
            else:
                pass
        await self.cancel_jobs(self.running_jobs(), "the input ended")

    async def respond(self, call: ToolCall, turn: int) -> ToolResponse:
        tool = self.config.tools.get(call.name)
        if call.name == CANCEL_TOOL_NAME and self.config.cancel_tool:
            return ToolResponse(
                id=call.id, name=call.name, response=await self.cancel_by_call(call)
            )
        elif tool is None or tool.mode == "sync":
            try:
                return await asyncio.wait_for(
                    self.sync_runner.respond(call), self.config.sync_timeout_s
                )
            except TimeoutError:
                return ToolResponse(
                    id=call.id,
                    name=call.name,
                    response={"error": "the tool did not answer in time"},
                )
        elif not isinstance(call.arguments, dict):
            return ToolResponse(
                id=call.id,
                name=call.name,
                response={"error": "arguments must be an object"},
            )
        elif len(self.running_jobs()) >= self.config.max_jobs_per_session:
            return ToolResponse(
                id=call.id,
                name=call.name,
                response={
                    "status": "rejected",
                    "error": "busy: too many tasks are running",
                },
            )
        elif turn in self.interrupted_turns and tool.on_interrupt == "cancel":
            return ToolResponse(
                id=call.id,
                name=call.name,
                response={"status": "cancelled", "error": "the reply was interrupted"},
            )
        else:
            self.task_count += 1
            job = Job(
                task_id=f"task_{self.task_count}",
                call=call,
                origin_turn=turn,
                config=tool,
            )
            self.jobs[job.task_id] = job
            self.publish(job)
            job.task = asyncio.create_task(self.run_job(job))
            return ToolResponse(
                id=call.id,
                name=call.name,
                response={"status": "accepted", "task_id": job.task_id},
            )

    async def cancel_by_call(self, call: ToolCall) -> dict[str, object]:
        task_id = (
            call.arguments.get("task_id") if isinstance(call.arguments, dict) else None
        )
        job = self.jobs.get(str(task_id))
        if job is None:
            return {"error": f"no task {task_id}"}
        elif job.status not in ("created", "running"):
            return {"task_id": job.task_id, "status": job.status}
        else:
            await self.cancel_jobs([job], "the user asked to cancel it")
            return {"task_id": job.task_id, "status": "cancelled"}

    async def run_job(self, job: Job) -> None:
        tool = self.background_tools[job.call.name]
        job.status = "running"
        self.publish(job)
        try:
            outcome = await asyncio.wait_for(
                tool(job.call.arguments, job.config.options), job.config.timeout_s
            )
        except asyncio.CancelledError:
            self.finish(job, "cancelled", error=job.cancel_reason)
            raise
        except TimeoutError:
            self.finish(
                job, "failed", error=f"no result within {job.config.timeout_s:g} s"
            )
        except Exception as error:
            logger.warning(f"task {job.task_id} {job.call.name} failed", exc_info=error)
            self.finish(job, "failed", error=str(error) or type(error).__name__)
        else:
            if (
                outcome.artifact is not None
                and len(outcome.artifact.data) * 3 // 4 > self.config.max_artifact_bytes
            ):
                self.finish(job, "failed", error="the result is too large to send")
            else:
                self.finish(
                    job, "succeeded", summary=outcome.summary, artifact=outcome.artifact
                )

    def finish(
        self,
        job: Job,
        status: TaskStatus,
        *,
        summary: str = "",
        error: str = "",
        artifact: TaskArtifact | None = None,
    ) -> None:
        """End a task once; a closing Session sends nothing more."""
        if job.status not in ("created", "running") or self.is_closing:
            job.status = status
            return
        else:
            job.status = status
        self.publish(job, error=error or None, artifact=artifact)
        assert self.emit is not None
        self.emit(
            task_result_chunk(
                TaskResult(
                    task_id=job.task_id,
                    name=job.call.name,
                    status=status,
                    summary=summary,
                    error=error or None,
                    follow_up=(
                        "silent" if status == "cancelled" else job.config.follow_up
                    ),
                )
            )
        )

    def publish(
        self,
        job: Job,
        *,
        error: str | None = None,
        artifact: TaskArtifact | None = None,
    ) -> None:
        assert self.emit is not None
        self.emit(
            task_update_chunk(
                TaskUpdate(
                    task_id=job.task_id,
                    call_id=job.call.id,
                    name=job.call.name,
                    arguments=job.call.arguments,
                    origin_turn=job.origin_turn,
                    status=job.status,
                    error=error,
                    artifact=artifact,
                )
            )
        )

    async def cancel_jobs(self, jobs: list[Job], reason: str) -> None:
        """Cancel the jobs and end each as cancelled, also one whose task was cancelled before it began and so never ran its own end."""
        tasks = [
            job.task for job in jobs if job.task is not None and not job.task.done()
        ]
        for job in jobs:
            job.cancel_reason = reason
            if job.task is not None:
                job.task.cancel()
            else:
                pass
        if tasks:
            await asyncio.wait(tasks, timeout=CLOSE_GRACE_S)
        else:
            pass
        for job in jobs:
            self.finish(job, "cancelled", error=reason)

    async def control(self, event: TimedChunk) -> None:
        if control_type(event) == INTERRUPT and event.turn is not None:
            self.interrupted_turns.add(event.turn)
            await self.cancel_jobs(
                [
                    job
                    for job in self.running_jobs()
                    if job.origin_turn == event.turn
                    and job.config.on_interrupt == "cancel"
                ],
                "the reply that started it was interrupted",
            )
        else:
            pass

    async def close(self) -> None:
        self.is_closing = True
        await self.cancel_jobs(self.running_jobs(), "the session closed")
