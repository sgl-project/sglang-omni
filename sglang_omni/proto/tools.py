# SPDX-License-Identifier: Apache-2.0
"""Tool call and response payloads carried by Session streams."""

from typing import Literal

from pydantic import BaseModel

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    TASK,
    TASK_RESULT,
    TASK_UPDATE,
    TEXT,
    TOOL,
    TOOL_CALL,
    TOOL_EXCHANGE,
    TOOL_RESPONSE,
)

TaskStatus = Literal["created", "running", "succeeded", "failed", "cancelled"]
FollowUp = Literal["speak", "silent"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: object


class ToolResponse(BaseModel):
    id: str
    name: str
    response: object


def tool_call_chunk(calls: list[ToolCall], turn: int) -> TimedChunk:
    return TimedChunk(
        TOOL,
        0,
        0,
        0,
        {"calls": [call.model_dump() for call in calls]},
        format=TOOL_CALL,
        turn=turn,
    )


def tool_response_chunk(responses: list[ToolResponse], turn: int) -> TimedChunk:
    return TimedChunk(
        TOOL,
        0,
        0,
        0,
        {"responses": [response.model_dump() for response in responses]},
        format=TOOL_RESPONSE,
        turn=turn,
    )


def tool_calls_of(chunk: TimedChunk) -> list[ToolCall]:
    if chunk.format != TOOL_CALL or not isinstance(chunk.payload, dict):
        raise ValueError("expected a tool_call chunk")
    else:
        return [ToolCall.model_validate(call) for call in chunk.payload["calls"]]


def tool_responses_of(chunk: TimedChunk) -> list[ToolResponse]:
    if chunk.format != TOOL_RESPONSE or not isinstance(chunk.payload, dict):
        raise ValueError("expected a tool_response chunk")
    else:
        return [
            ToolResponse.model_validate(response)
            for response in chunk.payload["responses"]
        ]


class ToolExchange(BaseModel):
    """One round of a reply's tool calls and their responses, after text_offset characters of the reply's text."""

    text_offset: int
    calls: list[ToolCall]
    responses: list[ToolResponse]


class TaskArtifact(BaseModel):
    type: Literal["image"]
    media_type: str
    data: str  # base64


class TaskUpdate(BaseModel):
    task_id: str
    call_id: str
    name: str
    arguments: object
    origin_turn: int
    status: TaskStatus
    error: str | None = None
    artifact: TaskArtifact | None = None


class TaskResult(BaseModel):
    task_id: str
    name: str
    status: TaskStatus
    summary: str = ""
    error: str | None = None
    follow_up: FollowUp


def tool_exchange_chunk(exchange: ToolExchange, turn: int) -> TimedChunk:
    return TimedChunk(
        TEXT, 0, 0, 0, exchange.model_dump(), format=TOOL_EXCHANGE, turn=turn
    )


def tool_exchange_of(chunk: TimedChunk) -> ToolExchange:
    if chunk.format != TOOL_EXCHANGE:
        raise ValueError("expected a tool_exchange chunk")
    else:
        return ToolExchange.model_validate(chunk.payload)


def task_update_chunk(update: TaskUpdate) -> TimedChunk:
    """A task's state for the client; it carries no turn, so an interrupted reply does not fence the task that reply started."""
    return TimedChunk(
        TASK, 0, 0, 0, update.model_dump(exclude_none=True), format=TASK_UPDATE
    )


def task_result_chunk(result: TaskResult) -> TimedChunk:
    return TimedChunk(TASK, 0, 0, 0, result.model_dump(), format=TASK_RESULT)


def task_result_of(chunk: TimedChunk) -> TaskResult:
    if chunk.format != TASK_RESULT:
        raise ValueError("expected a task_result chunk")
    else:
        return TaskResult.model_validate(chunk.payload)
