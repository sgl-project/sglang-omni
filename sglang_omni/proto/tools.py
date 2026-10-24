# SPDX-License-Identifier: Apache-2.0
"""Tool call and response payloads carried by Session streams."""

from pydantic import BaseModel

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import TOOL, TOOL_CALL, TOOL_RESPONSE


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
