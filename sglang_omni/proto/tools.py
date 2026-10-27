# SPDX-License-Identifier: Apache-2.0
"""Tool call and response payloads carried by Session streams."""

from pydantic import BaseModel

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import IMAGE, TOOL, TOOL_CALL, TOOL_IMAGE, TOOL_RESPONSE


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: object


class ToolImage(BaseModel):
    """An image a tool made for the client: its MIME type and its bytes as base64. The model reads only the response's data."""

    mime_type: str
    data: str


class ToolResult(BaseModel):
    """What a tool returns when it made images beside the data the model reads."""

    response: object
    images: list[ToolImage] = []


class ToolResponse(BaseModel):
    id: str
    name: str
    response: object
    images: list[ToolImage] = []


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


def tool_response_chunk(responses: list[ToolResponse], turn: int | None) -> TimedChunk:
    return TimedChunk(
        TOOL,
        0,
        0,
        0,
        {"responses": [response.model_dump() for response in responses]},
        format=TOOL_RESPONSE,
        turn=turn,
    )


def tool_image_chunk(call: ToolCall, image: ToolImage, turn: int) -> TimedChunk:
    """One image a tool made, published with the call that asked for it in the turn that reports the result."""
    return TimedChunk(
        IMAGE,
        0,
        0,
        0,
        {
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments,
            "mime_type": image.mime_type,
            "data": image.data,
        },
        format=TOOL_IMAGE,
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
