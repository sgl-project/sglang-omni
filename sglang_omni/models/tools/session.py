# SPDX-License-Identifier: Apache-2.0
"""Adapt Python tool functions to a CPU Session stage."""

import asyncio
import logging

from sglang_omni.models.tools.runner import ToolRunner
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import TOOL_CALL
from sglang_omni.proto.tools import tool_calls_of, tool_response_chunk
from sglang_omni.scheduling.session import (
    DEFAULT_MAX_OPEN_SESSIONS,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)

logger = logging.getLogger(__name__)


class ToolHooks(SessionHooks):
    """The tools as a stage: one node of a Session graph, answering inside the append that carried the call."""

    def __init__(self, runner: ToolRunner) -> None:
        self.runner = runner

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        pass

    def close(self, session_identity: SessionIdentity) -> None:
        pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if chunk.format == TOOL_CALL:
            calls = tool_calls_of(chunk)
            responses = asyncio.run(self.runner.respond_all(calls))
            for call, answered in zip(calls, responses):
                logger.info(
                    "tool %s %s(%s) -> %s turn=%s",
                    call.id,
                    call.name,
                    call.arguments,
                    answered.response,
                    chunk.turn,
                )
            response = tool_response_chunk(responses, chunk.turn)
            context.emit(
                TimedChunk(
                    response.modality,
                    chunk.t_start_ms,
                    0,
                    chunk.seq,
                    response.payload,
                    format=response.format,
                    turn=response.turn,
                )
            )
        else:
            raise ValueError("the tool stage takes tool_call chunks")
        payload.data = {}
        return payload


def create_tool_scheduler(
    tools: dict[str, str], *, max_open_sessions: int = DEFAULT_MAX_OPEN_SESSIONS
) -> SessionScheduler:
    return SessionScheduler(
        ToolHooks(ToolRunner.from_paths(tools)), max_open_sessions=max_open_sessions
    )
