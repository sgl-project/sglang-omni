# SPDX-License-Identifier: Apache-2.0
"""A graph node that runs external tools for a session.

Input chunk payload: {"calls": [{"name": str, "arguments": {...}}, ...]}.
Each call runs a registered Python function; the node emits one
tool_response chunk {"responses": [{"name": str, "response": {...}}, ...]}.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Protocol

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)

TOOL_RESPONSE_MODALITY = "tool_response"


class ToolFunction(Protocol):
    def __call__(self, arguments: Mapping[str, object]) -> dict[str, object]: ...


class ToolHooks(SessionHooks):
    """Stateless across units; a bad call becomes an error response, not a session failure."""

    def __init__(self, tools: dict[str, ToolFunction]) -> None:
        self.tools = tools

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        pass

    def append(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        calls = chunk.payload.get("calls") if isinstance(chunk.payload, dict) else None
        if not isinstance(calls, list):
            raise ValueError("tool chunk payload must carry a calls list")
        else:
            pass
        responses = [self.run_call(call) for call in calls]
        context.emit(
            TimedChunk(
                modality=TOOL_RESPONSE_MODALITY,
                t_start_ms=chunk.t_start_ms + chunk.duration_ms,
                duration_ms=0,
                seq=0,
                payload={"responses": responses},
            )
        )
        payload.data = {"tool_calls": len(calls)}
        return payload

    def run_call(self, call: object) -> dict[str, object]:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            return {"name": "", "response": {"error": "malformed tool call"}}
        else:
            pass
        name = call["name"]
        arguments = call.get("arguments", {})
        tool = self.tools.get(name)
        if tool is None:
            return {"name": name, "response": {"error": f"unknown tool {name}"}}
        elif not isinstance(arguments, dict):
            return {"name": name, "response": {"error": "arguments must be an object"}}
        else:
            try:
                response = tool(arguments)
                logger.info(f"tool {name} called with {arguments} returned {response}")
                return {"name": name, "response": response}
            except (TypeError, ValueError, KeyError) as exc:
                # Note (Dayuxiaoshui): Model-written arguments are untrusted; the model gets the error and can retry.
                logger.warning(f"tool {name} rejected arguments {arguments}: {exc}")
                return {"name": name, "response": {"error": str(exc)}}

    def close(self, session_identity: SessionIdentity) -> None:
        pass


def create_tool_scheduler(tools: dict[str, str]) -> SessionScheduler:
    """tools maps a tool name to the import path of its ToolFunction."""
    return SessionScheduler(
        ToolHooks({name: import_string(path) for name, path in tools.items()})
    )
