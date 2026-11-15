# SPDX-License-Identifier: Apache-2.0
"""Execute configured Python functions for tool calls."""

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping

from sglang_omni.proto.tools import ToolCall, ToolResponse
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)

ToolFunction = Callable[[Mapping[str, object]], object]


class ToolRunner:
    """Answers each tool call; an unknown tool, arguments that are not an object or an exception from the tool become `{"error": str}` responses, so the model answers them instead of the Session failing."""

    def __init__(self, tools: dict[str, ToolFunction]) -> None:
        self.tools = tools

    @classmethod
    def from_paths(cls, tools: dict[str, str]) -> "ToolRunner":
        return cls({name: import_string(path) for name, path in tools.items()})

    async def respond(self, call: ToolCall) -> ToolResponse:
        tool = self.tools.get(call.name)
        if tool is None:
            response: object = {"error": f"unknown tool {call.name}"}
        elif not isinstance(call.arguments, dict):
            response = {"error": "arguments must be an object"}
        else:
            try:
                if inspect.iscoroutinefunction(tool):
                    response = await tool(call.arguments)
                else:
                    response = await asyncio.to_thread(tool, call.arguments)
            except Exception as error:
                logger.warning(f"tool {call.name} failed", exc_info=error)
                response = {"error": str(error)}
        return ToolResponse(id=call.id, name=call.name, response=response)

    async def respond_all(self, calls: list[ToolCall]) -> list[ToolResponse]:
        return [await self.respond(call) for call in calls]
