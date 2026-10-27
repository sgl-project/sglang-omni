# SPDX-License-Identifier: Apache-2.0
"""The tools kind: the registered Python functions, answering the calls the generator makes."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sglang_omni.models.streaming_cascade.base import (
    ComponentConfig,
    StreamComponent,
    stop_running,
)
from sglang_omni.models.tools.runner import ToolRunner
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import INTERRUPT, TOOL_CALL, control_type
from sglang_omni.proto.tools import (
    ToolCall,
    ToolResponse,
    tool_calls_of,
    tool_response_chunk,
)
from sglang_omni.scheduling.session import ChunkEmitter

logger = logging.getLogger(__name__)


class ToolSettings(BaseModel):
    """One tool: `path` imports the Python function running it, which takes the call's arguments as a mapping and returns JSON data, synchronous or async. The reply that calls an `awaited` tool stays open until the result continues it; a call of an unawaited tool ends the reply, and its result opens a reply of its own once the dialog is idle. When the user interrupts the reply that made a call, `on_interrupt` cancels the call or retains it, and a retained call's result opens a reply of its own too."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    awaited: bool = True
    on_interrupt: Literal["cancel", "retain"] = "cancel"


class ToolsConfig(ComponentConfig):
    """`tools` maps each tool name to its settings, or to its import path alone for an awaited tool an interrupt cancels; every tool a generator of the Session offers must be among them. A call that takes longer than `timeout_s` answers with an error."""

    kind: Literal["tools"] = "tools"
    tools: dict[str, ToolSettings] = Field(min_length=1)

    @field_validator("tools", mode="before")
    @classmethod
    def settings_of_paths(cls, tools: object) -> object:
        if isinstance(tools, dict):
            return {
                name: {"path": tool} if isinstance(tool, str) else tool
                for name, tool in tools.items()
            }
        else:
            return tools


class Tools(StreamComponent):
    """Runs each call of a tool_call chunk as a task of its own and answers it with a tool_response chunk as soon as it completes.

    A response carries no turn: the dialog matches it to the call it awaits
    by id, so the graph's turn fence leaves tool results alone. An
    `interrupt` control of a turn cancels the turn's running calls whose
    tools an interrupt cancels and leaves the retained ones running, a
    call that outlasts the node's timeout answers with an error, and the
    end of the input waits for the calls still running.
    """

    def __init__(self, config: ToolsConfig) -> None:
        self.config = config
        self.runner = ToolRunner.from_paths(
            {name: tool.path for name, tool in config.tools.items()}
        )
        # The running calls, each with the turn of the reply that made it and the settings of its tool, which an unknown tool lacks.
        self.running: dict[
            asyncio.Task[None], tuple[int | None, ToolSettings | None]
        ] = {}

    def can_replace(self) -> bool:
        return not self.running

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        try:
            async for chunk in inputs:
                if chunk.format == TOOL_CALL:
                    for call in tool_calls_of(chunk):
                        task = asyncio.create_task(self.answer(call, chunk.turn, emit))
                        self.running[task] = (
                            chunk.turn,
                            self.config.tools.get(call.name),
                        )
                        task.add_done_callback(self.running.pop)
                else:
                    pass
            # Note (Xuesong): A call an interrupt cancels while the input end waits here is not the node's failure; a call that raised is.
            outcomes = await asyncio.gather(*self.running, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    raise outcome
                else:
                    pass
        finally:
            await stop_running(list(self.running))

    async def answer(
        self, call: ToolCall, turn: int | None, emit: ChunkEmitter
    ) -> None:
        try:
            response = await asyncio.wait_for(
                self.runner.respond(call), self.config.timeout_s
            )
        except TimeoutError:
            response = ToolResponse(
                id=call.id,
                name=call.name,
                response={
                    "error": f"{call.name} gave no answer within {self.config.timeout_s} s"
                },
            )
        logger.info(
            f"tool {call.id} {call.name}({call.arguments}) -> {response.response} turn={turn}"
        )
        emit(tool_response_chunk([response], None))

    async def control(self, event: TimedChunk) -> None:
        if control_type(event) == INTERRUPT:
            await stop_running(
                [
                    task
                    for task, (turn, tool) in self.running.items()
                    if (event.turn is None or turn == event.turn)
                    and (tool is None or tool.on_interrupt == "cancel")
                ]
            )
        else:
            pass


async def probe_tools(config: ToolsConfig) -> None:
    """Require every implementation to import."""
    ToolRunner.from_paths({name: tool.path for name, tool in config.tools.items()})
