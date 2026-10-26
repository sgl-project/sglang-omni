# SPDX-License-Identifier: Apache-2.0
"""Shared coordinator and adapter fakes for realtime session tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sglang_omni.serve.realtime.output import OutputEvent
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import InteractionAdapter, OutputSink, Unit

MODEL_NAME = "duplex-test"


class HealthCoordinator:
    def __init__(self, is_running: bool) -> None:
        self.is_running = is_running

    def health(self) -> dict[str, bool]:
        return {"running": self.is_running}


@dataclass(kw_only=True)
class ScriptedAdapter(InteractionAdapter):
    """Records native units and emits a fixed reply while processing each one."""

    reply: list[OutputEvent] = field(default_factory=list)
    unconsumed_samples: int = 0
    is_holding_input: bool = False
    open_error: str | None = None
    close_error: str | None = None
    should_stall_close: bool = False
    units: list[Unit] = field(default_factory=list)
    held_samples: int = 0
    output_sink: OutputSink | None = None
    is_closed: bool = False

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        if self.open_error is not None:
            raise RuntimeError(self.open_error)
        else:
            self.output_sink = emit

    async def process(self, unit: Unit) -> int | tuple[int, int]:
        assert self.output_sink is not None
        self.units.append(unit)
        for event in self.reply:
            await self.output_sink(event)
        if self.is_holding_input:
            self.held_samples += unit.real_samples
            return 0, 0
        else:
            return max(unit.real_samples - self.unconsumed_samples, 0)

    async def clear(self) -> int:
        held_samples, self.held_samples = self.held_samples, 0
        return held_samples

    async def close(self) -> None:
        self.is_closed = True
        if self.should_stall_close:
            await asyncio.Event().wait()
        elif self.close_error is not None:
            raise RuntimeError(self.close_error)
        else:
            pass
