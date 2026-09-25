"""Interaction bridges. No transport parsing or model-name dispatch."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Protocol

from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.serve.realtime.output import OutputEvent, TurnFailure
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks
from sglang_omni.serve.realtime.types import (
    InteractionAdapter,
    OutputSink,
    RuntimeLimits,
    Unit,
    samples_to_ms,
)

logger = logging.getLogger(__name__)


class RequestBuilder(Protocol):
    def __call__(self, config: SessionConfiguration, /) -> OmniRequest: ...


class OutputConverter(Protocol):
    def __call__(self, output: OutputChunk, /) -> Iterable[OutputEvent]: ...


class CoordinatorAdapter(InteractionAdapter):
    """Publish output after unit success."""

    def __init__(
        self,
        client: Client,
        *,
        stages: list[str],
        request_builder: RequestBuilder,
        output_converter: OutputConverter,
        input_sample_rate_hz: int | None = None,
        atomic_consumption: bool = False,
        limits: SessionLimits | None = None,
    ) -> None:
        if not atomic_consumption:
            raise ValueError("a producer atomic-consumption contract is required")
        else:
            pass
        self.client = client
        self.stages = stages
        self.request_builder = request_builder
        self.output_converter = output_converter
        self.input_sample_rate_hz = input_sample_rate_hz
        self.limits = limits or SessionLimits()
        self.cleanup_timeout_s = self.limits.operation_timeout_s
        self.session_identity: SessionIdentity | None = None
        self.output_reader_task: asyncio.Task[None] | None = None
        self.active_unit: Unit | None = None
        self.unit_completion: asyncio.Future[int] | None = None
        self.unit_output_events: list[OutputEvent] = []
        self.unit_output_bytes = 0
        self.output_sink: OutputSink | None = None
        self.reader_error: Exception | None = None
        self.is_closing = False

    def set_limits(self, limits: RuntimeLimits) -> None:
        self.cleanup_timeout_s = limits.cleanup_timeout_s

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        negotiated_sample_rate_hz = config["audio"]["input"]["format"]["rate"]
        if (
            self.input_sample_rate_hz is not None
            and self.input_sample_rate_hz != negotiated_sample_rate_hz
        ):
            raise ValueError(
                f"adapter input sample rate {self.input_sample_rate_hz} differs from "
                f"negotiated rate {negotiated_sample_rate_hz}"
            )
        else:
            self.input_sample_rate_hz = negotiated_sample_rate_hz
        self.output_sink = emit
        self.session_identity = await self.client.open_session(
            self.request_builder(config),
            stages=self.stages,
            limits=self.limits,
            session_id=session_id,
        )
        self.output_reader_task = asyncio.create_task(self.read_session_outputs())

    def reset_unit_outputs(self) -> None:
        self.unit_output_events.clear()
        self.unit_output_bytes = 0

    async def read_session_outputs(self) -> None:
        assert self.session_identity is not None and self.output_sink is not None
        try:
            async for output in self.client.session_outputs(self.session_identity):
                if (
                    self.active_unit is None
                    or output.input_seq != self.active_unit.index
                ):
                    continue
                elif output.kind == "input_done":
                    for event in self.unit_output_events:
                        await self.output_sink(event, self.active_unit)
                    self.reset_unit_outputs()
                    if (
                        self.unit_completion is not None
                        and not self.unit_completion.done()
                    ):
                        self.unit_completion.set_result(self.active_unit.real_samples)
                    else:
                        pass
                else:
                    for event in self.output_converter(output):
                        event_size_bytes = len(repr(event).encode())
                        if (
                            len(self.unit_output_events)
                            >= self.limits.max_output_chunks
                            or self.unit_output_bytes + event_size_bytes
                            > self.limits.max_output_bytes
                        ):
                            raise RuntimeError("native unit output budget exhausted")
                        else:
                            pass
                        self.unit_output_events.append(event)
                        self.unit_output_bytes += event_size_bytes
            if not self.is_closing:
                raise RuntimeError("session output stream closed")
            else:
                pass
        except Exception as exc:
            logger.exception("Realtime session output reader failed")
            self.reader_error = exc
            if self.unit_completion is not None and not self.unit_completion.done():
                self.unit_completion.set_exception(exc)
            else:
                await self.output_sink(
                    TurnFailure("server_error", "internal", str(exc))
                )

    async def process(self, unit: Unit) -> int:
        assert (
            self.session_identity is not None and self.input_sample_rate_hz is not None
        )
        if self.reader_error is not None:
            raise self.reader_error
        else:
            pass
        self.active_unit = unit
        self.unit_completion = asyncio.get_running_loop().create_future()
        timed_chunk = TimedChunk(
            "audio",
            samples_to_ms(unit.start_sample, self.input_sample_rate_hz),
            samples_to_ms(unit.real_samples, self.input_sample_rate_hz),
            unit.index,
            unit.pcm,
            format="pcm16",
            eos=unit.eos,
        )
        try:
            await self.client.append_session(self.session_identity, timed_chunk)
            return await self.unit_completion
        finally:
            self.active_unit = None
            self.unit_completion = None
            self.reset_unit_outputs()

    async def close(self) -> None:
        self.is_closing = True
        try:
            if self.session_identity is not None:
                await self.client.close_session(self.session_identity)
            else:
                pass
        finally:
            await cancel_local_tasks([self.output_reader_task], self.cleanup_timeout_s)
