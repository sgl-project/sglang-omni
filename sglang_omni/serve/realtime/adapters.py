"""Interaction bridges. No transport parsing or model-name dispatch."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    DEFAULT_IDLE_TIMEOUT_S,
    LM_STEP_MS_KEY,
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.serve.realtime.constants import CLOSE_REASON_INTERNAL
from sglang_omni.serve.realtime.output import OutputEvent, TurnFailure
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks
from sglang_omni.serve.realtime.types import (
    InteractionAdapter,
    OutputBudgetError,
    OutputSink,
    RuntimeLimits,
    Unit,
    failure_code,
    samples_to_ms,
)

logger = logging.getLogger(__name__)
RUNTIME_IDLE_TIMEOUT_PERIODS = 2


def pipeline_limits_with_idle_backup(
    pipeline_limits: SessionLimits, runtime_limits: RuntimeLimits
) -> SessionLimits:
    # note (Haiyang Luo): Leave time for runtime idle checks and native cleanup.
    pipeline_idle_timeout_seconds = max(
        DEFAULT_IDLE_TIMEOUT_S,
        pipeline_limits.idle_timeout_s,
        RUNTIME_IDLE_TIMEOUT_PERIODS * runtime_limits.idle_timeout_s
        + runtime_limits.cleanup_timeout_s
        + pipeline_limits.operation_timeout_s,
    )
    return replace(pipeline_limits, idle_timeout_s=pipeline_idle_timeout_seconds)


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
        # Note (Junnan Li): Outputs that arrive for a queued unit wait until it becomes
        # the active one.
        self.queued_units: deque[tuple[Unit, asyncio.Future[int]]] = deque()
        self.queued_output_events: dict[int, list[OutputEvent]] = {}
        self.queued_output_bytes: dict[int, int] = {}
        self.output_sink: OutputSink | None = None
        self.reader_error: Exception | None = None
        self.is_closing = False

    def set_limits(self, runtime_limits: RuntimeLimits) -> None:
        self.cleanup_timeout_s = runtime_limits.cleanup_timeout_s
        self.limits = pipeline_limits_with_idle_backup(self.limits, runtime_limits)

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
            limits=self.limits,
            session_id=session_id,
        )
        self.output_reader_task = asyncio.create_task(self.read_session_outputs())

    def reset_unit_outputs(self) -> None:
        self.unit_output_events.clear()
        self.unit_output_bytes = 0

    def activate_next_unit(self) -> None:
        self.reset_unit_outputs()
        if self.queued_units:
            self.active_unit, self.unit_completion = self.queued_units.popleft()
            self.queued_output_bytes.pop(self.active_unit.index, None)
            for event in self.queued_output_events.pop(self.active_unit.index, []):
                self.buffer_unit_output(event)
        else:
            self.active_unit, self.unit_completion = None, None

    def check_output_budget(self, chunks: int, output_bytes: int) -> None:
        if (
            chunks > self.limits.max_output_chunks
            or output_bytes > self.limits.max_output_bytes
        ):
            raise OutputBudgetError("native unit output budget exhausted")
        else:
            pass

    def buffer_unit_output(self, event: OutputEvent) -> None:
        event_size_bytes = len(repr(event).encode())
        self.check_output_budget(
            len(self.unit_output_events) + 1, self.unit_output_bytes + event_size_bytes
        )
        self.unit_output_events.append(event)
        self.unit_output_bytes += event_size_bytes

    def queue_unit_output(self, index: int, event: OutputEvent) -> None:
        events = self.queued_output_events.setdefault(index, [])
        output_bytes = self.queued_output_bytes.get(index, 0) + len(
            repr(event).encode()
        )
        self.check_output_budget(len(events) + 1, output_bytes)
        events.append(event)
        self.queued_output_bytes[index] = output_bytes

    async def read_session_outputs(self) -> None:
        assert self.session_identity is not None and self.output_sink is not None
        try:
            async for output in self.client.session_outputs(self.session_identity):
                if any(output.input_seq == unit.index for unit, _ in self.queued_units):
                    if output.kind != "input_done":
                        for event in self.output_converter(output):
                            self.queue_unit_output(output.input_seq, event)
                    else:
                        pass
                    continue
                elif (
                    self.active_unit is None
                    or output.input_seq != self.active_unit.index
                ):
                    continue
                elif output.kind == "input_done":
                    if (
                        isinstance(output.payload, dict)
                        and LM_STEP_MS_KEY in output.payload
                    ):
                        self.model_step_ms = output.payload[LM_STEP_MS_KEY]
                    else:
                        pass
                    for event in self.unit_output_events:
                        await self.output_sink(event, self.active_unit)
                    if (
                        self.unit_completion is not None
                        and not self.unit_completion.done()
                    ):
                        self.unit_completion.set_result(self.active_unit.real_samples)
                    else:
                        pass
                    self.activate_next_unit()
                else:
                    for event in self.output_converter(output):
                        self.buffer_unit_output(event)
            if not self.is_closing:
                raise RuntimeError("session output stream closed")
            else:
                pass
        except Exception as exc:
            if ContextExhaustedError.matches(exc):
                code = ContextExhaustedError.CODE
            else:
                code = failure_code(exc)
            if code == CLOSE_REASON_INTERNAL:
                logger.exception("Realtime session output reader failed")
            else:
                pass
            self.reader_error = exc
            for _, completion in self.queued_units:
                if not completion.done():
                    completion.set_exception(exc)
                else:
                    pass
            if self.unit_completion is not None and not self.unit_completion.done():
                self.unit_completion.set_exception(exc)
            else:
                await self.output_sink(TurnFailure("server_error", code, str(exc)))

    async def process(self, unit: Unit) -> int:
        assert (
            self.session_identity is not None and self.input_sample_rate_hz is not None
        )
        if self.reader_error is not None:
            raise self.reader_error
        else:
            pass
        completion: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        if self.active_unit is None:
            self.active_unit, self.unit_completion = unit, completion
        else:
            self.queued_units.append((unit, completion))
        timed_chunk = TimedChunk(
            "audio",
            samples_to_ms(unit.start_sample, self.input_sample_rate_hz),
            samples_to_ms(unit.real_samples, self.input_sample_rate_hz),
            unit.index,
            (
                unit.pcm
                if not unit.images
                else {"pcm": unit.pcm, "images": list(unit.images)}
            ),
            format="pcm16",
            eos=unit.eos,
        )
        tick_kwargs = {} if unit.tick is None else {"tick": unit.tick}
        try:
            ready_observation = unit.ready_observation
            await self.client.append_session(
                self.session_identity,
                timed_chunk,
                ready_timestamp_ns=(
                    ready_observation.timestamp_ns
                    if ready_observation is not None
                    else None
                ),
                ready_run_id=(
                    ready_observation.run_id if ready_observation is not None else None
                ),
                **tick_kwargs,
            )
            return await completion
        finally:
            if self.active_unit is unit:
                self.activate_next_unit()
            elif any(queued is unit for queued, _ in self.queued_units):
                self.queued_units = deque(
                    entry for entry in self.queued_units if entry[0] is not unit
                )
                self.queued_output_events.pop(unit.index, None)
                self.queued_output_bytes.pop(unit.index, None)
            else:
                pass

    async def close(self) -> None:
        self.is_closing = True
        try:
            if self.session_identity is not None:
                await self.client.close_session(self.session_identity)
            else:
                pass
        finally:
            await cancel_local_tasks([self.output_reader_task], self.cleanup_timeout_s)
