# SPDX-License-Identifier: Apache-2.0
"""Interleaved (UMM) requests on the existing stage graph.

A UMM request runs model phases, such as understanding, generation, and
inspection of the generated media, in turn or in parallel until its adapter
completes it. The controller is the scheduler of the entry stage, whose
route_fn sends each dispatched phase to the stage configuration maps it to.
The completed request goes to a terminal output stage in the controller's
process, which also relays accepted segments to the client in order.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal, Protocol, TypeAlias, runtime_checkable

from sglang_omni.admission import QueueFullError
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, RoutedPayloads, StagePayload
from sglang_omni.proto.session import SESSION_METADATA_KEY
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
CloseReason = Literal["completed", "failed", "aborted", "expired", "shed", "shutdown"]
# Request metadata key of the Continuation each controller output carries.
CONTINUATION_METADATA_KEY = "omni_continuation"
# A phase result reports context it produced under this data key, and the
# consuming phase finds it under this request metadata key.
CONTEXT_KEY = "umm_context"
CONTROLLER_FACTORY_PATH = f"{__name__}.create_umm_controller"
OUTPUT_FACTORY_PATH = f"{__name__}.create_umm_output_stage"
ROUTE_FN_PATH = f"{__name__}.route_umm"
STREAM_DONE_FN_PATH = f"{__name__}.umm_stream_done_targets"


@dataclass(frozen=True)
class Continuation:
    """Where one controller output goes, and which result may advance it.

    A phase stage returns its result with the request it received, so the
    continuation comes back in the request metadata. phase None marks the
    completed request on its way to the output stage.
    """

    execution_id: str
    index: int
    phase: str | None
    stage: str
    nonce: str


@dataclass(frozen=True)
class ContextRef:
    """Engine-owned context, such as KV pages, that one phase hands to the next.

    The owner stage keeps the context and releases it when the consuming phase
    finishes or the request aborts. The controller only routes this reference.
    """

    owner: str
    key: str


@dataclass(frozen=True)
class UMMOutput:
    kind: str
    data: JSONValue


@dataclass(frozen=True)
class UMMDispatch:
    """One phase a step starts, with its request and any engine context."""

    phase: str
    request: OmniRequest
    context: ContextRef | None = None


@dataclass(frozen=True)
class UMMStep:
    """The adapter's choice after start and after every phase result.

    A step starts the phase it names and every phase in dispatches, at most
    one per phase stage of the request. A step that starts none publishes its
    outputs and completes the request, or waits while phases of it run.
    """

    phase: str | None = None
    request: OmniRequest | None = None
    outputs: tuple[UMMOutput, ...] = ()
    context: ContextRef | None = None
    dispatches: tuple[UMMDispatch, ...] = ()


def limit_problems(limits: Mapping[str, object]) -> list[str]:
    """Name each controller bound that is missing or out of range."""
    problems = [
        f"{name} must be a positive integer"
        for name in (
            "max_dispatches",
            "max_outputs",
            "max_active",
            "max_retained_bytes",
        )
        if type(limits.get(name)) is not int or limits[name] <= 0
    ]
    timeout_s = limits.get("timeout_s")
    if type(timeout_s) not in (int, float) or not 0 < timeout_s < math.inf:
        problems.append("timeout_s must be a finite positive number")
    else:
        pass
    return problems


@dataclass(frozen=True, kw_only=True)
class UMMLimits:
    """Controller bounds, set by orchestrator factory arguments of the same names."""

    max_dispatches: int
    max_outputs: int
    max_active: int
    max_retained_bytes: int
    timeout_s: float

    def __post_init__(self) -> None:
        problems = limit_problems(vars(self))
        if problems:
            raise ValueError(". ".join(problems))
        else:
            pass


@dataclass(eq=False)
class UMMContext:
    """Per-request view passed to every adapter callback.

    Adapters read execution_id, request, deadline, remaining_dispatches,
    outputs, and shedding without changing them, and end a request early
    only through shed. The other fields belong to the controller.
    """

    execution_id: str
    request: OmniRequest
    deadline: float
    remaining_dispatches: int
    outputs: list[dict[str, JSONValue]] = field(default_factory=list)
    output_bytes: int = 0
    produced: ContextRef | None = None
    # The continuation of the phase each stage runs for this request.
    expected: dict[str, Continuation] = field(default_factory=dict)
    busy: bool = False
    closed: CloseReason | None = None
    shedding: bool = False

    def shed(self) -> None:
        """End this request because the deployment cannot finish it in time.

        For an adapter whose plan shows the request can no longer finish
        before its deadline, usually a waiting request other than the one it
        advances. The controller's deadline sweep settles it once no
        transition of it runs: the client gets the capacity error (HTTP 503,
        which says to retry later), the stages abort its queued and running
        phase work, and close runs once with reason shed. A request that
        completes or closes first is unaffected.
        """
        self.shedding = True


@runtime_checkable
class UMMAdapter(Protocol):
    """Model-owned decisions. Shared code never interprets phase names.

    An adapter keeps its per-request state keyed by execution_id, reports the
    bytes of that state through usage as it grows, and releases it in close,
    which the controller calls exactly once per request. start, advance, and
    usage run on the controller thread one at a time. close of an aborted
    request may run on the stage thread, never during that request's own
    transition.
    """

    def start(self, context: UMMContext) -> UMMStep: ...

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep: ...

    def usage(self, context: UMMContext) -> int: ...

    def close(self, context: UMMContext, reason: CloseReason) -> None: ...


class MonotonicClock(Protocol):
    def __call__(self) -> float: ...


class UMMRequestError(ValueError):
    """A client-caused UMM rejection, reported to the client as a bad request.

    The serving error classifier recognizes the message prefix, which
    survives stage IPC as text.
    """

    PREFIX = "Invalid UMM request: "

    def __init__(self, detail: str) -> None:
        super().__init__(self.PREFIX + detail)


class UMMDeadlineError(TimeoutError):
    """A UMM request reached its timeout_s deadline. Its close reason is expired."""


def encoded_bytes(output: UMMOutput) -> int:
    if isinstance(output.data, str):
        return len(output.data.encode("utf-8"))
    else:
        encoded = json.dumps(output.data, ensure_ascii=False, allow_nan=False)
        return len(encoded.encode("utf-8"))


class UMMController(
    SimpleScheduler[StagePayload, StagePayload | RoutedPayloads | None]
):
    """Advance each request through adapter-chosen phases.

    Native stages keep compute, batching, caches, and engine context. This
    controller keeps only per-request identity, accepted outputs, and bounds.
    """

    def __init__(
        self,
        adapter: UMMAdapter,
        *,
        limits: UMMLimits,
        phases: Mapping[str, str],
        output_stage: str,
        stage_name: str = "orchestrator",
        clock: MonotonicClock = time.monotonic,
    ) -> None:
        if not phases or not all(
            isinstance(phase, str) and isinstance(stage, str)
            for phase, stage in phases.items()
        ):
            raise ValueError("phases must map each adapter phase name to a stage")
        else:
            pass
        self.adapter = adapter
        self.limits = limits
        self.phases = dict(phases)
        self.output_stage = output_stage
        # Errors name the settings of this controller's own stage.
        self.factory_prefix = f"stages.{stage_name}.factory"
        self.clock = clock
        self.contexts: dict[str, UMMContext] = {}
        self.lock = threading.Lock()
        super().__init__(
            self.advance,
            abort_callback=self.forget,
            shutdown_callback=self.forget_all,
        )

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        try:
            self.adapter.close(context, reason)
        except Exception:
            logger.exception(f"UMM adapter close failed for {context.execution_id}")

    def release(self, request_id: str, reason: CloseReason) -> bool:
        with self.lock:
            context = self.contexts.pop(request_id, None)
            if context is None:
                return False
            elif context.busy:
                # The running transition closes it once the adapter returns.
                context.closed = reason
                return True
            else:
                pass
        self.close(context, reason)
        return True

    def forget(self, request_id: str) -> None:
        self.release(request_id, "aborted")

    def forget_all(self) -> None:
        with self.lock:
            request_ids = list(self.contexts)
        for request_id in request_ids:
            self.release(request_id, "shutdown")

    def budget_error(self, name: str, limit: int, *, client: bool) -> Exception:
        # Before the first dispatch only the client's input exists, so a bound it
        # crosses is a client error. Later breaches come from model output.
        detail = (
            f"UMM request exceeds {name}={limit}. "
            f"Raise {self.factory_prefix}.{name} to allow more"
        )
        return UMMRequestError(detail) if client else RuntimeError(detail)

    def deadline_error(self) -> UMMDeadlineError:
        return UMMDeadlineError(
            f"UMM request deadline exceeded after timeout_s={self.limits.timeout_s}. "
            f"Raise {self.factory_prefix}.timeout_s for longer requests"
        )

    def shed_error(self) -> QueueFullError:
        # Starts with QueueFullError.MESSAGE, so it classifies as a full queue.
        return QueueFullError(
            f"{QueueFullError.MESSAGE} The UMM request was ended before its "
            f"timeout_s={self.limits.timeout_s} deadline because this deployment "
            "cannot finish it in time at its current load. Retry it later, or "
            f"raise {self.factory_prefix}.timeout_s or add serving capacity"
        )

    def check_deadline(self, context: UMMContext) -> None:
        if self.clock() >= context.deadline:
            raise self.deadline_error()
        else:
            pass

    def expire(self) -> float | None:
        """Fail each waiting request that reached its deadline or was shed.

        Returns the seconds until the earliest deadline of a request left
        waiting, or None when no request waits.
        """
        now = self.clock()
        ended: list[tuple[str, UMMContext]] = []
        earliest = math.inf
        with self.lock:
            for request_id, context in self.contexts.items():
                if context.busy:
                    pass
                elif now >= context.deadline or context.shedding:
                    ended.append((request_id, context))
                elif context.deadline < earliest:
                    earliest = context.deadline
                else:
                    pass
        for request_id, context in ended:
            shed = context.shedding
            if self.release(request_id, "shed" if shed else "expired"):
                # The request waits at a phase stage, so this stage stopped
                # tracking it. Admitting it again lets the stage report the
                # failure, which aborts the phase everywhere.
                self.outbox.put(OutgoingMessage(request_id, "admitted"))
                error = self.shed_error() if shed else self.deadline_error()
                self.emit_error(request_id, error, self.outbox)
            else:
                pass
        return None if earliest == math.inf else earliest - now

    def next_message(self) -> IncomingMessage | None:
        while True:
            wait = self.expire()
            if wait is None or self.pending_messages:
                return super().next_message()
            else:
                pass
            # The earliest deadline bounds the wait, so the sweep fails that
            # request on time whether or not the scheduler loop polls. Waits
            # stop at the longest timeout the platform accepts.
            try:
                return self.inbox.get(timeout=min(wait, threading.TIMEOUT_MAX))
            except queue.Empty:
                pass

    def stop(self) -> None:
        super().stop()
        # A wait bounded by a deadline would otherwise last until that
        # deadline, so stop releases it.
        self.inbox.put(IncomingMessage("", "wakeup"))

    @staticmethod
    def emit_result(
        request_id: str,
        result: StagePayload | RoutedPayloads | None,
        outbox: queue.Queue[OutgoingMessage],
    ) -> None:
        # None is a stale continuation, an already reported rejection, or a
        # request that waits for its running phases.
        if result is not None:
            SimpleScheduler.emit_result(request_id, result, outbox)
        else:
            pass

    def admit(self, payload: StagePayload) -> UMMContext:
        if SESSION_METADATA_KEY in payload.request.metadata:
            raise UMMRequestError(
                "UMM pipelines do not support session operations. "
                "Submit ordinary requests instead"
            )
        elif len(self.contexts) >= self.limits.max_active:
            raise QueueFullError()
        else:
            pass
        context = UMMContext(
            execution_id=uuid.uuid4().hex,
            request=payload.request,
            deadline=self.clock() + self.limits.timeout_s,
            remaining_dispatches=self.limits.max_dispatches,
        )
        self.contexts[payload.request_id] = context
        return context

    def advance(self, payload: StagePayload) -> StagePayload | RoutedPayloads | None:
        try:
            return self.transition(payload)
        except (QueueFullError, UMMRequestError) as exc:
            # Rejections of client requests are expected and need no traceback.
            self.emit_error(payload.request_id, exc, self.outbox)
            return None

    def transition(self, payload: StagePayload) -> StagePayload | RoutedPayloads | None:
        request_id = payload.request_id
        token = payload.request.metadata.get(CONTINUATION_METADATA_KEY)
        if not isinstance(token, Continuation):
            token = None
        else:
            pass
        phase = None
        lost = False
        with self.lock:
            context = self.contexts.get(request_id)
            if context is None and token is None:
                context = self.admit(payload)
            elif context is None:
                raise RuntimeError(
                    f"UMM result for inactive request {request_id} was discarded"
                )
            elif token is None:
                self.contexts.pop(request_id)
                lost = True
            elif token != context.expected.get(token.stage):
                # A duplicate or stale continuation never advances the request.
                return None
            else:
                phase = token.phase
                del context.expected[token.stage]
            if not lost:
                context.busy = True
            else:
                pass
        if lost:
            self.close(context, "failed")
            raise RuntimeError(
                f"A phase result for live UMM request {request_id} lost its "
                "continuation. Phase stages must return the request they received"
            )
        else:
            pass
        try:
            self.check_deadline(context)
            if phase is None:
                step = self.adapter.start(context)
            else:
                data = payload.data
                context.produced = (
                    data.get(CONTEXT_KEY) if isinstance(data, Mapping) else None
                )
                step = self.adapter.advance(context, phase, data)
            result, added, started = self.apply(
                request_id, context, step, client=phase is None
            )
            self.check_deadline(context)
        except BaseException as exc:
            with self.lock:
                context.busy = False
                if self.contexts.get(request_id) is context:
                    self.contexts.pop(request_id)
                else:
                    pass
            reason: CloseReason = (
                "expired" if isinstance(exc, UMMDeadlineError) else "failed"
            )
            self.close(context, context.closed or reason)
            raise
        with self.lock:
            context.busy = False
            live = self.contexts.get(request_id) is context
            completed = live and not started and not context.expected
            if completed:
                self.contexts.pop(request_id)
            elif live:
                context.expected.update(started)
            else:
                pass
        if not live:
            self.close(context, context.closed or "aborted")
            return None
        else:
            pass
        # Publish only after the whole transition validated. The serial
        # scheduler queues these before its routed result.
        if context.request.params.get("stream", False):
            for segment in added:
                self.outbox.put(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=segment,
                        metadata={"modality": segment["kind"]},
                    )
                )
        else:
            pass
        if completed:
            self.close(context, "completed")
        else:
            pass
        return result

    def apply(
        self, request_id: str, context: UMMContext, step: UMMStep, *, client: bool
    ) -> tuple[
        StagePayload | RoutedPayloads | None,
        list[dict[str, JSONValue]],
        dict[str, Continuation],
    ]:
        limits = self.limits
        if not isinstance(step, UMMStep):
            raise TypeError("UMM adapter must return a UMMStep")
        else:
            pass
        offered = [step.context, *(dispatch.context for dispatch in step.dispatches)]
        if context.produced is not None and context.produced not in offered:
            raise RuntimeError(
                f"UMM adapter left context from stage {context.produced.owner!r} "
                "unconsumed. Pass it to the next phase so its owner releases it"
            )
        elif len(context.outputs) + len(step.outputs) > limits.max_outputs:
            raise self.budget_error("max_outputs", limits.max_outputs, client=client)
        else:
            pass
        retained = context.output_bytes + sum(
            encoded_bytes(output) for output in step.outputs
        )
        state_bytes = self.adapter.usage(context)
        if type(state_bytes) is not int or state_bytes < 0:
            raise TypeError("UMM adapter usage must return a byte count")
        elif retained + state_bytes > limits.max_retained_bytes:
            raise self.budget_error(
                "max_retained_bytes", limits.max_retained_bytes, client=client
            )
        elif step.phase is None and (
            step.request is not None or step.context is not None
        ):
            raise ValueError("A UMM step without a phase takes no request or context")
        else:
            pass
        dispatches = (
            (UMMDispatch(step.phase, step.request, step.context), *step.dispatches)
            if step.phase is not None
            else step.dispatches
        )
        stages: list[str] = []
        for dispatch in dispatches:
            if dispatch.request is None:
                raise ValueError(
                    f"UMM step for phase {dispatch.phase!r} has no request"
                )
            elif len(stages) >= context.remaining_dispatches:
                raise self.budget_error(
                    "max_dispatches", limits.max_dispatches, client=False
                )
            elif dispatch.phase not in self.phases:
                raise ValueError(
                    f"UMM adapter chose unmapped phase {dispatch.phase!r}. "
                    f"Map it in {self.factory_prefix}.phases"
                )
            elif self.phases[dispatch.phase] in (*context.expected, *stages):
                # A stage keys its request state by request id.
                raise ValueError(
                    f"UMM adapter dispatched phase {dispatch.phase!r} to stage "
                    f"{self.phases[dispatch.phase]!r}, which already runs a phase "
                    "of this request. Dispatch it once advance reports that "
                    "stage's result"
                )
            else:
                stages.append(self.phases[dispatch.phase])
        added: list[dict[str, JSONValue]] = [
            {
                "type": "segment",
                "execution_id": context.execution_id,
                "segment_index": len(context.outputs) + offset,
                "kind": output.kind,
                "data": output.data,
            }
            for offset, output in enumerate(step.outputs)
        ]
        context.outputs.extend(added)
        context.output_bytes = retained
        context.produced = None
        if not dispatches and context.expected:
            # The request waits for the results of its running phases.
            return None, added, {}
        elif not dispatches:
            streamed = bool(context.request.params.get("stream", False))
            continuation = Continuation(
                execution_id=context.execution_id,
                index=limits.max_dispatches - context.remaining_dispatches,
                phase=None,
                stage=self.output_stage,
                nonce=uuid.uuid4().hex,
            )
            request = replace(
                context.request,
                metadata={
                    **context.request.metadata,
                    CONTINUATION_METADATA_KEY: continuation,
                },
            )
            result = {
                "type": "umm_result",
                "execution_id": context.execution_id,
                # A streamed request already received each segment with its data.
                "segments": [
                    (
                        {
                            name: value
                            for name, value in segment.items()
                            if name != "data"
                        }
                        if streamed
                        else segment
                    )
                    for segment in context.outputs
                ],
                "finish_reason": "stop",
            }
            return StagePayload(request_id, request, result), added, {}
        else:
            pass
        payloads: dict[str, StagePayload] = {}
        started: dict[str, Continuation] = {}
        for dispatch, stage in zip(dispatches, stages, strict=True):
            continuation = Continuation(
                execution_id=context.execution_id,
                index=limits.max_dispatches - context.remaining_dispatches,
                phase=dispatch.phase,
                stage=stage,
                nonce=uuid.uuid4().hex,
            )
            context.remaining_dispatches -= 1
            metadata = {
                **dispatch.request.metadata,
                CONTINUATION_METADATA_KEY: continuation,
            }
            if dispatch.context is not None:
                metadata[CONTEXT_KEY] = dispatch.context
            else:
                pass
            # Phases never stream. The controller publishes accepted outputs.
            request = replace(
                dispatch.request,
                params={**dispatch.request.params, "stream": False},
                metadata=metadata,
            )
            payloads[stage] = StagePayload(request_id, request, request.inputs)
            started[stage] = continuation
        if len(payloads) == 1 and not context.expected:
            (result,) = payloads.values()
        else:
            # One routing pass sends every payload with the request's replica
            # bindings. Another running phase may return before the stage
            # routes this result, so the stage keeps the request's state.
            result = RoutedPayloads(payloads, keep_request_state=True)
        return result, added, started


class UMMOutputRelay:
    """Scheduler of the UMM output stage.

    Forwards accepted segments to the client as they arrive and completes the
    request with the final payload. It keeps no per-request state, so aborts
    and failures leave nothing to release here.
    """

    def __init__(self) -> None:
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.running = False

    def warm_up_serving_thread(self) -> None:
        """Keep the scheduler contract. The relay has nothing to warm."""

    def start(self) -> None:
        self.running = True
        while self.running:
            try:
                message = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.type == "stream_chunk":
                item = message.data
                assert isinstance(item, StreamItem), "stages deliver StreamItem chunks"
                self.outbox.put(
                    OutgoingMessage(
                        request_id=message.request_id,
                        type="stream",
                        data=item.data,
                        metadata=item.metadata,
                    )
                )
            elif message.type == "new_request":
                self.outbox.put(
                    OutgoingMessage(
                        request_id=message.request_id,
                        type="result",
                        data=message.data,
                    )
                )
            else:
                pass  # A stream end carries nothing to relay.

    def stop(self) -> None:
        self.running = False

    def abort(self, request_id: str) -> None:
        """Keep the scheduler contract. The relay holds nothing per request."""


def create_umm_controller(
    model_path: str,
    *,
    adapter: str,
    phases: Mapping[str, str],
    output_stage: str,
    max_dispatches: int,
    max_outputs: int,
    max_active: int,
    max_retained_bytes: int,
    timeout_s: float,
    stage_name: str = "orchestrator",
) -> UMMController:
    """Build the controller around the adapter of the model being served.

    adapter is the import path of a callable that takes model_path and
    returns the model's UMMAdapter. stage_name is the controller's stage,
    which UMMPipelineConfig passes so that errors name its settings.
    """
    adapter_factory = import_string(adapter)
    if not callable(adapter_factory):
        raise TypeError(f"UMM adapter factory {adapter!r} is not callable")
    else:
        pass
    model_adapter = adapter_factory(model_path)
    if not isinstance(model_adapter, UMMAdapter):
        raise TypeError(f"{adapter!r} did not return a UMMAdapter")
    else:
        pass
    return UMMController(
        model_adapter,
        limits=UMMLimits(
            max_dispatches=max_dispatches,
            max_outputs=max_outputs,
            max_active=max_active,
            max_retained_bytes=max_retained_bytes,
            timeout_s=timeout_s,
        ),
        phases=phases,
        output_stage=output_stage,
        stage_name=stage_name,
    )


def create_umm_output_stage() -> UMMOutputRelay:
    return UMMOutputRelay()


def route_umm(
    request_id: str, payload: StagePayload | RoutedPayloads
) -> str | list[str]:
    """Send a controller output to the stage its continuation names, or each
    payload of a routed result to its own stage."""
    if isinstance(payload, RoutedPayloads):
        return list(payload.payloads)
    else:
        pass
    continuation = payload.request.metadata.get(CONTINUATION_METADATA_KEY)
    if not isinstance(continuation, Continuation):
        raise TypeError(f"UMM output for {request_id} carries no continuation")
    else:
        pass
    return continuation.stage


def umm_stream_done_targets(request_id: str, payload: StagePayload) -> None:
    """Keep the stream_done_to_fn contract and send no stream end.

    The output stage forwards each segment as it arrives and completes on
    the final payload, so no controller visit ends a stream.
    """


class UMMPipelineConfig(PipelineConfig):
    """Base of pipelines whose entry stage is a UMM controller.

    The controller stage uses create_umm_controller, route_fn route_umm,
    stream_done_to_fn umm_stream_done_targets, stream_to the output stage,
    and next naming the output stage and every phase stage, whose own next is
    the controller and which streams nowhere. The output stage is terminal,
    uses create_umm_output_stage, accepts streams before its payload, and
    shares the controller's process.
    """

    umm_controller_stage: ClassVar[str] = "orchestrator"
    umm_output_stage: ClassVar[str] = "output"
    interleaved_output: ClassVar[bool] = True

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Segments reach the output stage as local objects.
        return frozenset({(cls.umm_controller_stage, cls.umm_output_stage)})

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == type(self).umm_controller_stage:
            return {
                "output_stage": type(self).umm_output_stage,
                "stage_name": stage_name,
            }
        else:
            return super().stage_factory_kwargs(stage_name)

    def entry_stage_admission_limit(self) -> int:
        controller = self.stage_named(type(self).umm_controller_stage)
        max_active = resolve_stage_typed_kwargs(controller)["max_active"]
        assert (
            type(max_active) is int
        ), "final configuration validation checks max_active"
        return max_active

    def check_local_conflicts(self) -> None:
        super().check_local_conflicts()
        cls = type(self)
        controller_name = cls.umm_controller_stage
        output_name = cls.umm_output_stage
        stages = {stage.name: stage for stage in self.stages}
        controller = stages.get(controller_name)
        output = stages.get(output_name)
        if controller is None or output is None:
            raise ValueError(
                f"UMM pipeline needs stages {controller_name!r} and {output_name!r}"
            )
        else:
            pass
        factory_args = resolve_stage_typed_kwargs(controller)
        phases = factory_args.get("phases")
        phase_targets = list(phases.values()) if isinstance(phases, Mapping) else []
        phase_stages = {name for name in phase_targets if isinstance(name, str)}
        streaming_phase_stages = sorted(
            name for name in phase_stages if name in stages and stages[name].stream_to
        )
        if isinstance(controller.next, list):
            next_stages = set(controller.next)
        else:
            next_stages = {controller.next}
        prefix = f"stages.{controller_name}"
        problems = [
            message
            for holds, message in (
                (
                    self.resolved_entry_stage == controller_name,
                    f"entry_stage must be {controller_name!r}",
                ),
                (
                    controller.factory_path == CONTROLLER_FACTORY_PATH,
                    f"{prefix}.factory_path must be {CONTROLLER_FACTORY_PATH}",
                ),
                (
                    controller.route_fn == ROUTE_FN_PATH,
                    f"{prefix}.route_fn must be {ROUTE_FN_PATH}",
                ),
                (
                    controller.stream_done_to_fn == STREAM_DONE_FN_PATH,
                    f"{prefix}.stream_done_to_fn must be {STREAM_DONE_FN_PATH}",
                ),
                (
                    controller.stream_to == [output_name],
                    f"{prefix}.stream_to must be [{output_name!r}]",
                ),
                (
                    bool(phase_targets)
                    and all(
                        isinstance(name, str)
                        and name in stages
                        and name not in (controller_name, output_name)
                        and stages[name].next in (controller_name, [controller_name])
                        for name in phase_targets
                    ),
                    f"{prefix}.factory.phases must map each phase to a stage "
                    f"whose next is {controller_name!r}",
                ),
                (
                    {output_name, *phase_stages} <= next_stages,
                    f"{prefix}.next must name {output_name!r} and every stage in "
                    "factory.phases",
                ),
                (
                    not streaming_phase_stages,
                    ", ".join(
                        f"stages.{name}.stream_to" for name in streaming_phase_stages
                    )
                    + " must be empty, since the controller streams accepted outputs",
                ),
                (
                    factory_args.get("output_stage", output_name) == output_name,
                    f"{prefix}.factory.output_stage must stay unset",
                ),
                (
                    output.terminal
                    and output.factory_path == OUTPUT_FACTORY_PATH
                    and output.can_accept_stream_before_payload,
                    f"stages.{output_name} must be terminal, use "
                    f"{OUTPUT_FACTORY_PATH}, and accept streams before its payload",
                ),
            )
            if not holds
        ] + [f"{prefix}.factory.{problem}" for problem in limit_problems(factory_args)]
        if problems:
            raise ValueError("Invalid UMM pipeline. " + ". ".join(problems))
        else:
            pass
