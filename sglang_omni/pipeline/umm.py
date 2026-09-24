# SPDX-License-Identifier: Apache-2.0
"""Request-scoped interleaving above native model execution backends."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Literal, Protocol, TypeAlias, TypedDict

from sglang_omni.admission import InvalidRequestError, QueueFullError
from sglang_omni.proto.continuation import ContinuationToken, UMMSegment
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SESSION_METADATA_KEY
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


class UMMMediaSegment(TypedDict):
    kind: Literal["image", "video", "audio", "action"]
    data: dict[str, JSONValue]


@dataclass(frozen=True)
class UMMDecision:
    kind: Literal["final", "generate"]
    text: str = ""
    generation: dict[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("final", "generate") or not isinstance(self.text, str):
            raise ValueError("Invalid structured UMM decision")
        else:
            pass
        if self.kind == "generate" and not isinstance(self.generation, dict):
            raise ValueError("Generation decisions require a structured request")
        else:
            pass
        if self.kind == "final" and self.generation is not None:
            raise ValueError("Final decisions cannot dispatch generation")
        else:
            pass


@dataclass(frozen=True)
class UMMLimits:
    """Controller bounds, set by orchestrator factory arguments of the same names."""

    max_turns: int = 8
    # None allows one message and one media segment per turn plus the final text.
    max_segments: int | None = None
    max_sessions: int = 32
    max_context_bytes: int = 64 * 1024 * 1024
    timeout_s: float = 3600.0

    def __post_init__(self) -> None:
        if self.max_segments is None and type(self.max_turns) is int:
            object.__setattr__(self, "max_segments", 2 * self.max_turns + 1)
        else:
            pass
        for name in ("max_turns", "max_segments", "max_sessions", "max_context_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            else:
                pass
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number")
        else:
            pass


def budget_error(name: str, limit: int) -> InvalidRequestError:
    return InvalidRequestError(
        f"UMM request exceeds {name}={limit}. "
        f"Raise stages.<orchestrator>.factory.{name} to allow more"
    )


class UMMAdapter(Protocol):
    """Model-owned structured decisions and native input conversion.

    The orchestrator stage is terminal, runs UMMController with route_umm
    and declares next as exactly the stages named reasoner and generation.
    Both route back to it with the input StagePayload, or a result that
    copies its continuation.
    Internal turns never stream, because the controller streams accepted
    segments. Generation returns inline media rather than stage-local files,
    because the controller retains media after the stage releases its outputs.
    For each generation result the controller calls media_segments before
    incorporate_media, so an adapter may validate media content once there.
    Keep PipelineConfig.max_in_flight at or below UMMLimits.max_sessions so the
    coordinator refuses excess requests before they reach the controller.
    """

    def start(self, request: OmniRequest) -> JSONValue:
        """Return the initial history that the controller retains as JSON."""

    def reasoner_request(
        self,
        history: JSONValue,
        request: OmniRequest,
        *,
        remaining_generation_turns: int,
    ) -> OmniRequest:
        """Build a reasoner turn that allows only a final decision at 0 turns."""

    def interpret_reasoner(self, result: object) -> UMMDecision: ...

    def generation_request(
        self, decision: UMMDecision, request: OmniRequest
    ) -> OmniRequest: ...

    def incorporate_media(
        self, history: JSONValue, decision: UMMDecision, result: object
    ) -> JSONValue: ...

    def media_segments(self, result: object) -> list[UMMMediaSegment]: ...


@dataclass
class Session:
    session_id: str
    request: OmniRequest
    history: JSONValue
    deadline: float
    turn_index: int = 0
    expected: ContinuationToken | None = None
    decision: UMMDecision | None = None
    segments: list[UMMSegment] = field(default_factory=list)


@dataclass(frozen=True)
class Discarded:
    arrival_id: object | None
    keep_active: bool


class UMMController(SimpleScheduler):
    """Route bounded model turns using the existing stage scheduler contract.

    Native stages retain compute, batching, cache and media-file ownership.
    This controller only owns conversation state and accepted output segments.
    """

    def __init__(
        self,
        adapter: UMMAdapter,
        *,
        limits: UMMLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.adapter = adapter
        self.limits = limits or UMMLimits()
        self.clock = clock
        self.sessions: dict[str, Session] = {}
        self.session_lock = threading.RLock()
        super().__init__(
            self.advance,
            abort_callback=self.forget,
            shutdown_callback=self.forget_all,
        )

    @property
    def active_sessions(self) -> int:
        with self.session_lock:
            return len(self.sessions)

    def forget(self, request_id: str) -> None:
        with self.session_lock:
            self.sessions.pop(request_id, None)

    def forget_all(self) -> None:
        with self.session_lock:
            self.sessions.clear()

    def deadline_error(self) -> TimeoutError:
        return TimeoutError(
            f"UMM session deadline exceeded after timeout_s={self.limits.timeout_s}. "
            "Raise stages.<orchestrator>.factory.timeout_s for longer requests"
        )

    def expire(self) -> None:
        with self.session_lock:
            now = self.clock()
            expired = [
                request_id
                for request_id, session in self.sessions.items()
                if now >= session.deadline
            ]
            for request_id in expired:
                self.sessions.pop(request_id)
                self.emit_error(request_id, self.deadline_error(), self.outbox)

    def validate_result(self, payload: StagePayload) -> bool:
        """Revalidate a queued continuation before Stage forwards native work."""
        if payload.continuation is None:
            # Completed snapshots do not retain a live session or deadline.
            return True
        else:
            pass
        with self.session_lock:
            session = self.sessions.get(payload.request_id)
            if session is None or payload.continuation != session.expected:
                return False
            else:
                pass
            if self.clock() >= session.deadline:
                self.sessions.pop(payload.request_id)
                self.emit_error(payload.request_id, self.deadline_error(), self.outbox)
                return False
            else:
                pass
            return True

    def next_message(self):
        self.expire()
        return super().next_message()

    def check_deadline(self, session: Session) -> None:
        if self.clock() >= session.deadline:
            raise self.deadline_error()
        else:
            pass

    def check_size(self, session: Session, stage_request: OmniRequest | None) -> None:
        state = {
            "request": session.request.to_dict(),
            "history": session.history,
            "decision": asdict(session.decision) if session.decision else None,
            "segments": [segment.to_dict() for segment in session.segments],
            "stage_request": stage_request.to_dict() if stage_request else None,
        }
        # Measure the complete retained JSON representation, including repeated
        # inline references. A byte budget is stronger than a media item limit.
        size = len(
            json.dumps(state, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        if size > self.limits.max_context_bytes:
            raise budget_error("max_context_bytes", self.limits.max_context_bytes)
        else:
            pass

    def append_segment(self, session: Session, kind: str, data: Any) -> None:
        if len(session.segments) >= self.limits.max_segments:
            raise budget_error("max_segments", self.limits.max_segments)
        else:
            pass
        session.segments.append(
            UMMSegment(
                session_id=session.session_id,
                segment_index=len(session.segments),
                kind=kind,
                data=copy.deepcopy(data),
            )
        )

    def dispatch(
        self, request_id: str, session: Session, phase: str, request: OmniRequest
    ) -> StagePayload:
        if not isinstance(request, OmniRequest):
            raise TypeError("UMM adapter must return an OmniRequest")
        else:
            pass
        # The controller streams accepted segments, so internal turns never do.
        request = replace(request, params={**request.params, "stream": False})
        self.check_size(session, request)
        self.check_deadline(session)
        session.expected = ContinuationToken(
            session_id=session.session_id,
            turn_index=session.turn_index,
            phase=phase,
            nonce=uuid.uuid4().hex,
        )
        return StagePayload(
            request_id=request_id,
            request=request,
            data=request.inputs,
            continuation=session.expected,
        )

    def dispatch_reasoner(self, request_id: str, session: Session) -> StagePayload:
        return self.dispatch(
            request_id,
            session,
            "reasoner",
            self.adapter.reasoner_request(
                session.history,
                session.request,
                remaining_generation_turns=self.limits.max_turns - session.turn_index,
            ),
        )

    def advance(self, payload: StagePayload) -> StagePayload | Discarded:
        if not isinstance(payload, StagePayload):
            raise TypeError("UMM controller requires a StagePayload")
        else:
            pass
        request_id = payload.request_id
        with self.session_lock:
            session = self.sessions.get(request_id)
            token = payload.continuation
            if token is None:
                if session is not None:
                    self.sessions.pop(request_id)
                    raise RuntimeError(
                        "A stage result for a live UMM session dropped its "
                        "continuation. Return the input StagePayload or copy "
                        "payload.continuation into the result"
                    )
                else:
                    pass
                if SESSION_METADATA_KEY in payload.request.metadata:
                    raise InvalidRequestError(
                        "UMM pipelines do not support session operations. "
                        "Submit ordinary requests instead"
                    )
                else:
                    pass
                if len(self.sessions) >= self.limits.max_sessions:
                    raise QueueFullError()
                else:
                    pass
                deadline = self.clock() + self.limits.timeout_s
                request = copy.deepcopy(payload.request)
                session = Session(
                    session_id=uuid.uuid4().hex,
                    request=request,
                    history=self.adapter.start(request),
                    deadline=deadline,
                )
                self.sessions[request_id] = session
            elif session is None or token != session.expected:
                return Discarded(payload.arrival_id, keep_active=session is not None)
            else:
                pass

            first_segment = len(session.segments)
            try:
                self.check_deadline(session)
                if token is None:
                    result = self.dispatch_reasoner(request_id, session)
                elif token.phase == "reasoner":
                    decision = self.adapter.interpret_reasoner(payload.data)
                    if not isinstance(decision, UMMDecision):
                        raise TypeError("UMM adapter must return a UMMDecision")
                    else:
                        pass
                    session.expected = None
                    if decision.text:
                        self.append_segment(session, "text", decision.text)
                    else:
                        pass
                    if decision.kind == "final":
                        self.check_size(session, None)
                        result = StagePayload(
                            request_id=request_id,
                            request=session.request,
                            data={
                                "type": "umm_result",
                                "session_id": session.session_id,
                                "text": "".join(
                                    segment.data
                                    for segment in session.segments
                                    if segment.kind == "text"
                                ),
                                "segments": [
                                    segment.to_dict() for segment in session.segments
                                ],
                                "finish_reason": "stop",
                            },
                        )
                        self.sessions.pop(request_id)
                    else:
                        if session.turn_index >= self.limits.max_turns:
                            raise budget_error("max_turns", self.limits.max_turns)
                        else:
                            pass
                        session.decision = decision
                        result = self.dispatch(
                            request_id,
                            session,
                            "generation",
                            self.adapter.generation_request(decision, session.request),
                        )
                else:
                    assert session.decision is not None
                    media_segments = self.adapter.media_segments(payload.data)
                    if not isinstance(media_segments, list) or not media_segments:
                        raise ValueError("UMM generation returned no media segments")
                    else:
                        pass
                    for segment in media_segments:
                        self.append_segment(session, segment["kind"], segment["data"])
                    session.history = self.adapter.incorporate_media(
                        session.history, session.decision, payload.data
                    )
                    session.decision = None
                    session.turn_index += 1
                    result = self.dispatch_reasoner(request_id, session)
                self.check_deadline(session)
                # Publish only after validation of the entire transition. The
                # serial scheduler queues these before its routed result.
                if session.request.params.get("stream", False):
                    for segment in session.segments[first_segment:]:
                        self.outbox.put(
                            OutgoingMessage(
                                request_id=request_id,
                                type="stream",
                                data=segment.to_dict(),
                                metadata={"modality": segment.kind},
                            )
                        )
                else:
                    pass
                result.arrival_id = payload.arrival_id
                return result
            except Exception:
                self.sessions.pop(request_id, None)
                raise

    @staticmethod
    def emit_result(request_id, result, outbox) -> None:
        if isinstance(result, Discarded):
            outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="discard",
                    metadata={
                        "arrival_id": result.arrival_id,
                        "keep_active": result.keep_active,
                    },
                )
            )
        else:
            SimpleScheduler.emit_result(request_id, result, outbox)


def route_umm(request_id: str, payload: StagePayload) -> str | None:
    """Select only the phase attested by the controller's continuation token."""
    if payload.request_id != request_id:
        raise ValueError("UMM routed request identity mismatch")
    else:
        pass
    return payload.continuation.phase if payload.continuation else None
