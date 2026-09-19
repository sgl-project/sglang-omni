# SPDX-License-Identifier: Apache-2.0
"""Request-scoped interleaving above native model execution backends."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Protocol

from sglang_omni.admission import QueueFullError
from sglang_omni.proto.continuation import ContinuationToken, UMMSegment
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


@dataclass(frozen=True)
class UMMDecision:
    kind: Literal["final", "generate"]
    text: str = ""
    generation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("final", "generate") or not isinstance(self.text, str):
            raise ValueError("Invalid structured UMM decision")
        if self.kind == "generate" and not isinstance(self.generation, dict):
            raise ValueError("Generation decisions require a structured request")
        if self.kind == "final" and self.generation is not None:
            raise ValueError("Final decisions cannot dispatch generation")


@dataclass(frozen=True)
class UMMLimits:
    max_turns: int = 8
    max_segments: int = 16
    max_sessions: int = 32
    max_context_bytes: int = 64 * 1024 * 1024
    timeout_s: float = 3600.0

    def __post_init__(self) -> None:
        for name in ("max_turns", "max_segments", "max_sessions", "max_context_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number")


class UMMAdapter(Protocol):
    """Model-owned structured decisions and native input conversion."""

    def start(self, request: OmniRequest) -> Any: ...

    def reasoner_request(self, history: Any, request: OmniRequest) -> OmniRequest: ...

    def interpret_reasoner(self, result: Any) -> UMMDecision: ...

    def generation_request(
        self, decision: UMMDecision, request: OmniRequest
    ) -> OmniRequest: ...

    def incorporate_media(
        self, history: Any, decision: UMMDecision, result: Any
    ) -> Any: ...

    def media_segments(self, result: Any) -> list[dict[str, Any]]: ...


@dataclass
class _Session:
    session_id: str
    request: OmniRequest
    history: Any
    deadline: float
    turn_index: int = 0
    expected: ContinuationToken | None = None
    decision: UMMDecision | None = None
    segments: list[UMMSegment] = field(default_factory=list)


@dataclass(frozen=True)
class _Discarded:
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
        self._clock = clock
        self._sessions: dict[str, _Session] = {}
        self._session_lock = threading.RLock()
        super().__init__(
            self._advance,
            abort_callback=self._forget,
            shutdown_callback=self._forget_all,
        )

    @property
    def active_sessions(self) -> int:
        with self._session_lock:
            return len(self._sessions)

    def _forget(self, request_id: str) -> None:
        with self._session_lock:
            self._sessions.pop(request_id, None)

    def _forget_all(self) -> None:
        with self._session_lock:
            self._sessions.clear()

    def _expire(self) -> None:
        with self._session_lock:
            now = self._clock()
            expired = [
                request_id
                for request_id, session in self._sessions.items()
                if now >= session.deadline
            ]
            for request_id in expired:
                self._sessions.pop(request_id)
                self._emit_error(
                    request_id,
                    TimeoutError("UMM session deadline exceeded"),
                    self.outbox,
                )

    def validate_result(self, payload: StagePayload) -> bool:
        """Revalidate a queued continuation before Stage forwards native work."""
        if payload.continuation is None:
            # Completed snapshots do not retain a live session or deadline.
            return True
        with self._session_lock:
            session = self._sessions.get(payload.request_id)
            if session is None or payload.continuation != session.expected:
                return False
            if self._clock() >= session.deadline:
                self._sessions.pop(payload.request_id)
                self._emit_error(
                    payload.request_id,
                    TimeoutError("UMM session deadline exceeded"),
                    self.outbox,
                )
                return False
            return True

    def _next_message(self):
        self._expire()
        return super()._next_message()

    def _check_deadline(self, session: _Session) -> None:
        if self._clock() >= session.deadline:
            raise TimeoutError("UMM session deadline exceeded")

    def _check_size(self, session: _Session, stage_request: OmniRequest | None) -> None:
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
            raise ValueError("UMM retained context exceeds max_context_bytes")

    def _append_segment(self, session: _Session, kind: str, data: Any) -> None:
        if len(session.segments) >= self.limits.max_segments:
            raise ValueError("UMM output exceeds max_segments")
        session.segments.append(
            UMMSegment(
                session_id=session.session_id,
                segment_index=len(session.segments),
                kind=kind,
                data=copy.deepcopy(data),
            )
        )

    def _dispatch(
        self, request_id: str, session: _Session, phase: str, request: OmniRequest
    ) -> StagePayload:
        if not isinstance(request, OmniRequest):
            raise TypeError("UMM adapter must return an OmniRequest")
        self._check_size(session, request)
        self._check_deadline(session)
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

    def _advance(self, payload: StagePayload) -> StagePayload | _Discarded:
        if not isinstance(payload, StagePayload):
            raise TypeError("UMM controller requires a StagePayload")
        request_id = payload.request_id
        with self._session_lock:
            session = self._sessions.get(request_id)
            token = payload.continuation
            if token is None:
                if session is not None:
                    return _Discarded(payload.arrival_id, keep_active=True)
                if len(self._sessions) >= self.limits.max_sessions:
                    raise QueueFullError()
                deadline = self._clock() + self.limits.timeout_s
                request = copy.deepcopy(payload.request)
                session = _Session(
                    session_id=uuid.uuid4().hex,
                    request=request,
                    history=self.adapter.start(request),
                    deadline=deadline,
                )
                self._sessions[request_id] = session
            elif session is None or token != session.expected:
                return _Discarded(payload.arrival_id, keep_active=session is not None)

            first_segment = len(session.segments)
            try:
                self._check_deadline(session)
                if token is None:
                    result = self._dispatch(
                        request_id,
                        session,
                        "reasoner",
                        self.adapter.reasoner_request(session.history, session.request),
                    )
                elif token.phase == "reasoner":
                    decision = self.adapter.interpret_reasoner(payload.data)
                    if not isinstance(decision, UMMDecision):
                        raise TypeError("UMM adapter must return a UMMDecision")
                    session.expected = None
                    if decision.text:
                        self._append_segment(session, "text", decision.text)
                    if decision.kind == "final":
                        self._check_size(session, None)
                        result = StagePayload(
                            request_id=request_id,
                            request=session.request,
                            data={
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
                        self._sessions.pop(request_id)
                    else:
                        if session.turn_index >= self.limits.max_turns:
                            raise ValueError("UMM generation exceeds max_turns")
                        session.decision = decision
                        result = self._dispatch(
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
                    for segment in media_segments:
                        self._append_segment(session, segment["kind"], segment["data"])
                    session.history = self.adapter.incorporate_media(
                        session.history, session.decision, payload.data
                    )
                    session.decision = None
                    session.turn_index += 1
                    result = self._dispatch(
                        request_id,
                        session,
                        "reasoner",
                        self.adapter.reasoner_request(session.history, session.request),
                    )
                self._check_deadline(session)
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
                result.arrival_id = payload.arrival_id
                return result
            except Exception:
                self._sessions.pop(request_id, None)
                raise

    @staticmethod
    def _emit_result(request_id, result, outbox) -> None:
        if isinstance(result, _Discarded):
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
            SimpleScheduler._emit_result(request_id, result, outbox)


def route_umm(request_id: str, payload: StagePayload) -> str | None:
    """Select only the phase attested by the controller's continuation token."""
    if payload.request_id != request_id:
        raise ValueError("UMM routed request identity mismatch")
    return payload.continuation.phase if payload.continuation else None
