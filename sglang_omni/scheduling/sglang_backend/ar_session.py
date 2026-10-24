# SPDX-License-Identifier: Apache-2.0
"""Bridge pipeline units to streaming sessions."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import torch
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.session.session_controller import SessionController

from sglang_omni.admission import (
    REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV,
    QueueFullError,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SessionIdentity,
    SessionOperation,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
)
from sglang_omni.scheduling.types import RequestOutput

SESSION_STRING_LENGTH_LIMIT_CHARACTERS = 0


class SessionKV(Protocol):
    kv_allocated_len: int
    holds_kv: bool


class SessionTreeCache(Protocol):
    def session_records(self) -> Mapping[str, SessionKV]:
        pass

    def evictable_size(self) -> int:
        pass


class RequestToTokenPool(Protocol):
    free_slots: list[int]


class TokenToKVPoolAllocator(Protocol):
    def available_size(self) -> int:
        pass


class ModelVocabulary(Protocol):
    vocab_size: int


class BridgeScheduler(Protocol):
    """The scheduler surface the bridge drives; OmniScheduler satisfies it."""

    session_controller: SessionController
    tree_cache: SessionTreeCache
    req_to_token_pool: RequestToTokenPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    model_config: ModelVocabulary
    waiting_queue: list[Req]
    chunked_req: Req | None
    max_running_requests: int

    def abort(self, request_id: str) -> None:
        pass

    def release_request_kv_cache(self, req: Req) -> None:
        pass

    def run_abort_callback(self, request_id: str) -> None:
        pass

    def synchronize_launched_decode(self) -> None:
        pass

    def resolve_pending_async(self) -> None:
        pass


def is_close_request(payload: StagePayload) -> bool:
    """Return whether the payload carries a session close."""
    operation_metadata = find_session_operation(payload.request.metadata)
    if operation_metadata is None:
        return False
    else:
        return operation_metadata.operation == "close"


@dataclass(frozen=True, kw_only=True)
class ARSessionPreparation:
    reset_history: bool = False
    bypass_generation: bool = False


class ARSessionAdapter:
    """Convert unit inputs and outputs without mutating streaming session state."""

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        """Initialize auxiliary model history after the streaming session is created."""

    def close(self, session_identity: SessionIdentity) -> None:
        """Release auxiliary history after streaming session work and KV are released."""

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        """Handle a control event; an adapter without control semantics lets it pass."""

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        """Select a fresh model turn or relay a unit that needs no generation."""
        return ARSessionPreparation()

    def complete_without_generation(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> StagePayload:
        """The result of a unit its model cannot generate for."""
        return payload

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload | None:
        """Return a relayed EOS payload, or None to use normal generation."""
        return None

    def bind_request(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        unit: "SessionUnit",
    ) -> None:
        """Attach model-specific inputs to the native request once retained turns precede the unit's ids."""

    def on_abort(self, session_identity: SessionIdentity, unit: "SessionUnit") -> None:
        """Take what an aborted unit generated before its native request is dropped."""

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        raise NotImplementedError

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        raise NotImplementedError

    def stream(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        output: RequestOutput,
    ) -> Iterable[TimedChunk]:
        return ()

    def flush(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> Iterable[TimedChunk]:
        return ()


@dataclass(kw_only=True)
class SessionUnit:
    request_id: str
    session_identity: SessionIdentity
    stages: tuple[str, ...]
    chunk: TimedChunk
    # Note (chenyang):
    # There are multiple requests in SGLang Omni. Let's make a distinction between them.
    # payload.request is the user's OmniRequest: inputs, params, and metadata. It is not queued.
    # The session scheduler creates session_request with Session.create_req and queues that same
    # SGLang Req. create_req builds it from token ids produced from the OmniRequest, joins tokens
    # from earlier turns, and binds it to the streaming session.
    session_request: Req | None = None
    # Note (Junnan Li): A deferred payload passes through routing again; the unit is prepared once.
    is_prepared: bool = False
    is_enqueued: bool = False
    # note (Junnan Li): Only completed units commit their spans to retained history.
    embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    native_offset: int = 0
    prompt_length: int = 0


@dataclass(kw_only=True)
class BridgeSession:
    session_identity: SessionIdentity
    unit: SessionUnit | None = None
    # note (Junnan Li): KV rebuilds need embeddings from every completed unit.
    embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    # The session's read; an AR stage emits only inside appends, so the close answers it.
    output_read: StagePayload | None = None


@dataclass(kw_only=True)
class SessionAdmission:
    """Capacity derived from live requests for one host-only admission pass."""

    records: Mapping[str, SessionKV] | None = None
    demands: dict[str, tuple[int, int]] = field(default_factory=dict)
    request_rows: int = 0
    kv_tokens: int = 0

    def include(self, unit: SessionUnit) -> None:
        request = unit.session_request
        if request is None:
            return
        else:
            assert self.records is not None
        rows = int(
            not request.kv.holds_kv and unit.session_identity.id not in self.records
        )
        tokens = max(
            0,
            len(request.origin_input_ids)
            + int(request.sampling_params.max_new_tokens or 0)
            - request.kv.kv_allocated_len,
        )
        previous_rows, previous_tokens = self.demands.get(unit.request_id, (0, 0))
        self.request_rows += rows - previous_rows
        self.kv_tokens += tokens - previous_tokens
        self.demands[unit.request_id] = rows, tokens


class ARSessionBridge:
    """Map pipeline open, append, and close onto one SGLang streaming session."""

    def __init__(
        self, bridge_scheduler: BridgeScheduler, adapter: ARSessionAdapter
    ) -> None:
        self.bridge_scheduler = bridge_scheduler
        self.adapter = adapter
        self.sessions: dict[str, BridgeSession] = {}
        self.units_by_request_id: dict[str, SessionUnit] = {}
        self.cancelling_request_id: str | None = None
        self.admission: SessionAdmission | None = None

    @contextmanager
    def admission_batch(self) -> Iterator[None]:
        self.admission = SessionAdmission()
        try:
            yield
        finally:
            self.admission = None

    def admission_snapshot(self) -> SessionAdmission:
        admission = self.admission
        if admission is None:
            admission = SessionAdmission()
        else:
            admission.demands.clear()
            admission.request_rows = 0
            admission.kv_tokens = 0
        admission.records = self.bridge_scheduler.tree_cache.session_records()
        for unit in self.units_by_request_id.values():
            admission.include(unit)
        return admission

    def invalidate_admission(self) -> None:
        if self.admission is not None:
            self.admission.records = None
        else:
            pass

    def drain(self) -> None:
        self.invalidate_admission()
        self.bridge_scheduler.synchronize_launched_decode()
        self.bridge_scheduler.resolve_pending_async()

    def apply_operation(
        self, payload: StagePayload, operation: SessionOperation
    ) -> list[StagePayload]:
        """Apply an operation other than append; returns the operations it answers, none for a read the close answers."""
        session_identity = operation.session_identity
        operation_kind = operation.operation
        session = self.sessions.get(session_identity.id)
        if (
            session is not None
            and session_identity.open_index != session.session_identity.open_index
        ):
            raise ValueError("stale session open index")
        elif operation_kind == "open":
            if session is not None:
                raise ValueError("session already opened")
            elif len(self.sessions) >= self.bridge_scheduler.max_running_requests:
                raise QueueFullError()
            elif self.open_streaming_session(session_identity.id):
                self.sessions[session_identity.id] = BridgeSession(
                    session_identity=session_identity
                )
                self.adapter.open(session_identity, payload.request)
                payload.data = {"opened": True}
                return [payload]
            else:
                raise ValueError("streaming session open failed")
        elif operation_kind == "close":
            payload.data = {"closed": True}
            if session is not None:
                return [*self.close_session(session), payload]
            else:
                # note (chenyang): coding style, each if should be closed by else.
                return [payload]
        elif session is None:
            raise ValueError("unknown session open index")
        elif operation_kind == "read":
            if session.output_read is not None:
                raise ValueError("session already has an output reader")
            else:
                session.output_read = payload
                return []
        elif operation_kind == "control":
            assert operation.chunk is not None, "control operation carries no event"
            if operation.should_preempt and session.unit is not None:
                self.cancel(session.unit.request_id)
            else:
                pass
            self.adapter.control(session_identity, operation.chunk)
            payload.data = {"controlled": True}
            return [payload]
        else:
            raise ValueError("unknown session operation")

    def accept(self, payload: StagePayload, operation: SessionOperation) -> SessionUnit:
        chunk = operation.chunk
        assert (
            chunk is not None
        ), f"session append {payload.request_id} requires a chunk"
        session_identity = operation.session_identity
        session = self.sessions.get(session_identity.id)
        if session is None or session_identity != session.session_identity:
            raise ValueError("unknown or stale session open index")
        else:
            if session.unit is None:
                session.unit = SessionUnit(
                    request_id=payload.request_id,
                    session_identity=session_identity,
                    stages=operation.stages,
                    chunk=chunk,
                )
                self.units_by_request_id[payload.request_id] = session.unit
                return session.unit
            elif session.unit.request_id != payload.request_id:
                raise ValueError("session already has an active request")
            else:
                return session.unit

    def prepare_unit(self, unit: SessionUnit, payload: StagePayload) -> bool:
        if unit.is_prepared:
            return False
        else:
            unit.is_prepared = True
        preparation = self.adapter.prepare_unit(
            unit.session_identity, unit.chunk, payload
        )
        if preparation.reset_history:
            session_id = unit.session_identity.id
            self.sessions[session_id].embedding_spans.clear()
            self.release_streaming_session(session_id)
            if not self.open_streaming_session(session_id):
                raise RuntimeError("native session reset failed")
            else:
                pass
        else:
            pass
        return preparation.bypass_generation

    def open_streaming_session(self, session_id: str) -> bool:
        result = self.bridge_scheduler.session_controller.open(
            OpenSessionReqInput(
                session_id=session_id,
                capacity_of_str_len=SESSION_STRING_LENGTH_LIMIT_CHARACTERS,
                streaming=True,
                timeout=None,
            )
        )
        return result.success

    def release_streaming_session(self, session_id: str) -> None:
        self.drain()
        self.bridge_scheduler.session_controller.close(
            CloseSessionReqInput(session_id=session_id)
        )
        if (
            self.bridge_scheduler.session_controller.get(session_id) is not None
            or session_id in self.bridge_scheduler.tree_cache.session_records()
        ):
            raise RuntimeError("streaming session close is still pending")
        else:
            pass

    def create_session_request(
        self, payload: StagePayload, request_data: SGLangARRequestData
    ) -> None:
        if self.admission is None:
            self.drain()
        elif self.admission.records is None:
            self.drain()
            self.admission = self.admission_snapshot()
        else:
            pass
        unit = self.units_by_request_id[payload.request_id]
        adapter_request = request_data.req
        assert (
            adapter_request.rid == payload.request_id
        ), f"session adapter changed request ID: expected {payload.request_id}, got {adapter_request.rid}"
        if (
            adapter_request.input_embeds is not None
            or adapter_request.multimodal_inputs is not None
        ):
            raise ValueError(
                "a session unit carries its embeddings as spans, not as input_embeds or multimodal inputs"
            )
        elif request_data.prefill_input_embeds is not None:
            prompt = request_data.prefill_input_embeds
            if request_data.unit_embedding_spans or int(prompt.shape[0]) != len(
                adapter_request.origin_input_ids
            ):
                raise ValueError(
                    "a projected prompt is the unit's one span and covers all of its ids"
                )
            else:
                request_data.unit_embedding_spans = [
                    EmbeddingSpan(
                        start=0, end=int(prompt.shape[0]), input_embeds=prompt
                    )
                ]
                request_data.prefill_input_embeds = None
        else:
            pass
        native_session = self.bridge_scheduler.session_controller.get(
            unit.session_identity.id
        )
        tokenized_input = TokenizedGenerateReqInput(
            rid=adapter_request.rid,
            input_text=None,
            input_ids=array("q", adapter_request.origin_input_ids),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=adapter_request.sampling_params,
            logprob_start_len=adapter_request.logprob_start_len,
            session_params=SessionParams(id=native_session.session_id),
            stream=adapter_request.stream,
            return_logprob=adapter_request.return_logprob,
            return_sampling_mask=adapter_request.return_sampling_mask,
            lora_id=adapter_request.lora_id,
            custom_logit_processor=adapter_request.custom_logit_processor,
            require_reasoning=adapter_request.require_reasoning,
            return_hidden_states=adapter_request.return_hidden_states,
            return_routed_experts=adapter_request.return_routed_experts,
            routed_experts_start_len=adapter_request.routed_experts_start_len,
            priority=adapter_request.priority,
            routing_key=adapter_request.routing_key,
            extra_key=adapter_request.extra_key,
            cache_salt=adapter_request.cache_salt,
            http_worker_ipc=adapter_request.http_worker_ipc,
            top_logprobs_num=adapter_request.logprob.top_logprobs_num,
            token_ids_logprob=adapter_request.logprob.token_ids_logprob,
        )
        session_request = native_session.create_req(
            tokenized_input,
            adapter_request.tokenizer,
            self.bridge_scheduler.model_config.vocab_size,
            eos_token_ids=adapter_request.eos_token_ids,
        )
        if session_request.to_finish is not None:
            raise ValueError("native session rejected append")
        else:
            pass
        unit.session_request = session_request
        # note (Junnan Li): Native requests prepend retained turns to this unit's ids.
        offset = len(session_request.origin_input_ids) - len(
            adapter_request.origin_input_ids
        )
        unit.native_offset = offset
        unit.prompt_length = len(adapter_request.origin_input_ids)
        unit.embedding_spans = [
            EmbeddingSpan(
                start=span.start + offset,
                end=span.end + offset,
                input_embeds=span.input_embeds,
            )
            for span in request_data.unit_embedding_spans
        ]
        if any(span.start < 0 for span in unit.embedding_spans):
            raise ValueError("embedding span precedes the native sequence")
        else:
            pass
        # note (Junnan Li): Placeholder ids under embedding spans do not identify their rows, so such a unit stays out of the radix tree.
        session_request.skip_radix_cache_insert = bool(unit.embedding_spans)
        session = self.sessions[unit.session_identity.id]
        request_data.session_embedding_spans = [
            *session.embedding_spans,
            *unit.embedding_spans,
        ]
        session_request.logprob_start_len = adapter_request.logprob_start_len
        session_request._omni_prompt_cache_key = getattr(
            adapter_request, "_omni_prompt_cache_key", None
        )  # noqa: leading-underscore
        request_data.req = session_request
        if request_data.stage_payload is None:
            request_data.stage_payload = payload
        else:
            # Note (Junnan Li): An adapter that rewrote the unit's request keeps its payload; the stream builder reads the params from it.
            pass
        self.adapter.bind_request(unit.session_identity, request_data, unit)

    def release_append_unit(self, request_id: str) -> None:
        self.invalidate_admission()
        unit = self.units_by_request_id.pop(request_id, None)
        if unit is not None:
            self.adapter.on_abort(unit.session_identity, unit)
            streaming_session = self.bridge_scheduler.session_controller.get(
                unit.session_identity.id
            )
            if streaming_session is not None and unit.session_request is not None:
                streaming_session.abort_req()
            else:
                pass
            self.sessions[unit.session_identity.id].unit = None
        else:
            pass

    def cancel(self, request_id: str) -> None:
        unit = self.units_by_request_id[request_id]
        session_request = unit.session_request
        previous_cancelling_request_id = self.cancelling_request_id
        self.cancelling_request_id = request_id
        try:
            if not unit.is_enqueued:
                # note (Junnan Li): Before enqueue, release_append_unit must preserve the prior unit's KV.
                self.bridge_scheduler.abort(request_id)
                self.release_append_unit(request_id)
            else:
                assert (
                    session_request is not None
                ), f"enqueued session request {request_id} has no Req"
                is_queued = any(
                    queued_request is session_request
                    for queued_request in self.bridge_scheduler.waiting_queue
                )
                if is_queued:
                    # note (Junnan Li): Failed prefill admission may have restored the prior slot.
                    session_request.detach_kv()
                    session_request.session = None
                    self.bridge_scheduler.abort(request_id)
                    self.drain()
                else:
                    self.bridge_scheduler.abort(request_id)
                    self.drain()
                    session_request.finished_reason = FINISH_ABORT()
                    self.bridge_scheduler.release_request_kv_cache(session_request)
                    # note (Junnan Li): Batch selection must filter rows and tensors together.
                    if self.bridge_scheduler.chunked_req is session_request:
                        self.bridge_scheduler.chunked_req = None
                    else:
                        pass
                    if session_request.omni_data is not None:
                        self.bridge_scheduler.run_abort_callback(request_id)
                        session_request.omni_data = None
                    else:
                        pass
                self.release_append_unit(request_id)
        finally:
            self.cancelling_request_id = previous_cancelling_request_id

    def check_session_capacity(self, request_id: str) -> str | None:
        unit = self.units_by_request_id[request_id]
        session_request = unit.session_request
        assert (
            session_request is not None
        ), f"session capacity check {request_id} requires a session request"
        cache = self.bridge_scheduler.tree_cache
        admission = self.admission
        if admission is None:
            admission = self.admission_snapshot()
        elif admission.records is None:
            admission = self.admission_snapshot()
            self.admission = admission
        else:
            admission.include(unit)
        records = admission.records
        assert records is not None
        record = records.get(unit.session_identity.id)
        retained_kv_tokens = record.kv_allocated_len if record is not None else 0
        free_request_slots = self.bridge_scheduler.req_to_token_pool.free_slots
        own_rows, own_tokens = admission.demands[request_id]
        unallocated_request_count = admission.request_rows - own_rows
        if (
            not retained_kv_tokens
            and len(free_request_slots)
            <= unallocated_request_count
            + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
        ):
            capacity_message = (
                "session request-to-token slots exhausted "
                "(one slot kept for a session that already holds KV)"
            )
        else:
            required_kv_tokens = len(session_request.origin_input_ids) + int(
                session_request.sampling_params.max_new_tokens or 0
            )
            reserved_kv_tokens = admission.kv_tokens - own_tokens
            available_kv_tokens = (
                self.bridge_scheduler.token_to_kv_pool_allocator.available_size()
                + cache.evictable_size()
            )
            if (
                required_kv_tokens - retained_kv_tokens + reserved_kv_tokens
                > available_kv_tokens
            ):
                capacity_message = "session KV capacity exhausted"
            else:
                capacity_message = None
        return capacity_message

    def count_row_reusing_requests(
        self, waiting_queue: list[Req], free_request_rows: int
    ) -> int:
        """Count queued units that reuse their slot's row, up to the first request no free row is left for."""
        records = self.bridge_scheduler.tree_cache.session_records()
        fresh_request_count = 0
        row_reusing_request_count = 0
        for queued_request in waiting_queue:
            session = queued_request.session
            if session is not None and session.streaming:
                record = records.get(session.session_id)
            else:
                record = None
            # note (Junnan Li): An aborted unit does not take its slot's row.
            if (
                record is not None
                and record.holds_kv
                and queued_request.to_finish is None
            ):
                row_reusing_request_count += 1
            elif fresh_request_count < free_request_rows:
                fresh_request_count += 1
            else:
                break
        return row_reusing_request_count

    def stream_messages(
        self,
        request_id: str,
        request_data: SGLangARRequestData,
        output: RequestOutput | None = None,
        *,
        should_flush: bool = False,
    ) -> Iterable[OutgoingMessage]:
        unit = self.units_by_request_id[request_id]
        if should_flush:
            chunks = self.adapter.flush(unit.session_identity, request_data)
        else:
            assert (
                output is not None
            ), f"session stream {request_id} requires request output"
            chunks = self.adapter.stream(unit.session_identity, request_data, output)
        for chunk in chunks:
            yield OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=chunk.to_dict(),
                metadata={"modality": chunk.modality},
            )

    def complete(
        self, request_id: str, request_data: SGLangARRequestData | None = None
    ) -> None:
        self.invalidate_admission()
        unit = self.units_by_request_id.pop(request_id, None)
        if unit is not None:
            session = self.sessions[unit.session_identity.id]
            session.unit = None
            session.embedding_spans.extend(unit.embedding_spans)
            if (
                request_data is not None
                and request_data.input_embeds_are_projected
                and request_data.decode_input_embeds
            ):
                # Note (Junnan Li): A projected talker feeds each generated token back as a computed row, so a KV rebuild needs those rows, not the token embeddings.
                start = unit.native_offset + unit.prompt_length
                session.embedding_spans.append(
                    EmbeddingSpan(
                        start=start,
                        end=start + len(request_data.decode_input_embeds),
                        input_embeds=torch.stack(request_data.decode_input_embeds),
                    )
                )
            else:
                pass
        else:
            pass

    def close_streaming_session(self, session: BridgeSession) -> None:
        if session.unit is not None:
            self.bridge_scheduler.abort(session.unit.request_id)
        else:
            pass
        self.release_streaming_session(session.session_identity.id)

    def close_session(self, session: BridgeSession) -> list[StagePayload]:
        """Release the session; returns its read, answered."""
        self.close_streaming_session(session)
        self.adapter.close(session.session_identity)
        self.sessions.pop(session.session_identity.id)
        if session.output_read is None:
            return []
        else:
            session.output_read.data = {"closed": True}
            return [session.output_read]

    def close_open_sessions(self) -> None:
        adapter_failures: dict[str, Exception] = {}
        for session in list(self.sessions.values()):
            self.close_streaming_session(session)
            session_id = session.session_identity.id
            try:
                self.adapter.close(session.session_identity)
            except Exception as adapter_error:
                adapter_failures[session_id] = adapter_error
            else:
                self.sessions.pop(session_id)
        if not adapter_failures:
            return
        else:
            failure_text = "; ".join(
                f"{session_id}: {adapter_error}"
                for session_id, adapter_error in adapter_failures.items()
            )
            raise RuntimeError(
                f"AR session adapter cleanup failed for {failure_text}"
            ) from next(iter(adapter_failures.values()))
