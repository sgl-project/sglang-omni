# SPDX-License-Identifier: Apache-2.0
"""Bridge pipeline units to native streaming sessions."""
from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
from typing import Iterable

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT

from sglang_omni.admission import QueueFullError
from sglang_omni.profiler.event_recorder import get_active_stage
from sglang_omni.proto import StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    ResourceUsage,
    SessionRef,
    TimedChunk,
    wire_size,
)
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput


class ARSessionAdapter:
    """Convert unit inputs and outputs without mutating native session state.

    Build from the relayed payload; only the entry stage consumes chunk.payload.
    """

    def open(self, ref: SessionRef, request) -> None:
        """Initialize auxiliary model history after native session creation."""

    def close(self, ref: SessionRef) -> None:
        """Release auxiliary history after native work and KV are released."""

    def finish_input(
        self, ref: SessionRef, payload: StagePayload
    ) -> StagePayload | None:
        """Return a relayed EOS payload, or None to use normal generation."""
        return None

    def build(
        self, ref: SessionRef, chunk: TimedChunk, payload: StagePayload
    ) -> SGLangARRequestData:
        raise NotImplementedError

    def result(self, ref: SessionRef, data: SGLangARRequestData) -> StagePayload:
        raise NotImplementedError

    def stream(
        self, ref: SessionRef, data: SGLangARRequestData, output: RequestOutput
    ) -> Iterable[TimedChunk]:
        return ()

    def flush(self, ref: SessionRef, data: SGLangARRequestData) -> Iterable[TimedChunk]:
        return ()


@dataclass
class _Owner:
    ref: SessionRef
    active: str | None = None
    req: object = None
    native_owned: bool = False
    enqueued: bool = False
    payload: StagePayload | None = None
    emitted_count: int = 0
    emitted_bytes: int = 0


class ARSessionBridge:
    """All methods run on the scheduler thread, including lifecycle commands."""

    def __init__(self, scheduler, adapter: ARSessionAdapter):
        self.scheduler = scheduler
        self.adapter = adapter
        self.owners: dict[str, _Owner] = {}
        self.requests: dict[str, _Owner] = {}
        self.cancelling: str | None = None

    def _drain(self) -> None:
        pending = self.scheduler._async_pending
        if pending is not None:
            pending[2].event.synchronize()
        self.scheduler._resolve_pending_async()

    @staticmethod
    def native_id(ref: SessionRef) -> str:
        return ref.session_id

    @staticmethod
    def metadata(payload):
        return payload.request.metadata.get(SESSION_METADATA_KEY)

    @classmethod
    def is_cleanup(cls, payload) -> bool:
        command = cls.metadata(payload)
        return command is not None and command.get("op") in {"abort", "close"}

    def command(self, payload: StagePayload) -> StagePayload:
        command = self.metadata(payload)
        ref = SessionRef(**command["ref"])
        sid = self.native_id(ref)
        op = command["op"]
        controller = self.scheduler.session_controller
        owner = self.owners.get(sid)
        if owner is not None and ref.incarnation != owner.ref.incarnation:
            raise ValueError("stale session incarnation")
        if op == "open":
            if owner is not None:
                raise ValueError("session already opened")
            if len(self.owners) >= self.scheduler.max_running_requests:
                raise QueueFullError()
            result = controller.open(
                OpenSessionReqInput(
                    session_id=sid, capacity_of_str_len=0, streaming=True, timeout=None
                )
            )
            if not result.success:
                raise ValueError("native session open failed")
            self.owners[sid] = _Owner(ref)
            self.adapter.open(ref, payload.request)
            payload.data = {"opened": True}
        elif op == "close":
            if owner is not None:
                if owner.active is not None:
                    self.scheduler.abort(owner.active)
                self._drain()
                controller.close(CloseSessionReqInput(session_id=sid))
                if (
                    controller.get(sid) is not None
                    or sid in self.scheduler.tree_cache.slots
                ):
                    raise RuntimeError("native session close is still pending")
                self.adapter.close(owner.ref)
                del self.owners[sid]
            payload.data = {"closed": True}
        elif op == "abort":
            if owner is None:
                raise ValueError("unknown session incarnation")
            if ref.epoch != owner.ref.epoch + 1:
                raise ValueError("invalid abort epoch")
            if owner.active is not None:
                raise RuntimeError("session cancel must follow unit completion")
            self._drain()
            owner.ref = ref
            payload.data = {"aborted": True}
        else:
            raise ValueError("unknown session operation")
        return payload

    def accept(self, payload: StagePayload) -> None:
        ref = SessionRef(**self.metadata(payload)["ref"])
        owner = self.owners.get(self.native_id(ref))
        if owner is None or ref != owner.ref:
            raise ValueError("unknown or stale session incarnation/epoch")
        if owner.active is not None:
            if owner.active == payload.request_id:
                return
            raise ValueError("session already has an active request")
        owner.active = payload.request_id
        owner.payload = payload
        owner.emitted_count = owner.emitted_bytes = 0
        self.requests[payload.request_id] = owner

    def build(self, payload):
        owner = self.requests[payload.request_id]
        data = self.adapter.build(
            owner.ref, TimedChunk(**self.metadata(payload)["chunk"]), payload
        )
        if not isinstance(data, SGLangARRequestData):
            raise TypeError(
                "session build must return SGLangARRequestData synchronously"
            )
        return data

    def finish_input(self, payload):
        owner = self.requests[payload.request_id]
        try:
            result = self.adapter.finish_input(owner.ref, payload)
        except Exception:
            self.complete(payload.request_id)
            raise
        if result is not None:
            self.complete(payload.request_id)
        return result

    def materialize(self, payload, data: SGLangARRequestData) -> None:
        self._drain()
        owner = self.requests[payload.request_id]
        old = data.req
        if old.rid != payload.request_id:
            raise ValueError("session adapter changed request ID")
        if (
            data.prefill_input_embeds is not None
            or data.decode_input_embeds
            or data.input_embeds_are_projected
            or old.input_embeds is not None
            or old.multimodal_inputs is not None
        ):
            raise ValueError(
                "history-aware session embedding and multimodal inputs are not supported"
            )
        native = self.scheduler.session_controller.get(self.native_id(owner.ref))
        fields = {
            name: getattr(old, name)
            for name in (
                "stream",
                "return_logprob",
                "return_sampling_mask",
                "lora_id",
                "custom_logit_processor",
                "require_reasoning",
                "return_hidden_states",
                "return_routed_experts",
                "routed_experts_start_len",
                "priority",
                "routing_key",
                "extra_key",
                "cache_salt",
                "http_worker_ipc",
            )
        }
        fields["top_logprobs_num"] = old.logprob.top_logprobs_num
        fields["token_ids_logprob"] = old.logprob.token_ids_logprob
        tokenized = TokenizedGenerateReqInput(
            rid=old.rid,
            input_text=None,
            input_ids=array("q", old.origin_input_ids),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=old.sampling_params,
            logprob_start_len=old.logprob_start_len,
            session_params=SessionParams(id=native.session_id),
            **fields,
        )
        req = native.create_req(
            tokenized,
            old.tokenizer,
            self.scheduler.model_config.vocab_size,
            eos_token_ids=old.eos_token_ids,
        )
        if req.to_finish is not None:
            raise ValueError("native session rejected append")
        owner.req = req
        owner.native_owned = True
        req.logprob_start_len = old.logprob_start_len
        req._omni_prompt_cache_key = getattr(old, "_omni_prompt_cache_key", None)
        data.req = req
        data.stage_payload = payload

    def rollback(self, rid: str) -> None:
        owner = self.requests.pop(rid, None)
        if owner is None:
            return
        native = self.scheduler.session_controller.get(self.native_id(owner.ref))
        if native is not None and owner.native_owned:
            native.abort_req()
        owner.native_owned = False
        owner.enqueued = False
        owner.active = owner.req = owner.payload = None

    def cancel(self, rid: str) -> None:
        owner = self.requests[rid]
        req = owner.req
        previous_cancelling = self.cancelling
        self.cancelling = rid
        try:
            if not owner.enqueued:
                # Note (Junnan Li): Before enqueue, KV still belongs to the prior
                # unit; rejection must roll back history without releasing it.
                self.scheduler.abort(rid)
                self.rollback(rid)
                return
            queued = req is not None and any(
                r is req for r in self.scheduler.waiting_queue
            )
            if queued:
                # Note (Junnan Li): Failed prefill admission may have restored the prior slot.
                req.detach_kv()
                req.session = None
            self.scheduler.abort(rid)
            self._drain()
            if req is not None and not queued:
                req.finished_reason = FINISH_ABORT()
                self.scheduler._release_request_kv_cache(req)
                # Note (Junnan Li): Removing only reqs would misalign batch tensors;
                # let the next batch selection filter finished rows together.
                if self.scheduler.chunked_req is req:
                    self.scheduler.chunked_req = None
                if req._omni_data is not None:
                    self.scheduler._run_abort_callback(rid)
                    req._omni_data = None

            self.rollback(rid)
        finally:
            self.cancelling = previous_cancelling

    def capacity_error(self, rid: str) -> str | None:
        owner = self.requests[rid]
        req = owner.req
        cache = self.scheduler.tree_cache
        slot = cache.slots.get(self.native_id(owner.ref))
        held = slot.kv.kv_allocated_len if slot is not None else 0
        free_slots = self.scheduler.req_to_token_pool.free_slots
        unallocated = sum(
            other is not owner
            and other.req is not None
            and not other.req.kv.holds_kv
            and self.native_id(other.ref) not in cache.slots
            for other in self.requests.values()
        )
        # Note (Junnan Li): Prefill needs a free slot even when reusing one;
        # reserving it prevents retained sessions from deadlocking on append.
        if not held and len(free_slots) <= unallocated + 1:
            return (
                "session request slot capacity exhausted (one admission slot reserved)"
            )
        required = len(req.origin_input_ids) + int(
            req.sampling_params.max_new_tokens or 0
        )
        reserved = 0
        for other in self.requests.values():
            if other is owner or other.req is None:
                continue
            r = other.req
            allocated = r.kv.kv_allocated_len if r.kv is not None else 0
            reserved += max(
                0,
                len(r.origin_input_ids)
                + int(r.sampling_params.max_new_tokens or 0)
                - allocated,
            )
        available = (
            self.scheduler.token_to_kv_pool_allocator.available_size()
            + cache.evictable_size()
        )
        if required - held + reserved > available:
            return "session KV capacity exhausted"
        return None

    def result(self, rid, data):
        return self.adapter.result(self.requests[rid].ref, data)

    def messages(self, rid, data, output=None, *, flush=False):
        owner = self.requests[rid]
        stages = self.metadata(owner.payload)["stages"]
        if get_active_stage() != stages[-1]:
            return
        hook = self.adapter.flush if flush else self.adapter.stream
        chunks = hook(owner.ref, data) if flush else hook(owner.ref, data, output)
        for chunk in chunks:
            if not isinstance(chunk, TimedChunk):
                raise TypeError("session stream adapter must return TimedChunk")
            encoded = asdict(chunk)
            owner.emitted_count += 1
            owner.emitted_bytes += wire_size(encoded)
            limits = self.metadata(owner.payload)["output_limits"]
            if (
                owner.emitted_count > limits["chunks"]
                or owner.emitted_bytes > limits["bytes"]
            ):
                raise QueueFullError()
            yield OutgoingMessage(
                request_id=rid,
                type="stream",
                data=encoded,
                metadata={"modality": chunk.modality},
            )

    def complete(self, rid):
        owner = self.requests.pop(rid, None)
        if owner is not None:
            owner.native_owned = False
            owner.enqueued = False
            owner.active = owner.req = owner.payload = None

    def usage(self, ref: SessionRef) -> ResourceUsage:
        sid = self.native_id(ref)
        slot = self.scheduler.tree_cache.slots.get(sid)
        owner = self.owners.get(sid)
        active = owner.req if owner is not None else None
        kv = (
            active.kv
            if active is not None and active.kv is not None
            else (slot.kv if slot is not None else None)
        )
        if kv is None:
            return ResourceUsage()
        page_size = self.scheduler.tree_cache.page_size
        tokens = (kv.kv_allocated_len + page_size - 1) // page_size * page_size
        pool = self.scheduler.token_to_kv_pool_allocator.get_kvcache()
        token_bytes = sum(
            pool.get_key_buffer(i)[0].nbytes + pool.get_value_buffer(i)[0].nbytes
            for i in range(pool.start_layer, pool.end_layer)
        )
        return ResourceUsage(
            kv_tokens=tokens, slots={"request": 1}, bytes=tokens * token_bytes
        )

    def shutdown(self):
        for owner in list(self.owners.values()):
            if owner.active is not None:
                self.scheduler.abort(owner.active)
            sid = self.native_id(owner.ref)
            self._drain()
            self.scheduler.session_controller.close(
                CloseSessionReqInput(session_id=sid)
            )
            if (
                self.scheduler.session_controller.get(sid) is not None
                or sid in self.scheduler.tree_cache.slots
            ):
                raise RuntimeError("native session shutdown still pending")
            self.adapter.close(owner.ref)
            del self.owners[sid]
