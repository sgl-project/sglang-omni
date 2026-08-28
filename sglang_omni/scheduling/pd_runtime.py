# SPDX-License-Identifier: Apache-2.0
"""Generic TP=1 SGLang Prefill-Decode runtime helpers."""

from __future__ import annotations

import inspect
import threading
import time
from array import array
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pd_continuation import DecodeContinuation
from sglang_omni.scheduling.pd_kv_adapter import ReservedAllocation
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(frozen=True)
class PDPrefillHandoff:
    continuation: DecodeContinuation
    source_pool_id: str
    target_pool_id: str
    source_page_indices: tuple[int, ...]
    to_stage: str
    lease: Any


@dataclass(frozen=True)
class PDDecodeAdmission:
    continuation: DecodeContinuation
    allocation: ReservedAllocation


class DecodeRequestPoolExhausted(RuntimeError):
    pass


class DecodeOwnershipCapacityExhausted(RuntimeError):
    """Decode cannot accept another KV-owning continuation."""


PDDecodeState = Literal[
    "reserved",
    "committed",
    "ready",
    "deferred",
    "waiting",
    "running",
    "terminal_pending",
]


@dataclass
class PDDecodeOwnedRequest:
    request_id: str
    deadline_unix_s: float | None
    state: PDDecodeState = "reserved"
    allocation: ReservedAllocation | None = None
    req: Any = None


class PDDecodeOwnershipTracker:
    """One accounting domain for every Decode-owned request state.

    ``max_queued_requests`` covers reserved, committed, ready, deferred and
    waiting requests. Moving a request to running returns its queue credit but
    retains the ownership record until terminal cleanup.
    """

    _QUEUED_STATES = frozenset(
        {"reserved", "committed", "ready", "deferred", "waiting"}
    )
    _TRANSITIONS: dict[PDDecodeState, frozenset[PDDecodeState]] = {
        "reserved": frozenset({"committed"}),
        "committed": frozenset({"ready"}),
        "ready": frozenset({"deferred", "waiting"}),
        "deferred": frozenset({"waiting"}),
        "waiting": frozenset({"running"}),
        "running": frozenset({"waiting", "terminal_pending"}),
        "terminal_pending": frozenset(),
    }

    def __init__(self, max_queued_requests: int | None) -> None:
        self._max_queued_requests = max_queued_requests
        self._lock = threading.RLock()
        self._owned: dict[str, PDDecodeOwnedRequest] = {}

    def reserve(self, request_id: str, deadline_unix_s: float | None) -> None:
        with self._lock:
            if request_id in self._owned:
                raise RuntimeError(f"duplicate Decode ownership for {request_id!r}")
            if (
                self._max_queued_requests is not None
                and self._queued_count_locked() >= self._max_queued_requests
            ):
                raise DecodeOwnershipCapacityExhausted(
                    "Decode-owned request capacity is full"
                )
            owned = PDDecodeOwnedRequest(request_id, deadline_unix_s)
            self._owned[request_id] = owned

    def mark_committed(self, request_id: str) -> bool:
        return self.transition_if_owned(request_id, {"reserved"}, "committed")

    def mark_ready(self, request_id: str, allocation: ReservedAllocation) -> bool:
        with self._lock:
            owned = self._owned.get(request_id)
            if owned is None:
                return False
            self._validate_transition(owned.state, "ready")
            owned.allocation = allocation
            owned.state = "ready"
            return True

    def mark_waiting(self, request_id: str, req: Any) -> bool:
        with self._lock:
            owned = self._owned.get(request_id)
            if owned is None:
                return False
            self._validate_transition(owned.state, "waiting")
            owned.req = req
            owned.state = "waiting"
            return True

    def transition_if_owned(
        self,
        request_id: str,
        expected_states: set[PDDecodeState] | frozenset[PDDecodeState],
        new_state: PDDecodeState,
    ) -> bool:
        if new_state not in self._TRANSITIONS:
            raise ValueError(f"unknown Decode ownership state {new_state!r}")
        with self._lock:
            owned = self._owned.get(request_id)
            if owned is None or owned.state not in expected_states:
                return False
            if owned.state != new_state:
                self._validate_transition(owned.state, new_state)
                owned.state = new_state
            if new_state == "terminal_pending":
                owned.deadline_unix_s = None
            return True

    def owns(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._owned

    def pop_for_release(self, request_id: str) -> PDDecodeOwnedRequest | None:
        with self._lock:
            return self._owned.pop(request_id, None)

    def request_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._owned)

    def claim_expired(
        self, now_unix_s: float
    ) -> tuple[tuple[PDDecodeOwnedRequest, bool], ...]:
        with self._lock:
            claimed = []
            for request_id, owned in list(self._owned.items()):
                if owned.deadline_unix_s is None or now_unix_s < owned.deadline_unix_s:
                    continue
                if owned.state == "running":
                    owned.state = "terminal_pending"
                    owned.deadline_unix_s = None
                    claimed.append((owned, True))
                else:
                    del self._owned[request_id]
                    claimed.append((owned, False))
            return tuple(claimed)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(Counter(owned.state for owned in self._owned.values()))

    def queued_count(self) -> int:
        with self._lock:
            return self._queued_count_locked()

    def _queued_count_locked(self) -> int:
        return sum(owned.state in self._QUEUED_STATES for owned in self._owned.values())

    def _validate_transition(
        self, current: PDDecodeState, new_state: PDDecodeState
    ) -> None:
        if new_state not in self._TRANSITIONS[current]:
            raise RuntimeError(
                f"invalid Decode ownership transition {current!r} -> {new_state!r}"
            )


def sampling_params_to_dict(params: Any) -> dict[str, Any]:
    allowed = inspect.signature(type(params)).parameters
    return {name: getattr(params, name) for name in allowed if hasattr(params, name)}


def continuation_from_req(
    req: Any,
    transfer_id: str,
    state_builder: Any = None,
) -> DecodeContinuation:
    if not req.output_ids:
        raise ValueError(f"PD Prefill request {req.rid!r} has no sampled output token")
    data = req._omni_data
    stage_payload = data.stage_payload
    stage_payload_dict = stage_payload.to_dict()
    multimodal_resume = None
    origin_input_ids = list(req.origin_input_ids)
    if state_builder is not None:
        stage_payload_dict, multimodal_resume, origin_input_ids = state_builder(req)
    from sglang.srt.environ import envs

    timeout_s = float(envs.SGLANG_REQ_WAITING_TIMEOUT.get())
    deadline_unix_s = None
    if timeout_s > 0:
        entry = float(req.time_stats.wait_queue_entry_time or 0.0)
        elapsed = max(0.0, time.perf_counter() - entry) if entry > 0 else 0.0
        deadline_unix_s = time.time() + max(0.0, timeout_s - elapsed)
    return DecodeContinuation(
        request_id=req.rid,
        transfer_id=transfer_id,
        origin_input_ids=origin_input_ids,
        origin_input_ids_unpadded=(
            list(req.origin_input_ids_unpadded)
            if req.origin_input_ids_unpadded is not None
            else None
        ),
        output_ids=list(req.output_ids),
        vocab_size=int(req.vocab_size),
        sampling_params=sampling_params_to_dict(req.sampling_params),
        cached_tokens=int(req.cached_tokens),
        deadline_unix_s=deadline_unix_s,
        eos_token_ids=(list(req.eos_token_ids) if req.eos_token_ids else None),
        cached_tokens_device=int(req.cached_tokens_device),
        cached_tokens_host=int(req.cached_tokens_host),
        cached_tokens_storage=int(req.cached_tokens_storage),
        mm_image_tokens=int(req.mm_image_tokens),
        mm_audio_tokens=int(req.mm_audio_tokens),
        mm_video_tokens=int(req.mm_video_tokens),
        return_logprob=bool(data.return_logprob),
        output_token_logprobs=list(data.output_token_logprobs),
        top_logprobs_num=int(req.logprob.top_logprobs_num),
        token_ids_logprob=(
            list(req.logprob.token_ids_logprob)
            if req.logprob.token_ids_logprob is not None
            else None
        ),
        logprob_start_len=int(req.logprob_start_len),
        return_hidden_states=bool(req.return_hidden_states),
        return_sampling_mask=bool(req.return_sampling_mask),
        return_routed_experts=bool(req.return_routed_experts),
        return_indexer_topk=bool(req.return_indexer_topk),
        custom_logit_processor=req.custom_logit_processor,
        input_embeds_are_projected=bool(data.input_embeds_are_projected),
        speculative=False,
        multimodal_resume=multimodal_resume,
        stage_payload=stage_payload_dict,
    )


def req_from_continuation(
    continuation: DecodeContinuation,
    allocation: ReservedAllocation,
    *,
    req_to_token_pool: Any,
    state_restorer: Any = None,
) -> Any:
    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
    from sglang.srt.sampling.sampling_params import SamplingParams

    sampling_params = SamplingParams(**continuation.sampling_params)
    req = Req(
        rid=continuation.request_id,
        origin_input_text="",
        origin_input_ids=array("q", continuation.origin_input_ids),
        origin_input_ids_unpadded=(
            array("q", continuation.origin_input_ids_unpadded)
            if continuation.origin_input_ids_unpadded is not None
            else None
        ),
        sampling_params=sampling_params,
        # Note (Yue Yin): Omni owns resumed logprobs because upstream Req has no
        # corresponding Prefill logits object on the Decode process.
        return_logprob=False,
        top_logprobs_num=continuation.top_logprobs_num,
        token_ids_logprob=continuation.token_ids_logprob,
        return_sampling_mask=continuation.return_sampling_mask,
        custom_logit_processor=continuation.custom_logit_processor,
        return_hidden_states=continuation.return_hidden_states,
        return_routed_experts=continuation.return_routed_experts,
        return_indexer_topk=continuation.return_indexer_topk,
        eos_token_ids=(
            set(continuation.eos_token_ids)
            if continuation.eos_token_ids is not None
            else None
        ),
        vocab_size=continuation.vocab_size,
    )
    req.output_ids.extend(continuation.output_ids)
    req.cached_tokens = continuation.cached_tokens
    req.already_computed = continuation.cached_tokens
    req.cached_tokens_device = continuation.cached_tokens_device
    req.cached_tokens_host = continuation.cached_tokens_host
    req.cached_tokens_storage = continuation.cached_tokens_storage
    req.mm_image_tokens = continuation.mm_image_tokens
    req.mm_audio_tokens = continuation.mm_audio_tokens
    req.mm_video_tokens = continuation.mm_video_tokens
    req.logprob_start_len = continuation.logprob_start_len

    payload = StagePayload.from_dict(continuation.stage_payload or {})
    data = SGLangARRequestData(
        input_ids=torch.tensor(continuation.origin_input_ids, dtype=torch.long),
        output_ids=req.output_ids,
        req=req,
        stage_payload=payload,
        max_new_tokens=int(sampling_params.max_new_tokens),
        temperature=float(sampling_params.temperature),
        return_logprob=continuation.return_logprob,
        output_token_logprobs=list(continuation.output_token_logprobs),
    )
    req._omni_data = data
    if state_restorer is not None:
        state_restorer(req, data, continuation.multimodal_resume)
    indices = req_to_token_pool.alloc([req])
    if indices is None:
        raise DecodeRequestPoolExhausted("decode request pool is exhausted")
    try:
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, allocation.seq_len)), allocation.slots
        )
    except Exception:
        req_to_token_pool.free(req)
        raise
    req.prefix_indices = allocation.slots
    req.kv_committed_len = allocation.seq_len
    req.kv = ReqKvInfo(kv_allocated_len=allocation.seq_len, swa_evicted_seqlen=0)
    req.set_extend_range(allocation.seq_len, allocation.seq_len)
    req._omni_terminal_claimed = False
    req._coalesce_enqueue_t = 0.0
    return req
