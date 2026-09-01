# SPDX-License-Identifier: Apache-2.0
"""Utilities shared by the explicit Prefill and Decode schedulers."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import queue
import threading
from array import array
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import msgspec
import torch

from sglang_omni.comm import KVBufferRegion, KVPageDestination, KVPool
from sglang_omni.proto import KVTransferPrepareMessage, StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

logger = logging.getLogger(__name__)

CONTINUATION_VERSION = 1
_TRANSFER_TOMBSTONE_LIMIT = 10000


@dataclasses.dataclass(frozen=True)
class DecodeContinuation:
    """Serializable request state needed after the Prefill KV has arrived."""

    request_id: str
    transfer_id: str
    origin_input_ids: list[int]
    output_ids: list[int]
    vocab_size: int
    sampling_params: dict[str, Any]
    stage_payload: dict[str, Any]
    origin_input_ids_unpadded: list[int] | None = None
    eos_token_ids: list[int] | None = None
    cached_tokens: int = 0
    cached_tokens_device: int = 0
    cached_tokens_host: int = 0
    cached_tokens_storage: int = 0
    mm_image_tokens: int = 0
    mm_audio_tokens: int = 0
    mm_video_tokens: int = 0
    return_logprob: bool = False
    output_token_logprobs: list[Any] = dataclasses.field(default_factory=list)
    top_logprobs_num: int = 0
    token_ids_logprob: list[int] | None = None
    logprob_start_len: int = -1
    return_hidden_states: bool = False
    return_sampling_mask: bool = False
    return_routed_experts: bool = False
    return_indexer_topk: bool = False
    multimodal_resume: dict[str, Any] | None = None
    version: int = CONTINUATION_VERSION

    def __post_init__(self) -> None:
        if self.version != CONTINUATION_VERSION:
            raise ValueError(f"unsupported decode continuation version {self.version}")
        if not self.request_id or not self.transfer_id:
            raise ValueError("decode continuation ids must be non-empty")
        if not self.output_ids:
            raise ValueError("decode continuation requires the Prefill token")
        if self.vocab_size <= 0:
            raise ValueError("decode continuation vocab_size must be positive")
        if not isinstance(self.sampling_params, dict):
            raise TypeError("decode continuation sampling_params must be a mapping")
        if not isinstance(self.stage_payload, dict):
            raise TypeError("decode continuation stage_payload must be a mapping")
        if self.multimodal_resume is not None and not isinstance(
            self.multimodal_resume, dict
        ):
            raise TypeError("decode continuation multimodal_resume must be a mapping")
        try:
            msgspec.msgpack.encode(self.sampling_params)
        except Exception as exc:
            raise ValueError(
                "decode continuation sampling state is not serializable"
            ) from exc

    def encode(self) -> bytes:
        return msgspec.msgpack.encode(dataclasses.asdict(self))

    @classmethod
    def decode(cls, value: bytes) -> DecodeContinuation:
        try:
            decoded = msgspec.msgpack.decode(value)
        except Exception as exc:
            raise ValueError("invalid decode continuation encoding") from exc
        if not isinstance(decoded, dict):
            raise TypeError("decode continuation must contain a mapping")
        expected = {field.name for field in dataclasses.fields(cls)}
        unknown = set(decoded) - expected
        if unknown:
            raise ValueError(f"unknown decode continuation fields: {unknown}")
        try:
            return cls(**decoded)
        except TypeError as exc:
            raise ValueError(f"invalid decode continuation: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class ReservedKV:
    slots: torch.Tensor
    page_indices: tuple[int, ...]
    seq_len: int


@dataclasses.dataclass(frozen=True)
class DecodeAdmission:
    continuation: DecodeContinuation
    allocation: ReservedKV


StateBuilder = Callable[[Any], tuple[dict[str, Any], dict[str, Any] | None, list[int]]]
StateRestorer = Callable[[Any, SGLangARRequestData, dict[str, Any] | None], None]


def default_state_builder(
    req: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[int]]:
    return req._omni_data.stage_payload.to_dict(), None, list(req.origin_input_ids)


def default_state_restorer(
    req: Any,
    data: SGLangARRequestData,
    resume: dict[str, Any] | None,
) -> None:
    del req, data
    if resume is not None:
        raise ValueError("generic PD resume does not accept model-specific state")


def continuation_from_req(
    req: Any,
    transfer_id: str,
    state_builder: StateBuilder,
) -> DecodeContinuation:
    """Snapshot a completed Prefill request without serializing runtime tensors."""

    if not req.output_ids:
        raise ValueError(f"Prefill request {req.rid!r} produced no token")
    if req.custom_logit_processor:
        raise NotImplementedError("PD does not support custom logit processors")
    data = req._omni_data
    if data.input_embeds_are_projected or getattr(
        req, "_input_embeds_are_projected", False
    ):
        raise NotImplementedError("PD does not support projected input embeddings")
    sampling = _sampling_params_to_dict(req.sampling_params)
    if any(
        sampling.get(key)
        for key in (
            "grammar",
            "json_schema",
            "regex",
            "ebnf",
            "structural_tag",
        )
    ):
        raise NotImplementedError("PD does not support structured-output sampling")
    try:
        msgspec.msgpack.encode(sampling)
    except Exception as exc:
        raise ValueError("PD sampling state is not serializable") from exc

    payload, multimodal_resume, origin_input_ids = state_builder(req)
    if not isinstance(payload, dict):
        raise TypeError("PD state builder must return a payload mapping")
    if multimodal_resume is not None and not isinstance(multimodal_resume, dict):
        raise TypeError("PD state builder must return a resume mapping")
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
        sampling_params=sampling,
        stage_payload=payload,
        eos_token_ids=list(req.eos_token_ids) if req.eos_token_ids else None,
        cached_tokens=int(req.cached_tokens),
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
        multimodal_resume=multimodal_resume,
    )


class DecodeRequestPoolExhausted(RuntimeError):
    """The KV is committed, but no request-table row is currently available."""


def req_from_continuation(
    continuation: DecodeContinuation,
    allocation: ReservedKV,
    *,
    req_to_token_pool: Any,
    state_restorer: StateRestorer,
) -> Any:
    """Install a transferred request as SGLang's existing PREBUILT input."""

    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
    from sglang.srt.sampling.sampling_params import SamplingParams

    sampling_values = dict(continuation.sampling_params)
    if isinstance(sampling_values.get("stop_token_ids"), list):
        sampling_values["stop_token_ids"] = set(sampling_values["stop_token_ids"])
    sampling_params = SamplingParams(**sampling_values)
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
        # Prefill logprobs are already retained by Omni request data.
        return_logprob=False,
        top_logprobs_num=continuation.top_logprobs_num,
        token_ids_logprob=continuation.token_ids_logprob,
        return_sampling_mask=continuation.return_sampling_mask,
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

    payload = StagePayload.from_dict(continuation.stage_payload)
    data = SGLangARRequestData(
        input_ids=torch.tensor(continuation.origin_input_ids, dtype=torch.long),
        output_ids=req.output_ids,
        req=req,
        stage_payload=payload,
        max_new_tokens=int(sampling_params.max_new_tokens),
        temperature=float(sampling_params.temperature),
        top_p=float(sampling_params.top_p),
        top_k=int(sampling_params.top_k),
        repetition_penalty=float(sampling_params.repetition_penalty),
        return_logprob=continuation.return_logprob,
        output_token_logprobs=list(continuation.output_token_logprobs),
    )
    req._omni_data = data
    state_restorer(req, data, continuation.multimodal_resume)

    if req_to_token_pool.alloc([req]) is None:
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


def _sampling_params_to_dict(params: Any) -> dict[str, Any]:
    allowed = inspect.signature(type(params)).parameters
    return {name: getattr(params, name) for name in allowed if hasattr(params, name)}


@contextmanager
def defer_first_token_finish(reqs: list[Any]):
    """Let normal Prefill accounting run while Decode owns stop decisions."""

    saved = []
    for req in reqs:
        params = req.sampling_params
        values = (
            params.max_new_tokens,
            params.ignore_eos,
            params.stop_strs,
            params.stop_regex_strs,
        )
        saved.append((params, values))
        params.max_new_tokens = max(int(params.max_new_tokens), len(req.output_ids) + 2)
        params.ignore_eos = True
        params.stop_strs = []
        params.stop_regex_strs = []
    try:
        yield
    finally:
        for params, values in saved:
            (
                params.max_new_tokens,
                params.ignore_eos,
                params.stop_strs,
                params.stop_regex_strs,
            ) = values


def build_kv_pool(token_to_kv_pool: Any, *, pool_id: str) -> KVPool:
    getter = getattr(token_to_kv_pool, "_pd_registerable_tensors", None)
    if callable(getter):
        tensors = tuple(getter())
    else:
        tensors = tuple(
            tensor
            for layer_id in range(int(token_to_kv_pool.layer_num))
            for tensor in (
                token_to_kv_pool.get_key_buffer(layer_id),
                token_to_kv_pool.get_value_buffer(layer_id),
            )
        )
    _, _, item_lens = token_to_kv_pool.get_contiguous_buf_infos()
    if not tensors or len(tensors) != len(item_lens):
        raise ValueError("SGLang KV pool exposed incompatible buffer metadata")
    return KVPool(
        pool_id=pool_id,
        layout_id=(
            f"{type(token_to_kv_pool).__module__}."
            f"{type(token_to_kv_pool).__qualname__}:page_size=1"
        ),
        page_size=1,
        buffers=tuple(
            KVBufferRegion(
                name=f"kv_buffer.{index}",
                tensor=tensor,
                bytes_per_page=int(item_len),
            )
            for index, (tensor, item_len) in enumerate(zip(tensors, item_lens))
        ),
    )


def request_page_indices(req_to_token_pool: Any, req: Any) -> tuple[int, ...]:
    if req.req_pool_idx is None:
        raise RuntimeError(f"request {req.rid!r} has no KV mapping")
    seq_len = len(req.origin_input_ids)
    if seq_len <= 0:
        raise ValueError("PD cannot transfer an empty prompt")
    return tuple(
        int(slot)
        for slot in req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len].tolist()
    )


@dataclasses.dataclass
class _Reservation:
    continuation: DecodeContinuation
    allocation: ReservedKV


class DecodeKVReceiver:
    """Reserve destination pages, then queue one admission when copy commits."""

    def __init__(
        self,
        *,
        pool_id: str,
        allocator: Any,
        admissions: queue.SimpleQueue[DecodeAdmission],
        allowed_resume_schemas: frozenset[str],
    ) -> None:
        self.pool_id = pool_id
        self._allocator = allocator
        self._admissions = admissions
        self._allowed_resume_schemas = allowed_resume_schemas
        self._lock = threading.Lock()
        self._reservations: dict[str, _Reservation] = {}
        self._transfer_tombstones: dict[str, None] = {}
        self._closed = False

    def _remember_finished_transfer(self, transfer_id: str) -> None:
        """Keep a bounded dedup window for transfers we no longer own."""

        if transfer_id in self._transfer_tombstones:
            return
        if len(self._transfer_tombstones) >= _TRANSFER_TOMBSTONE_LIMIT:
            del self._transfer_tombstones[next(iter(self._transfer_tombstones))]
        self._transfer_tombstones[transfer_id] = None

    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination:
        if request.target_pool_id != self.pool_id:
            raise ValueError(f"KV receiver for {self.pool_id!r} got another pool")
        raw = request.metadata.get("decode_continuation")
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("KV transfer is missing decode continuation bytes")
        continuation = DecodeContinuation.decode(bytes(raw))
        if (
            continuation.request_id != request.request_id
            or continuation.transfer_id != request.transfer_id
        ):
            raise ValueError("KV transfer and decode continuation ids differ")
        resume = continuation.multimodal_resume
        if resume is not None:
            schema = resume.get("schema")
            if schema not in self._allowed_resume_schemas:
                raise ValueError(f"unsupported multimodal resume schema {schema!r}")

        count = len(request.source_page_indices)
        if count <= 0:
            raise ValueError("KV transfer contains no pages")
        with self._lock:
            if self._closed:
                raise RuntimeError("decode KV receiver is closed")
            if (
                request.transfer_id in self._reservations
                or request.transfer_id in self._transfer_tombstones
            ):
                raise RuntimeError(f"duplicate KV transfer {request.transfer_id!r}")
            if int(self._allocator.available_size()) < count:
                raise RuntimeError(
                    f"decode KV pool exhausted: need {count}, "
                    f"have {self._allocator.available_size()}"
                )
            slots = self._allocator.alloc(count)
            if slots is None:
                raise RuntimeError(f"decode KV allocator failed to allocate {count}")
            # Note (Audrey Zheng): three different quantities that happen to
            # be equal at page_size == 1 -- how many tokens the request has,
            # how many slots back them, and which pages hold them. Deriving
            # all three from one number is what makes page_size > 1 look like
            # a one-line change when it is not. The token count is the only
            # one that is a property of the request rather than of the
            # allocation, so take it from the continuation and check it.
            seq_len = len(continuation.origin_input_ids)
            if seq_len != count:
                self._allocator.free(slots)
                raise RuntimeError(
                    f"KV transfer {request.transfer_id!r} carries {count} "
                    f"pages for {seq_len} tokens; PD requires page_size == 1"
                )
            allocation = ReservedKV(
                slots=slots,
                page_indices=tuple(int(slot) for slot in slots.tolist()),
                seq_len=seq_len,
            )
            self._reservations[request.transfer_id] = _Reservation(
                continuation, allocation
            )
        return KVPageDestination(self.pool_id, allocation.page_indices)

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None:
        with self._lock:
            reservation = self._reservations.pop(request.transfer_id, None)
            if reservation is None:
                raise RuntimeError(
                    f"commit for unknown KV transfer {request.transfer_id!r}"
                )
            self._remember_finished_transfer(request.transfer_id)
            if (
                self._closed
                or request.request_id != reservation.continuation.request_id
                or request.target_pool_id != self.pool_id
                or destination.pool_id != self.pool_id
                or len(request.source_page_indices) != reservation.allocation.seq_len
                or reservation.allocation.page_indices != destination.page_indices
            ):
                self._allocator.free(reservation.allocation.slots)
                raise RuntimeError("KV commit does not match its reservation")
            try:
                self._admissions.put(
                    DecodeAdmission(reservation.continuation, reservation.allocation)
                )
            except Exception:
                self._allocator.free(reservation.allocation.slots)
                raise

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None:
        del destination
        with self._lock:
            reservation = self._reservations.pop(request.transfer_id, None)
            if reservation is not None:
                self._remember_finished_transfer(request.transfer_id)
        if reservation is not None:
            self._allocator.free(reservation.allocation.slots)
        logger.warning("KV receive aborted for %s: %s", request.request_id, error)

    def close(self) -> None:
        """Release every receiver-owned reservation exactly once."""

        with self._lock:
            self._closed = True
            reservations = list(self._reservations.values())
            self._reservations.clear()
        for reservation in reservations:
            self._allocator.free(reservation.allocation.slots)


class SGLangKVLease:
    """Keep source pages owned until the receiver ACKs the copy.

    ``release`` is called by the comm event loop, from ``_watch_pending``'s
    ``finally``. What it releases belongs to the scheduler thread: with
    RadixCache disabled ``release_kv_cache`` frees through the allocator, and
    with it enabled it inserts into the prefix tree, drops a reference, and
    updates eviction bookkeeping -- all while the scheduler is matching and
    evicting against that same tree.

    So the lease does not release. It records that the release is due and the
    scheduler performs it on its own thread, which is the only thread that
    owns the tree. Handing the work over rather than locking is possible here
    because a release needs no answer; the reservation on the receive side
    does, which is why that one takes a lock instead.
    """

    def __init__(self, req: Any, tree_cache: Any, due: queue.SimpleQueue) -> None:
        self._req = req
        self._tree_cache = tree_cache
        self._due = due
        self._lock = threading.Lock()

    def release(self) -> None:
        """Hand the request to the scheduler thread. Idempotent."""
        with self._lock:
            req = self._req
            self._req = None
        if req is None:
            return
        self._due.put(req)


def drain_due_releases(due: queue.SimpleQueue, tree_cache: Any) -> int:
    """Release every request whose copy has been acknowledged.

    Called on the scheduler thread. Returns how many were released, so a
    caller can assert the queue drains rather than grows.
    """
    from sglang.srt.mem_cache.common import release_kv_cache

    released = 0
    while True:
        try:
            req = due.get_nowait()
        except queue.Empty:
            return released
        try:
            release_kv_cache(req, tree_cache)
        except Exception:
            logger.exception(
                "releasing KV for request %s failed", getattr(req, "rid", "?")
            )
        released += 1
