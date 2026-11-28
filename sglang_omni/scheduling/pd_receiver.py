# SPDX-License-Identifier: Apache-2.0
"""Decode-side continuation schema, KV preallocation queue, and row-credit ownership."""

from __future__ import annotations

import _thread
import asyncio
import dataclasses
import logging
import threading
from collections import deque
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Literal, Protocol

import msgspec
import torch
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

from sglang_omni.admission import AdminDroppedRequestError, QueueFullError
from sglang_omni.comm.kv_transfer import KVPageDestination, KVTransferCancelled
from sglang_omni.proto.kv_transfer import KVTransferPrepareMessage

logger = logging.getLogger(__name__)

CONTINUATION_VERSION = 1
TRANSFER_TOMBSTONE_LIMIT = 10000
DEFAULT_DECODE_RUNWAY_TOKENS = 512


class WakeupNotifier(Protocol):
    def notify_pd_wakeup(self) -> None: ...


class DecodeCapacityCheck(Protocol):
    """Read-only active-request check for the zero-active runway bypass."""

    def has_active_decode_requests(self) -> bool: ...


def fail_future(
    future: asyncio.Future[KVPageDestination], error: BaseException
) -> None:
    if not future.done():
        future.set_exception(error)
    else:
        pass


def page_indices_from_heads(
    raw_heads: list[int],
    *,
    expected_pages: int,
    page_size: int,
    context_label: str,
) -> tuple[int, ...]:
    """Validate page-aligned slot heads and convert them to physical page indices."""
    if len(raw_heads) != expected_pages:
        raise RuntimeError(
            f"{context_label} expected {expected_pages} pages for "
            f"page_size={page_size}, got {len(raw_heads)}"
        )
    else:
        pass
    page_indices: list[int] = []
    for head in raw_heads:
        if head < 0 or head % page_size != 0:
            raise RuntimeError(
                f"{context_label} has unaligned page head {head} "
                f"for page_size={page_size}"
            )
        else:
            pass
        page_indices.append(head // page_size)
    return tuple(page_indices)


@dataclasses.dataclass(frozen=True)
class DecodeContinuation:
    """Serializable request state needed after the Prefill KV has arrived."""

    request_id: str
    transfer_id: str
    origin_input_ids: list[int]
    output_ids: list[int]
    vocab_size: int
    sampling_params: dict[str, object]
    stage_payload: dict[str, object]
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
    output_token_logprobs: list[list[float | int]] = dataclasses.field(
        default_factory=list
    )
    top_logprobs_num: int = 0
    token_ids_logprob: list[int] | None = None
    logprob_start_len: int = -1
    return_hidden_states: bool | Literal["last"] = False
    return_sampling_mask: bool = False
    return_routed_experts: bool = False
    return_indexer_topk: bool = False
    multimodal_resume: dict[str, object] | None = None
    priority: int | None = None
    version: int = CONTINUATION_VERSION

    def __post_init__(self) -> None:
        if self.version != CONTINUATION_VERSION:
            raise ValueError(f"unsupported decode continuation version {self.version}")
        else:
            pass
        if not self.request_id or not self.transfer_id:
            raise ValueError("decode continuation ids must be non-empty")
        else:
            pass
        if not self.output_ids:
            raise ValueError("decode continuation requires the Prefill token")
        else:
            pass
        if self.vocab_size <= 0:
            raise ValueError("decode continuation vocab_size must be positive")
        else:
            pass
        if self.priority is not None and type(self.priority) is not int:
            raise ValueError("decode continuation priority must be an integer or None")
        else:
            pass
        for name in ("sampling_params", "stage_payload"):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"decode continuation {name} must be a mapping")
            else:
                pass
        if self.multimodal_resume is not None and not isinstance(
            self.multimodal_resume, dict
        ):
            raise TypeError("decode continuation multimodal_resume must be a mapping")
        else:
            pass
        hidden_states = self.return_hidden_states
        if not isinstance(hidden_states, bool) and hidden_states != "last":
            raise ValueError(
                f"unsupported decode continuation "
                f"return_hidden_states {hidden_states!r}"
            )
        else:
            pass

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
        else:
            pass
        expected = {field.name for field in dataclasses.fields(cls)}
        unknown = set(decoded) - expected
        if unknown:
            raise ValueError(f"unknown decode continuation fields: {unknown}")
        else:
            pass
        try:
            return cls(**decoded)
        except TypeError as exc:
            raise ValueError(f"invalid decode continuation: {exc}") from exc


@dataclasses.dataclass
class ReservedKV:
    """Receiver-reserved KV slots paired with an idempotent row credit."""

    slots: torch.Tensor
    page_indices: tuple[int, ...]
    sequence_length: int
    row_credit_owner: DecodeKVReceiver | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    row_credit_held: bool = dataclasses.field(default=False, repr=False, compare=False)

    def release_row_credit(self) -> None:
        """Idempotently release the pre-reserved req_to_token_pool row credit."""
        if not self.row_credit_held or self.row_credit_owner is None:
            return
        else:
            pass
        self.row_credit_held = False
        self.row_credit_owner.release_row_credit()


@dataclasses.dataclass(frozen=True)
class DecodeAdmission:
    continuation: DecodeContinuation
    allocation: ReservedKV


@dataclasses.dataclass(frozen=True, kw_only=True)
class ParsedPrepareRequest:
    continuation: DecodeContinuation
    need_tokens: int
    page_count: int
    sequence_length: int


@dataclasses.dataclass(kw_only=True)
class DecodePreallocWaiter:
    request: KVTransferPrepareMessage
    parsed: ParsedPrepareRequest
    future: asyncio.Future[KVPageDestination]
    loop: asyncio.AbstractEventLoop


class DecodeKVReceiver:
    """Reserve destination KV pages and row credits in strict FIFO order."""

    def __init__(
        self,
        *,
        pool_id: str,
        allocator: BaseTokenToKVPoolAllocator,
        req_to_token_pool: ReqToTokenPool,
        admissions: deque[DecodeAdmission],
        resume_schema: str,
        lifecycle_lock: _thread.RLock | None = None,
        page_size: int = 1,
        decode_runway_tokens: int = DEFAULT_DECODE_RUNWAY_TOKENS,
        wakeup_notifier: WakeupNotifier | None = None,
        capacity_check: DecodeCapacityCheck | None = None,
    ) -> None:
        if page_size < 1:
            raise ValueError(
                f"decode KV receiver page_size must be >= 1, got {page_size}"
            )
        else:
            pass
        if decode_runway_tokens < 0:
            raise ValueError(
                f"decode_runway_tokens must be non-negative, got {decode_runway_tokens}"
            )
        else:
            pass
        self.pool_id = pool_id
        self.allocator = allocator
        self.req_to_token_pool = req_to_token_pool
        self.admissions = admissions
        self.resume_schema = resume_schema
        self.lock = lifecycle_lock or threading.RLock()
        self.page_size = page_size
        self.decode_runway_tokens = decode_runway_tokens
        self.wakeup_notifier = wakeup_notifier
        self.capacity_check = capacity_check
        self.reservations: dict[str, DecodeAdmission] = {}
        self.transfer_tombstones: dict[str, None] = {}
        self.reserved_request_rows = 0
        self.waiters: deque[DecodePreallocWaiter] = deque()
        self.waiter_transfer_ids: set[str] = set()
        self.drain_scheduled = False
        self.accepting_reservations = True
        self.closed = False
        if torch.cuda.is_available() and str(allocator.device).startswith("cuda"):
            with torch.cuda.device(allocator.device):
                self.allocation_stream: torch.cuda.Stream | None = torch.cuda.Stream(
                    device=allocator.device
                )
        else:
            self.allocation_stream = None

    def allocation_stream_context(self) -> AbstractContextManager[None]:
        if self.allocation_stream is not None:
            return torch.cuda.stream(self.allocation_stream)
        else:
            return nullcontext()

    def remember_finished_transfer(self, transfer_id: str) -> None:
        self.transfer_tombstones[transfer_id] = None
        if len(self.transfer_tombstones) > TRANSFER_TOMBSTONE_LIMIT:
            self.transfer_tombstones.pop(next(iter(self.transfer_tombstones)))
        else:
            pass

    def release_row_credit(self) -> None:
        with self.lock:
            if self.reserved_request_rows <= 0:
                raise RuntimeError("decode request row credit underflow")
            else:
                pass
            self.reserved_request_rows -= 1

    def parse_prepare_request(
        self, request: KVTransferPrepareMessage
    ) -> ParsedPrepareRequest:
        if request.target_pool_id != self.pool_id:
            raise ValueError(f"KV receiver for {self.pool_id!r} got another pool")
        else:
            pass
        encoded_continuation = request.metadata.get("decode_continuation")
        if not isinstance(encoded_continuation, (bytes, bytearray)):
            raise TypeError("KV transfer is missing decode continuation bytes")
        else:
            pass
        continuation = DecodeContinuation.decode(bytes(encoded_continuation))
        if (
            continuation.request_id != request.request_id
            or continuation.transfer_id != request.transfer_id
        ):
            raise ValueError("KV transfer and decode continuation ids differ")
        else:
            pass
        resume = continuation.multimodal_resume
        if resume is not None and resume.get("schema") != self.resume_schema:
            raise ValueError(
                f"unsupported multimodal resume schema {resume.get('schema')!r}"
            )
        else:
            pass

        page_count = len(request.source_page_indices)
        if page_count <= 0:
            raise ValueError("KV transfer contains no pages")
        else:
            pass
        sequence_length = len(continuation.origin_input_ids)
        if sequence_length <= 0:
            raise ValueError("PD cannot transfer an empty prompt")
        else:
            pass
        expected_pages = (sequence_length + self.page_size - 1) // self.page_size
        if page_count != expected_pages:
            raise ValueError(
                f"KV transfer page count {page_count} does not match expected "
                f"{expected_pages} for sequence_length={sequence_length}, "
                f"page_size={self.page_size}"
            )
        else:
            pass
        need_tokens = page_count * self.page_size
        pool_capacity_tokens = self.allocator.size
        if need_tokens > pool_capacity_tokens:
            raise ValueError(
                f"Request {request.request_id} exceeds the maximum number of tokens: "
                f"{need_tokens} > {pool_capacity_tokens}"
            )
        else:
            pass
        return ParsedPrepareRequest(
            continuation=continuation,
            need_tokens=need_tokens,
            page_count=page_count,
            sequence_length=sequence_length,
        )

    def check_reservation_allowed_locked(self, transfer_id: str) -> None:
        if self.closed:
            raise RuntimeError("decode KV receiver is closed")
        else:
            pass
        if not self.accepting_reservations:
            raise AdminDroppedRequestError()
        else:
            pass
        if (
            transfer_id in self.reservations
            or transfer_id in self.transfer_tombstones
            or transfer_id in self.waiter_transfer_ids
        ):
            raise RuntimeError(f"duplicate KV transfer {transfer_id!r}")
        else:
            pass

    def free_slots_on_allocation_stream(self, slots: torch.Tensor) -> None:
        with self.allocation_stream_context():
            self.allocator.free(slots)

    def try_allocate_locked(
        self,
        *,
        transfer_id: str,
        parsed: ParsedPrepareRequest,
    ) -> KVPageDestination | None:
        free_rows = self.req_to_token_pool.available_size() - self.reserved_request_rows
        if free_rows < 1:
            return None
        else:
            pass
        has_active = (
            self.capacity_check.has_active_decode_requests()
            if self.capacity_check is not None
            else bool(self.reservations or self.reserved_request_rows > 0)
        )
        # note (YzXiao): skip decode_runway_tokens when Decode has zero active requests so a single long prompt can still start.
        effective_runway_tokens = self.decode_runway_tokens if has_active else 0
        if (
            self.allocator.available_size()
            < parsed.need_tokens + effective_runway_tokens
        ):
            return None
        else:
            pass
        with self.allocation_stream_context():
            slots = self.allocator.alloc(parsed.need_tokens)
            if slots is None:
                return None
            else:
                pass
            raw_heads = slots[:: self.page_size].tolist()
        try:
            page_indices = page_indices_from_heads(
                raw_heads,
                expected_pages=parsed.page_count,
                page_size=self.page_size,
                context_label="decode KV allocator",
            )
        except Exception:
            self.free_slots_on_allocation_stream(slots)
            raise
        self.reserved_request_rows += 1
        allocation = ReservedKV(
            slots=slots,
            page_indices=page_indices,
            sequence_length=parsed.sequence_length,
            row_credit_owner=self,
            row_credit_held=True,
        )
        self.reservations[transfer_id] = DecodeAdmission(
            parsed.continuation, allocation
        )
        return KVPageDestination(self.pool_id, page_indices)

    def drain_all_waiters_locked(self) -> list[DecodePreallocWaiter]:
        dropped_waiters: list[DecodePreallocWaiter] = []
        while self.waiters:
            waiter = self.waiters.popleft()
            self.waiter_transfer_ids.discard(waiter.request.transfer_id)
            dropped_waiters.append(waiter)
        self.drain_scheduled = False
        return dropped_waiters

    def fail_waiter_threadsafe(
        self, waiter: DecodePreallocWaiter, error: BaseException
    ) -> None:
        if waiter.loop.is_closed():
            return
        else:
            pass
        try:
            waiter.loop.call_soon_threadsafe(fail_future, waiter.future, error)
        except RuntimeError:
            pass

    def cleanup_failed_waiter(self, waiter: DecodePreallocWaiter) -> None:
        with self.lock:
            was_head = bool(self.waiters) and self.waiters[0] is waiter
            try:
                self.waiters.remove(waiter)
            except ValueError:
                pass
            self.waiter_transfer_ids.discard(waiter.request.transfer_id)
            # note (YzXiao): drain_waiters may resolve waiter.future in the same loop tick that timeout or cancel fires; roll back the reservation so neither KV nor row credit leaks.
            if (
                waiter.future.done()
                and not waiter.future.cancelled()
                and waiter.future.exception() is None
            ):
                leaked = self.reservations.pop(waiter.request.transfer_id, None)
                if leaked is not None:
                    self.remember_finished_transfer(waiter.request.transfer_id)
                    self.free_slots_on_allocation_stream(leaked.allocation.slots)
                    leaked.allocation.release_row_credit()
                    should_notify = True
                else:
                    should_notify = was_head
            else:
                should_notify = was_head
        if should_notify:
            self.notify_capacity_released()
        else:
            pass

    async def reserve(
        self,
        request: KVTransferPrepareMessage,
        *,
        timeout_seconds: float,
    ) -> KVPageDestination:
        """Asynchronously reserve KV pages and a row credit with FIFO queuing."""
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        else:
            pass
        parsed = self.parse_prepare_request(request)
        loop = asyncio.get_running_loop()
        with self.lock:
            self.check_reservation_allowed_locked(request.transfer_id)
            if not self.waiters:
                destination = self.try_allocate_locked(
                    transfer_id=request.transfer_id,
                    parsed=parsed,
                )
                if destination is not None:
                    return destination
                else:
                    pass
            else:
                pass
            waiter = DecodePreallocWaiter(
                request=request,
                parsed=parsed,
                future=loop.create_future(),
                loop=loop,
            )
            self.waiters.append(waiter)
            self.waiter_transfer_ids.add(request.transfer_id)
        try:
            async with asyncio.timeout(timeout_seconds):
                return await waiter.future
        except TimeoutError as exc:
            self.cleanup_failed_waiter(waiter)
            raise QueueFullError(
                f"{QueueFullError.MESSAGE} Decode KV preallocation timed out "
                f"after {timeout_seconds:g}s"
            ) from exc
        except BaseException:
            self.cleanup_failed_waiter(waiter)
            raise

    def notify_capacity_released(self) -> None:
        """Schedule FIFO waiter drainage after KV pages or request rows are freed."""
        with self.lock:
            if not self.waiters or self.drain_scheduled:
                return
            else:
                pass
            loop = self.waiters[0].loop
            if loop.is_closed():
                return
            else:
                pass
            self.drain_scheduled = True
        try:
            loop.call_soon_threadsafe(self.drain_waiters)
        except RuntimeError:
            with self.lock:
                self.drain_scheduled = False

    def drain_waiters(self) -> None:
        """Wake queued waiters in strict FIFO order while capacity allows."""
        with self.lock:
            self.drain_scheduled = False
            if self.closed or not self.accepting_reservations:
                return
            else:
                pass
            while self.waiters:
                waiter = self.waiters[0]
                if waiter.future.done():
                    self.waiters.popleft()
                    self.waiter_transfer_ids.discard(waiter.request.transfer_id)
                    continue
                else:
                    pass
                try:
                    destination = self.try_allocate_locked(
                        transfer_id=waiter.request.transfer_id,
                        parsed=waiter.parsed,
                    )
                except Exception as exc:
                    self.waiters.popleft()
                    self.waiter_transfer_ids.discard(waiter.request.transfer_id)
                    fail_future(waiter.future, exc)
                    continue
                if destination is None:
                    break
                else:
                    pass
                self.waiters.popleft()
                self.waiter_transfer_ids.discard(waiter.request.transfer_id)
                waiter.future.set_result(destination)

    def cancel_request_waiters(self, request_id: str) -> None:
        """Fail queued waiters for an aborted request and unblock later waiters."""
        cancelled_waiters: list[DecodePreallocWaiter] = []
        was_head_removed = False
        with self.lock:
            if self.waiters and self.waiters[0].request.request_id == request_id:
                was_head_removed = True
            else:
                pass
            remaining_waiters: deque[DecodePreallocWaiter] = deque()
            while self.waiters:
                waiter = self.waiters.popleft()
                if waiter.request.request_id == request_id:
                    self.waiter_transfer_ids.discard(waiter.request.transfer_id)
                    cancelled_waiters.append(waiter)
                else:
                    remaining_waiters.append(waiter)
            self.waiters = remaining_waiters
        if cancelled_waiters:
            error = KVTransferCancelled(
                f"request {request_id!r} aborted while waiting for decode KV"
            )
            for waiter in cancelled_waiters:
                self.fail_waiter_threadsafe(waiter, error)
        else:
            pass
        if was_head_removed:
            self.notify_capacity_released()
        else:
            pass

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None:
        commit_failed = False
        with self.lock:
            reservation = self.reservations.pop(request.transfer_id, None)
            if reservation is None:
                raise RuntimeError(
                    f"commit for unknown KV transfer {request.transfer_id!r}"
                )
            else:
                pass
            self.remember_finished_transfer(request.transfer_id)
            if (
                self.closed
                or request.request_id != reservation.continuation.request_id
                or request.target_pool_id != self.pool_id
                or destination.pool_id != self.pool_id
                or reservation.allocation.page_indices != destination.page_indices
            ):
                self.free_slots_on_allocation_stream(reservation.allocation.slots)
                reservation.allocation.release_row_credit()
                commit_failed = True
            else:
                self.admissions.append(reservation)
        if commit_failed:
            self.notify_capacity_released()
            raise RuntimeError("KV commit does not match a live reservation")
        else:
            pass
        if self.wakeup_notifier is not None:
            self.wakeup_notifier.notify_pd_wakeup()
        else:
            pass

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None:
        # destination is required by the KVReceiver protocol; reservation lookup uses transfer_id.
        freed = False
        with self.lock:
            reservation = self.reservations.pop(request.transfer_id, None)
            if reservation is not None:
                self.remember_finished_transfer(request.transfer_id)
                self.free_slots_on_allocation_stream(reservation.allocation.slots)
                reservation.allocation.release_row_credit()
                freed = True
            else:
                pass
        if freed:
            self.notify_capacity_released()
        else:
            pass
        if not freed or isinstance(error, KVTransferCancelled):
            logger.debug(f"KV receive aborted for {request.request_id}: {error}")
        else:
            logger.warning(f"KV receive aborted for {request.request_id}: {error}")

    def has_reservations(self) -> bool:
        with self.lock:
            return bool(self.reservations) or self.reserved_request_rows > 0

    @contextmanager
    def suspend_reservations(self) -> Generator[None, None, None]:
        """Reject new reservations and drop queued waiters during admin operations."""
        with self.lock:
            was_accepting = self.accepting_reservations
            self.accepting_reservations = False
            dropped_waiters = self.drain_all_waiters_locked()
        for waiter in dropped_waiters:
            self.fail_waiter_threadsafe(waiter, AdminDroppedRequestError())
        try:
            yield
        finally:
            with self.lock:
                self.accepting_reservations = was_accepting and not self.closed

    def close(self) -> None:
        # note (YzXiao): CommEngine must finish or abort an in-flight copy before its pages can be freed, so closing only gates new reservations and fails queued waiters.
        with self.lock:
            self.accepting_reservations = False
            self.closed = True
            dropped_waiters = self.drain_all_waiters_locked()
        error = RuntimeError("decode KV receiver is closed")
        for waiter in dropped_waiters:
            self.fail_waiter_threadsafe(waiter, error)
