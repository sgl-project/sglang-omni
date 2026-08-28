# SPDX-License-Identifier: Apache-2.0
"""Generic Prefill-Decode request handoff contract for PR 2."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import msgspec

from sglang_omni.comm.kv_transfer import (
    KVPageDestination,
    KVReceiver,
    KVTransferPrepareMessage,
)

logger = logging.getLogger(__name__)

CONTINUATION_VERSION = "pd-continuation-v1"


class ContinuationSchemaError(ValueError):
    """Raised when a continuation payload has an unknown or incompatible schema."""


class PDCapabilityError(ValueError):
    """Raised when a continuation cannot be handled by the current PD policy."""


RankReadyCallback = Callable[["PendingHandoff"], None]
CleanupCallback = Callable[["PendingHandoff", str], None]


@dataclasses.dataclass(frozen=True)
class DecodeContinuation:
    """Per-request state a decode scheduler needs to resume generation.

    This is not a reconstructed `Req`; PR 3 will build `Req` objects from it.
    All fields are msgpack-serializable.
    """

    request_id: str
    transfer_id: str
    origin_input_ids: list[int]
    output_ids: list[int]
    vocab_size: int
    sampling_params: dict[str, Any]
    cached_tokens: int

    version: str = CONTINUATION_VERSION
    # Absolute Unix wall-clock deadline. Unlike a monotonic timestamp this is
    # meaningful on another host; multi-host PD deployments must keep their
    # wall clocks synchronized. None disables the end-to-end waiting deadline.
    deadline_unix_s: float | None = None
    origin_input_ids_unpadded: list[int] | None = None
    eos_token_ids: list[int] | None = None
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
    custom_logit_processor: str | None = None
    input_embeds_are_projected: bool = False
    prefill_input_embeds_shape: tuple[int, ...] | None = None
    speculative: bool = False
    multimodal_resume: dict[str, Any] | None = None
    stage_payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.version != CONTINUATION_VERSION:
            raise ContinuationSchemaError(
                f"unsupported continuation version {self.version!r}; "
                f"expected {CONTINUATION_VERSION!r}"
            )
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.transfer_id:
            raise ValueError("transfer_id must be non-empty")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.cached_tokens < 0:
            raise ValueError(
                f"cached_tokens must be non-negative, got {self.cached_tokens}"
            )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecodeContinuation":
        if not isinstance(data, dict):
            raise ContinuationSchemaError("continuation must be a dictionary")
        expected = {f.name for f in dataclasses.fields(cls)}
        extra = set(data) - expected
        if extra:
            raise ContinuationSchemaError(f"unknown continuation keys: {extra}")
        try:
            return cls(**{k: v for k, v in data.items() if k in expected})
        except TypeError as exc:
            raise ContinuationSchemaError(f"invalid continuation: {exc}") from exc


def encode_continuation(continuation: DecodeContinuation) -> bytes:
    return msgspec.msgpack.encode(continuation.to_dict())


def decode_continuation(data: bytes) -> DecodeContinuation:
    try:
        decoded = msgspec.msgpack.decode(data)
    except Exception as exc:
        raise ContinuationSchemaError("continuation msgpack decode failed") from exc
    if not isinstance(decoded, dict):
        raise ContinuationSchemaError("decoded continuation is not a dict")
    return DecodeContinuation.from_dict(decoded)


def validate_continuation(
    continuation: DecodeContinuation,
    *,
    source_tp_size: int,
    target_tp_size: int,
    is_local: bool = True,
    allowed_resume_schemas: frozenset[str] = frozenset(),
) -> None:
    """Raise `PDCapabilityError` if the continuation is not supported in PR 2."""

    if source_tp_size != target_tp_size:
        raise PDCapabilityError(
            f"TP size mismatch: prefill={source_tp_size} decode={target_tp_size}"
        )

    if not is_local:
        raise PDCapabilityError("cross-node PD handoff is not supported")

    if continuation.input_embeds_are_projected:
        raise PDCapabilityError("projected input embeddings are not supported")

    if continuation.speculative:
        raise PDCapabilityError("speculative decoding is not supported")

    if continuation.multimodal_resume is not None:
        schema = (
            continuation.multimodal_resume.get("schema")
            if isinstance(continuation.multimodal_resume, dict)
            else None
        )
        if schema not in allowed_resume_schemas:
            raise PDCapabilityError(
                f"multimodal resume schema {schema!r} is not supported"
            )

    sampling_params = continuation.sampling_params or {}
    if any(
        sampling_params.get(k)
        for k in ("grammar", "json_schema", "regex", "structural_tag", "ebnf")
    ):
        raise PDCapabilityError("grammar / structured-output sampling is not supported")

    if continuation.custom_logit_processor:
        raise PDCapabilityError("custom logit processor is not supported")


@dataclasses.dataclass
class PendingHandoff:
    request_id: str
    transfer_id: str
    continuation: DecodeContinuation | None = None
    continuation_expected: bool = True
    kv_committed: bool = False
    aborted: bool = False
    rank_ready: bool = False
    dispatching: bool = False
    deadline: float | None = None
    timeout_handle: Any | None = None


class PDHandoffController:
    """Join a per-rank continuation with a KV receiver commit.

    Fires the `rank_ready_callback` exactly once when this rank has both the
    continuation and the KV commit. The callback is rank-local; the production
    runtime currently admits TP=1 requests.

    The controller is callback-driven and thread-safe.  When no asyncio loop is
    running the caller must use `check_timeouts()`.
    """

    def __init__(
        self,
        *,
        default_timeout_s: float = 30.0,
        rank_ready_callback: RankReadyCallback | None = None,
        cleanup_callback: CleanupCallback | None = None,
    ) -> None:
        self._default_timeout_s = default_timeout_s
        self._rank_ready_callback = rank_ready_callback
        self._cleanup_callback = cleanup_callback
        self._lock = threading.RLock()
        self._pending: dict[str, PendingHandoff] = {}

    def start_handoff(
        self,
        request_id: str,
        transfer_id: str,
        timeout_s: float | None = None,
    ) -> PendingHandoff:
        with self._lock:
            if request_id in self._pending:
                pending = self._pending[request_id]
                if pending.transfer_id != transfer_id:
                    raise ContinuationSchemaError(
                        f"request {request_id!r} already has active transfer "
                        f"{pending.transfer_id!r}, got {transfer_id!r}"
                    )
                return pending

            effective_timeout = (
                timeout_s if timeout_s is not None else self._default_timeout_s
            )
            deadline = None
            handle = None
            if effective_timeout is not None:
                deadline = time.monotonic() + effective_timeout
                try:
                    loop = asyncio.get_running_loop()
                    handle = loop.call_later(
                        effective_timeout,
                        self._on_timeout,
                        request_id,
                        transfer_id,
                    )
                except RuntimeError:
                    pass

            pending = PendingHandoff(
                request_id=request_id,
                transfer_id=transfer_id,
                deadline=deadline,
                timeout_handle=handle,
            )
            self._pending[request_id] = pending
            logger.debug(
                "started handoff request=%s transfer=%s", request_id, transfer_id
            )
            return pending

    def set_continuation(
        self,
        request_id: str,
        continuation: DecodeContinuation,
    ) -> None:
        """Accept a decoded continuation for a request."""
        ready = None
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                logger.debug(
                    "ignoring continuation for inactive request %s", request_id
                )
                return
            if pending.dispatching or pending.aborted:
                return
            if continuation.request_id != request_id:
                raise ContinuationSchemaError(
                    f"continuation request_id {continuation.request_id!r} does not "
                    f"match handoff request_id {request_id!r}"
                )
            if continuation.transfer_id != pending.transfer_id:
                raise ContinuationSchemaError(
                    f"continuation transfer_id {continuation.transfer_id!r} does not "
                    f"match active transfer_id {pending.transfer_id!r}"
                )
            if pending.continuation is not None:
                logger.warning(
                    "duplicate continuation for request %s ignored", request_id
                )
                return

            pending.continuation = continuation
            logger.debug("continuation set for request=%s", request_id)
            ready = self._take_ready_locked(pending)
        if ready is not None:
            self._dispatch_ready(ready)

    def set_continuation_not_required(
        self,
        request_id: str,
        transfer_id: str | None = None,
    ) -> None:
        """Mark a non-rank-0 handoff as not expecting a continuation payload."""
        ready = None
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                return
            if pending.dispatching or pending.aborted:
                return
            if transfer_id is not None and transfer_id != pending.transfer_id:
                logger.debug(
                    "ignoring continuation marker for stale transfer request=%s "
                    "transfer=%s",
                    request_id,
                    transfer_id,
                )
                return
            pending.continuation_expected = False
            logger.debug("continuation not required for request=%s", request_id)
            ready = self._take_ready_locked(pending)
        if ready is not None:
            self._dispatch_ready(ready)

    def set_kv_committed(
        self,
        request_id: str,
        transfer_id: str | None = None,
    ) -> None:
        """Notify that the KV receiver has committed the transfer."""
        ready = None
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                logger.debug(
                    "ignoring kv_committed for inactive request %s", request_id
                )
                return
            if pending.dispatching or pending.aborted:
                return
            if transfer_id is not None and transfer_id != pending.transfer_id:
                logger.debug(
                    "ignoring kv_committed for stale transfer request=%s transfer=%s",
                    request_id,
                    transfer_id,
                )
                return
            if pending.kv_committed:
                logger.warning("duplicate kv_committed for request %s", request_id)
                return
            pending.kv_committed = True
            logger.debug("kv committed for request=%s", request_id)
            ready = self._take_ready_locked(pending)
        if ready is not None:
            self._dispatch_ready(ready)

    def abort(
        self,
        request_id: str,
        reason: str = "abort",
        transfer_id: str | None = None,
    ) -> None:
        """Abort a pending handoff before it becomes rank-ready."""
        aborted = None
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                return
            if pending.dispatching or pending.aborted:
                return
            if transfer_id is not None and transfer_id != pending.transfer_id:
                logger.debug(
                    "ignoring abort for stale transfer request=%s transfer=%s",
                    request_id,
                    transfer_id,
                )
                return
            aborted = self._take_abort_locked(pending)
        if aborted is not None:
            self._run_cleanup(aborted, reason)

    def check_timeouts(self) -> None:
        """Synchronous timeout helper for callers without a running event loop."""
        now = time.monotonic()
        expired = []
        with self._lock:
            for request_id, pending in list(self._pending.items()):
                if pending.deadline is not None and now >= pending.deadline:
                    logger.warning("handoff timeout for request=%s", request_id)
                    expired.append(self._take_abort_locked(pending))
        for pending in expired:
            self._run_cleanup(pending, "timeout")

    def get_pending(self, request_id: str) -> PendingHandoff | None:
        with self._lock:
            return self._pending.get(request_id)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def _take_ready_locked(self, pending: PendingHandoff) -> PendingHandoff | None:
        # Note (Yue Yin): Local commit is the ownership boundary, so waiting for
        # the sender ACK here would unnecessarily serialize Decode admission.
        if (
            (pending.continuation is not None or not pending.continuation_expected)
            and pending.kv_committed
            and not pending.dispatching
            and not pending.aborted
        ):
            pending.dispatching = True
            self._cancel_timeout(pending)
            return pending
        return None

    def _dispatch_ready(self, pending: PendingHandoff) -> None:
        try:
            if self._rank_ready_callback is not None:
                self._rank_ready_callback(pending)
        except Exception as exc:
            logger.exception("rank_ready callback failed for %s", pending.request_id)
            pending.aborted = True
            self._run_cleanup(pending, f"rank_ready callback failed: {exc}")
            return
        with self._lock:
            pending.rank_ready = True
            self._remove_locked(pending)
        logger.debug(
            "rank_ready request=%s transfer=%s",
            pending.request_id,
            pending.transfer_id,
        )

    def _on_timeout(self, request_id: str, transfer_id: str) -> None:
        expired = None
        with self._lock:
            pending = self._pending.get(request_id)
            if (
                pending is None
                or pending.transfer_id != transfer_id
                or pending.aborted
                or pending.dispatching
            ):
                return
            expired = self._take_abort_locked(pending)
        if expired is not None:
            self._run_cleanup(expired, "timeout")

    def _take_abort_locked(self, pending: PendingHandoff) -> PendingHandoff:
        pending.aborted = True
        self._cancel_timeout(pending)
        return pending

    def _run_cleanup(self, pending: PendingHandoff, reason: str) -> None:
        try:
            if self._cleanup_callback is not None:
                self._cleanup_callback(pending, reason)
        except Exception:
            logger.exception("cleanup callback failed for %s", pending.request_id)
        finally:
            with self._lock:
                self._remove_locked(pending)
        logger.debug("handoff aborted request=%s reason=%s", pending.request_id, reason)

    def _remove_locked(self, pending: PendingHandoff) -> None:
        if self._pending.get(pending.request_id) is pending:
            del self._pending[pending.request_id]

    def _cancel_timeout(self, pending: PendingHandoff) -> None:
        handle = pending.timeout_handle
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
            pending.timeout_handle = None
        pending.deadline = None


class ContinuationAwareKVReceiver:
    """Wrap a `KVReceiver` and feed continuation metadata to `PDHandoffController`.

    `CommEngine` treats `KVTransferPrepareMessage.metadata` as opaque.  This adapter
    extracts the rank-0 continuation and signals non-rank-0 shards so the
    controller can decide when this rank is ready.
    """

    def __init__(
        self,
        inner: KVReceiver,
        controller: PDHandoffController,
        *,
        source_tp_size: int = 1,
        target_tp_size: int = 1,
        is_local: bool = True,
        allowed_resume_schemas: frozenset[str] = frozenset(),
        ownership_reserve: Callable[[DecodeContinuation], None] | None = None,
        ownership_committed: Callable[[str], None] | None = None,
        ownership_release: Callable[[str], None] | None = None,
    ) -> None:
        self._inner = inner
        self._controller = controller
        self._source_tp_size = source_tp_size
        self._target_tp_size = target_tp_size
        self._is_local = is_local
        self._allowed_resume_schemas = allowed_resume_schemas
        self._ownership_reserve = ownership_reserve
        self._ownership_committed = ownership_committed
        self._ownership_release = ownership_release

    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination:
        continuation, continuation_expected = self._decode_metadata(request)
        ownership_reserved = False
        try:
            if continuation is not None and self._ownership_reserve is not None:
                self._ownership_reserve(continuation)
                ownership_reserved = True
            self._controller.start_handoff(request.request_id, request.transfer_id)
            if continuation is not None:
                self._controller.set_continuation(request.request_id, continuation)
            elif not continuation_expected:
                self._controller.set_continuation_not_required(
                    request.request_id, request.transfer_id
                )
            return self._inner.reserve(request)
        except Exception:
            self._controller.abort(
                request.request_id,
                reason="KV reservation failed",
                transfer_id=request.transfer_id,
            )
            if ownership_reserved and self._ownership_release is not None:
                self._ownership_release(request.request_id)
            raise

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None:
        self._inner.commit(request, destination)
        if self._ownership_committed is not None:
            self._ownership_committed(request.request_id)
        self._controller.set_kv_committed(request.request_id, request.transfer_id)

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None:
        if self._ownership_release is None:
            self._inner.abort(request, destination, error)
            self._controller.abort(
                request.request_id,
                reason=str(error) or type(error).__name__,
                transfer_id=request.transfer_id,
            )
        else:
            self._controller.abort(
                request.request_id,
                reason=str(error) or type(error).__name__,
                transfer_id=request.transfer_id,
            )
            self._ownership_release(request.request_id)

    def _decode_metadata(
        self, request: KVTransferPrepareMessage
    ) -> tuple[DecodeContinuation | None, bool]:
        present = request.metadata.get("pd_continuation_present")
        if present is False:
            return None, False

        raw = request.metadata.get("pd_continuation")
        if raw is None:
            if present is True:
                raise ContinuationSchemaError(
                    "pd_continuation required by metadata but missing"
                )
            return None, True

        if not isinstance(raw, (bytes, bytearray)):
            raise ContinuationSchemaError("pd_continuation metadata must be bytes")

        continuation = decode_continuation(bytes(raw))
        if continuation.request_id != request.request_id:
            raise ContinuationSchemaError(
                f"prepare request_id {request.request_id!r} does not match "
                f"continuation request_id {continuation.request_id!r}"
            )
        if continuation.transfer_id != request.transfer_id:
            raise ContinuationSchemaError(
                f"prepare transfer_id {request.transfer_id!r} does not match "
                f"continuation transfer_id {continuation.transfer_id!r}"
            )
        validate_continuation(
            continuation,
            source_tp_size=self._source_tp_size,
            target_tp_size=self._target_tp_size,
            is_local=self._is_local,
            allowed_resume_schemas=self._allowed_resume_schemas,
        )
        return continuation, True


class PrefillContinuationProducer:
    """Build per-rank metadata for `CommEngine.send_kv_pages`.

    The full `DecodeContinuation` is embedded only on rank 0.  Other ranks carry
    only a presence flag so the decode-side adapter knows no payload is expected.
    """

    def __init__(self, tp_size: int = 1) -> None:
        self._tp_size = tp_size

    def prepare_rank_metadata(
        self,
        continuation: DecodeContinuation,
        tp_rank: int,
    ) -> dict[str, Any]:
        if not 0 <= tp_rank < self._tp_size:
            raise ValueError(
                f"tp_rank {tp_rank} out of range for tp_size {self._tp_size}"
            )
        if tp_rank == 0:
            return {
                "pd_continuation": encode_continuation(continuation),
                "pd_continuation_present": True,
            }
        return {"pd_continuation_present": False}
