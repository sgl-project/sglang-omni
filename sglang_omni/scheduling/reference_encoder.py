# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import concurrent.futures
import json
import logging
import queue as _queue_mod
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar, cast

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
ArtifactT = TypeVar("ArtifactT")
StoredT = TypeVar("StoredT")


@dataclass(frozen=True)
class ReferenceEncodeKey:
    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str
    input_key: str
    options_key: str = ""

    def to_string(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class ReferenceEncodeHook(Generic[InputT, ArtifactT, StoredT]):
    def normalize_input(self, raw_input: Any) -> InputT:
        raise NotImplementedError

    def cache_key(self, item: InputT) -> ReferenceEncodeKey | None:
        raise NotImplementedError

    def encode_one(self, item: InputT) -> ArtifactT:
        raise NotImplementedError

    def store_artifact(self, artifact: ArtifactT) -> StoredT:
        raise NotImplementedError

    def load_artifact(self, stored: StoredT) -> ArtifactT:
        raise NotImplementedError

    def revalidate(self, item: InputT, key: ReferenceEncodeKey) -> bool:
        return True

    def can_encode_batch(self) -> bool:
        return False

    def encode_batch(self, items: list[InputT]) -> list[ArtifactT]:
        return [self.encode_one(item) for item in items]


class KeyedReferenceEncodeHook(ReferenceEncodeHook[InputT, ArtifactT, StoredT]):
    """Defaults for hooks with structured identity and option keys."""

    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str

    def normalize_input(self, raw_input: Any) -> InputT:
        return cast(InputT, raw_input)

    def input_key(self, item: InputT) -> str | None:
        raise NotImplementedError

    def options_key(self, item: InputT) -> str:
        return ""

    def cache_key(self, item: InputT) -> ReferenceEncodeKey | None:
        input_key = self.input_key(item)
        if input_key is None:
            return None
        return ReferenceEncodeKey(
            model_id=self.model_id,
            model_revision=self.model_revision,
            encoder_id=self.encoder_id,
            encoder_config_hash=self.encoder_config_hash,
            artifact_kind=self.artifact_kind,
            input_key=input_key,
            options_key=self.options_key(item),
        )

    def revalidate(self, item: InputT, key: ReferenceEncodeKey) -> bool:
        return (
            self.input_key(item) == key.input_key
            and self.options_key(item) == key.options_key
        )


class TensorReferenceEncodeHook(
    KeyedReferenceEncodeHook[InputT, torch.Tensor, torch.Tensor]
):
    """Defaults for reference encoders that cache CPU tensor artifacts."""

    storage_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None

    def store_artifact(self, artifact: torch.Tensor) -> torch.Tensor:
        return artifact.detach().to(device="cpu", dtype=self.storage_dtype, copy=True)

    def load_artifact(self, stored: torch.Tensor) -> torch.Tensor:
        return stored.detach().to(dtype=self.output_dtype, copy=True)


def _fresh_exception(exc: BaseException) -> BaseException:
    try:
        fresh = type(exc)(*getattr(exc, "args", ()))
    except Exception:
        fresh = RuntimeError(str(exc))
    for note in getattr(exc, "__notes__", ()):
        add_note = getattr(fresh, "add_note", None)
        if callable(add_note):
            add_note(note)
    return fresh


class ReferenceEncodeService(Generic[InputT, ArtifactT, StoredT]):
    _LOG_INTERVAL_S = 60.0

    def __init__(
        self,
        hook: ReferenceEncodeHook[InputT, ArtifactT, StoredT],
        *,
        max_items: int | None = 256,
        max_bytes: int | None = 64 * 1024 * 1024,
        timeout_s: float = 130.0,
        log_prefix: str | None = None,
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 0.0,
        batch_worker_name: str = "reference-encode-batch",
    ) -> None:
        if max_items is not None and max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if max_batch_wait_ms < 0:
            raise ValueError(f"max_batch_wait_ms must be >= 0, got {max_batch_wait_ms}")
        self._hook = hook
        self._cache = StageOutputCache(max_size=max_items, max_bytes=max_bytes)
        self._timeout_s = float(timeout_s)
        self._log_prefix = log_prefix
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future[StoredT]] = {}
        self._hits = 0
        self._misses = 0
        self._merged = 0
        self._failed = 0
        self._uncacheable = 0
        self._batches = 0
        self._batched_items = 0
        self._last_log_time = 0.0

        # Only distinct cache-miss leaders reach the queue; hits and same-key
        # followers resolve first, so batching preserves lookup order/single-flight.
        self._max_batch_size = int(max_batch_size)
        self._max_batch_wait_s = float(max_batch_wait_ms) / 1000.0
        self._batching = self._max_batch_size > 1 and bool(hook.can_encode_batch())
        self._batch_queue: (
            _queue_mod.Queue[tuple[InputT, concurrent.futures.Future[ArtifactT]] | None]
            | None
        ) = None
        self._batch_thread: threading.Thread | None = None
        if self._batching:
            self._batch_queue = _queue_mod.Queue()
            self._batch_thread = threading.Thread(
                target=self._batch_worker,
                name=batch_worker_name,
                daemon=True,
            )
            self._batch_thread.start()

    @property
    def batching_enabled(self) -> bool:
        return self._batching

    def close(self) -> None:
        if self._batch_queue is not None:
            self._batch_queue.put(None)
        thread = self._batch_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        close = getattr(self._hook, "close", None)
        if callable(close):
            close()

    def _encode_leader(self, item: InputT) -> ArtifactT:
        if self._batch_queue is None:
            return self._hook.encode_one(item)
        future: concurrent.futures.Future[ArtifactT] = concurrent.futures.Future()
        self._batch_queue.put((item, future))
        return future.result(timeout=self._timeout_s)

    def _drain_batch(
        self,
    ) -> tuple[list[tuple[InputT, concurrent.futures.Future[ArtifactT]]], bool]:
        assert self._batch_queue is not None
        first = self._batch_queue.get()
        if first is None:
            return [], True
        batch = [first]
        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            try:
                entry = self._batch_queue.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    entry = self._batch_queue.get(timeout=remaining)
                except _queue_mod.Empty:
                    break
            if entry is None:
                return batch, True
            batch.append(entry)
        return batch, False

    def _batch_worker(self) -> None:
        while True:
            try:
                batch, stopping = self._drain_batch()
            except Exception:
                logger.exception("reference encode batch worker: drain failed")
                continue
            if batch:
                try:
                    results: list[Any] = self._encode_batch([item for item, _ in batch])
                except BaseException as exc:  # never let the worker die silently
                    logger.exception("reference encode batch worker: encode failed")
                    results = [exc] * len(batch)
                for (_, future), outcome in zip(batch, results):
                    if future.cancelled():
                        continue
                    if isinstance(outcome, BaseException):
                        future.set_exception(_fresh_exception(outcome))
                    else:
                        future.set_result(cast(ArtifactT, outcome))
            if stopping:
                self._drain_pending_on_shutdown()
                return

    def _drain_pending_on_shutdown(self) -> None:
        """Fail queued waiters instead of leaving them to time out."""
        if self._batch_queue is None:
            return
        while True:
            try:
                entry = self._batch_queue.get_nowait()
            except _queue_mod.Empty:
                return
            if entry is None:
                continue
            _, future = entry
            if not future.cancelled():
                future.set_exception(
                    RuntimeError("reference encode service is shutting down")
                )

    def _encode_batch(self, items: list[InputT]) -> list[Any]:
        """Encode a drained batch, falling back to per-item encodes on failure."""
        try:
            artifacts = self._hook.encode_batch(items)
            if len(artifacts) != len(items):
                raise RuntimeError(
                    f"encode_batch returned {len(artifacts)} artifacts for "
                    f"{len(items)} items"
                )
            with self._lock:
                self._batches += 1
                self._batched_items += len(items)
            return list(artifacts)
        except Exception:
            logger.exception(
                "%s batched reference encode failed; retrying per item",
                self._log_prefix or "reference encode",
            )
        results: list[Any] = []
        for item in items:
            try:
                results.append(self._hook.encode_one(item))
            except Exception as exc:
                results.append(exc)
        return results

    def get_or_encode(self, raw_input: Any, *, desc: str | None = None) -> ArtifactT:
        item = self._hook.normalize_input(raw_input)
        key = self._hook.cache_key(item)
        if key is None:
            with self._lock:
                self._uncacheable += 1
            try:
                return self._encode_leader(item)
            except BaseException as exc:
                self._add_exception_note(exc, desc)
                with self._lock:
                    self._failed += 1
                raise

        cache_key = key.to_string()
        leader_fut: concurrent.futures.Future[StoredT] | None = None
        follower_fut: concurrent.futures.Future[StoredT] | None = None
        stored: StoredT | None = None
        with self._lock:
            stored = self._cache.get(cache_key)
            if stored is not None:
                self._hits += 1
            elif cache_key in self._inflight:
                self._merged += 1
                follower_fut = self._inflight[cache_key]
            else:
                self._misses += 1
                leader_fut = concurrent.futures.Future()
                self._inflight[cache_key] = leader_fut

        if stored is not None:
            self._maybe_log()
            return self._hook.load_artifact(stored)

        if follower_fut is not None:
            try:
                stored = follower_fut.result(timeout=self._timeout_s)
            except concurrent.futures.TimeoutError as exc:
                self._add_exception_note(exc, desc)
                raise
            except BaseException as exc:
                self._add_exception_note(exc, desc)
                raise _fresh_exception(exc) from exc
            return self._hook.load_artifact(stored)

        assert leader_fut is not None
        # revalidate() and cache.put() run inside the same guard as encode: any
        # exception here must still drop the in-flight entry and fail the future,
        # otherwise same-key followers block on a future that never resolves and
        # every later request for this key re-follows a dead leader (permanent
        # poison + timeout-length hangs). A hook's revalidate() may legitimately
        # raise (e.g. a reference file mutated during the encode window).
        try:
            artifact = self._encode_leader(item)
            stored = self._hook.store_artifact(artifact)
            should_cache = self._hook.revalidate(item, key)
            with self._lock:
                if should_cache:
                    self._cache.put(cache_key, stored)
                self._inflight.pop(cache_key, None)
        except BaseException as exc:
            self._add_exception_note(exc, desc)
            with self._lock:
                self._inflight.pop(cache_key, None)
                self._failed += 1
            leader_fut.set_exception(exc)
            raise
        leader_fut.set_result(stored)
        self._maybe_log()
        return self._hook.load_artifact(stored)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache.current_bytes,
                "evictions": self._cache.eviction_count,
                "failed": self._failed,
                "uncacheable": self._uncacheable,
                "batches": self._batches,
                "batched_items": self._batched_items,
            }

    @staticmethod
    def _add_exception_note(exc: BaseException, desc: str | None) -> None:
        if not desc:
            return
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(f"Reference encode context: {desc}")

    def _maybe_log(self) -> None:
        if self._log_prefix is None:
            return
        now = time.monotonic()
        if now - self._last_log_time < self._LOG_INTERVAL_S:
            return
        with self._lock:
            if now - self._last_log_time < self._LOG_INTERVAL_S:
                return
            self._last_log_time = now
            stats = {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache.current_bytes,
                "evictions": self._cache.eviction_count,
                "failed": self._failed,
                "uncacheable": self._uncacheable,
                "batches": self._batches,
                "batched_items": self._batched_items,
            }
        logger.info("%s reference encode stats: %s", self._log_prefix, stats)
