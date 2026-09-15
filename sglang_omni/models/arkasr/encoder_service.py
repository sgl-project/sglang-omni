# SPDX-License-Identifier: Apache-2.0
"""Precompute and cache LM-ready ARK-ASR audio embeddings.

Audio encoding runs before LM admission on a dedicated worker thread and CUDA
stream. The service batches queued requests, deduplicates identical audio, and
admits each request only after its complete embedding is attached.

The service batch limit controls how many queued requests enter one
``get_audio_feature`` call. The model's ``encoder_max_batch_size`` separately
bounds the microbatches used to execute that call.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import logging
import queue
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputFormat

from sglang_omni.scheduling.pre_lm_encoder import PreLMEncoderService, QueueEntry
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = 4096
_CACHE_MAX_BYTES = 2 * 1024**3
_SHUTDOWN = object()

# WhisperFeatureExtractor identity fields; a change to any of these changes the
# mel features and therefore the embedding.
_FRONTEND_CONFIG_FIELDS = (
    "feature_size",
    "sampling_rate",
    "hop_length",
    "chunk_length",
    "n_fft",
    "nb_max_frames",
    "padding_value",
)


@dataclass(frozen=True)
class _DetachedFailure:
    exception: Exception
    formatted_traceback: str


def build_cache_namespace(
    model: Any,
    *,
    model_path: str,
    feature_extractor: Any,
    mm_attention_backend: str | None,
) -> str:
    """Digest identifying this process's encoder pipeline for cache keying."""
    config = getattr(model, "config", None)
    if hasattr(config, "to_dict"):
        model_config: Any = config.to_dict()
    else:
        model_config = repr(config)
    reference = next(model.audio_encoder.parameters())
    payload = {
        "model_path": model_path,
        # ArkasrConfig.to_dict carries merge_factor and the nested
        # whisper_config, so an adapter or tower change re-keys the cache.
        "model_config": model_config,
        "frontend": {
            field: getattr(feature_extractor, field, None)
            for field in _FRONTEND_CONFIG_FIELDS
        },
        "dtype": str(reference.dtype),
        "mm_attention_backend": mm_attention_backend or "default",
        "device_type": reference.device.type,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _expected_audio_tokens(item: Any) -> int | None:
    """Audio placeholder token count for an item (rows the LM expects)."""
    num_tokens = getattr(item, "num_audio_tokens", None)
    return int(num_tokens) if num_tokens is not None else None


def _text_hidden_size(model: Any) -> int:
    # ArkasrConfig subclasses Qwen2Config, so the LM hidden size is top level;
    # it is also the adapter's output dim (ArkAudioMLPAdapter.adapting).
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("ARK-ASR config does not expose hidden_size")
    return int(hidden_size)


class ArkasrPreLMEncoderService(PreLMEncoderService[Any, torch.Tensor, torch.Tensor]):
    """Encode before admission with single-flight deduplication and a CPU LRU."""

    ENCODE_TIMEOUT_S = 300.0

    def __init__(
        self,
        model: Any,
        *,
        cache_namespace: str,
        cache_max_entries: int = _CACHE_MAX_ENTRIES,
        cache_max_bytes: int = _CACHE_MAX_BYTES,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 0,
        max_queue_size: int = 0,
    ) -> None:
        self._model = model
        reference = next(model.audio_encoder.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._hidden_size = _text_hidden_size(model)
        self._stream = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )
        self._cache = StageOutputCache(
            max_size=cache_max_entries,
            max_bytes=cache_max_bytes,
            cache_device="cpu",
        )
        self._namespace = cache_namespace
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._inflight: dict[str, concurrent.futures.Future[torch.Tensor]] = {}
        self._hits = 0
        self._early_hits = 0
        self._misses = 0
        self._merged = 0
        self._failed = 0
        self._batch_count = 0
        self._item_count = 0
        self._submitted = 0
        self._pending = 0
        self._queue_full_waits = 0
        self._queue_depth_max = 0
        self._queue_wait_count = 0
        self._queue_wait_total_s = 0.0
        self._queue_wait_max_s = 0.0
        self._encoder_time_s = 0.0
        super().__init__(
            worker_name="arkasr-audio-encode",
            max_queue_size=max_queue_size,
        )

    def close(self) -> None:
        """Stop the encoder worker after all queued requests finish."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=5)

    def _enqueue(
        self,
        item: Any,
        future: concurrent.futures.Future[torch.Tensor],
    ) -> None:
        queue_was_full = False
        entry = QueueEntry(
            item=item,
            future=future,
            enqueued_at=time.perf_counter(),
        )
        while True:
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("ARK-ASR pre-LM encoder service is closed")
                try:
                    self._queue.put_nowait(entry)
                    break
                except queue.Full:
                    queue_was_full = True
            with self._worker_state_lock:
                if self._worker_error is not None:
                    raise RuntimeError(
                        "pre-LM encoder worker has failed"
                    ) from self._worker_error
            time.sleep(0.01)
        queue_depth = self._queue.qsize()
        with self._lock:
            if queue_was_full:
                self._queue_full_waits += 1
            self._queue_depth_max = max(self._queue_depth_max, queue_depth)

    def submit_item(self, item: Any) -> concurrent.futures.Future[torch.Tensor]:
        """Return when the item has been queued for LM-ready encoding."""
        expected_tokens = _expected_audio_tokens(item)
        if expected_tokens is None:
            raise RuntimeError(
                "ARK-ASR pre-LM encode requires the item's num_audio_tokens"
            )
        key = self._cache_key(item)

        if key is None:
            return self._track_submission(self._submit(item))

        cached = self.lookup_cached_embedding(
            getattr(item, "audio_fingerprint", None),
            expected_tokens,
        )
        if cached is not None:
            return self._submit_cache_hit(item, cached, early=False)

        follower_of: concurrent.futures.Future[torch.Tensor] | None = None
        leader = False
        with self._lock:
            future = self._inflight.get(key)
            if future is None:
                cached = self._cache.get(key)
                if cached is None or not self._is_valid(cached, expected_tokens):
                    cached = None
                    future = concurrent.futures.Future()
                    self._inflight[key] = future
                    leader = True
                    self._misses += 1
            else:
                self._merged += 1
                follower_of = future
        if cached is not None:
            return self._submit_cache_hit(item, cached, early=False)
        if leader:
            future.add_done_callback(
                lambda done, cache_key=key: self._clear_inflight(cache_key, done)
            )
            try:
                self._submit(item, future)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                raise
        if follower_of is None:
            return self._track_submission(future)

        item.feature = None
        completion: concurrent.futures.Future[torch.Tensor] = (
            concurrent.futures.Future()
        )
        self._track_submission(completion)

        def attach_follower(done: concurrent.futures.Future[torch.Tensor]) -> None:
            try:
                embedding = done.result()
                if not self._is_valid(embedding, expected_tokens):
                    raise RuntimeError(
                        f"ARK-ASR pre-LM encode leader for {key} returned an "
                        "invalid embedding"
                    )
                self._attach_ready_embedding(item, embedding)
                completion.set_result(embedding)
            except Exception as exc:
                completion.set_exception(exc)

        follower_of.add_done_callback(attach_follower)
        return completion

    def submit_cached_item(
        self,
        item: Any,
        embedding: torch.Tensor,
    ) -> concurrent.futures.Future[torch.Tensor]:
        return self._submit_cache_hit(item, embedding, early=True)

    def _submit_cache_hit(
        self,
        item: Any,
        embedding: torch.Tensor,
        *,
        early: bool,
    ) -> concurrent.futures.Future[torch.Tensor]:
        future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
        self._track_submission(future)
        try:
            self._attach_ready_embedding(item, embedding)
        except Exception as exc:
            future.set_exception(exc)
            return future
        with self._lock:
            self._hits += 1
            if early:
                self._early_hits += 1
        future.set_result(embedding)
        return future

    def _attach_ready_embedding(self, item: Any, embedding: torch.Tensor) -> None:
        expected_tokens = _expected_audio_tokens(item)
        if expected_tokens is None or not self._is_valid(embedding, expected_tokens):
            raise RuntimeError("ARK-ASR embedding attachment failed validation")
        try:
            if self._device.type == "cuda":
                with torch.cuda.stream(torch.cuda.default_stream(self._device)):
                    self.attach_embedding(item, embedding)
            else:
                self.attach_embedding(item, embedding)
        except Exception as exc:
            failure = self._detach_failure(exc)
            logger.error(
                "ARK-ASR embedding attachment failed:\n%s",
                failure.formatted_traceback,
            )
            self._recover_after_failure(failure.exception)
            raise failure.exception from None

    def encode_item(self, item: Any) -> None:
        """Block until ``item.precomputed_embeddings`` holds the LM embedding.

        On success ``item.feature`` is cleared to release the CPU mel tensor.
        Raises on encode failure; the request must not be admitted without the
        complete embedding.
        """
        self.submit_item(item).result(timeout=self.ENCODE_TIMEOUT_S)

    def _track_submission(
        self, future: concurrent.futures.Future[torch.Tensor]
    ) -> concurrent.futures.Future[torch.Tensor]:
        with self._lock:
            self._submitted += 1
            self._pending += 1

        def finish(done: concurrent.futures.Future[torch.Tensor]) -> None:
            try:
                failed = done.exception() is not None
            except concurrent.futures.CancelledError:
                failed = True
            with self._lock:
                self._pending -= 1
                if failed:
                    self._failed += 1

        future.add_done_callback(finish)
        return future

    def _clear_inflight(
        self,
        key: str,
        future: concurrent.futures.Future[torch.Tensor],
    ) -> None:
        with self._lock:
            if self._inflight.get(key) is future:
                del self._inflight[key]

    def lookup_cached_embedding(
        self,
        audio_fingerprint: str | None,
        expected_tokens: int,
    ) -> torch.Tensor | None:
        """Return a validated cached embedding without starting an encode."""
        key = self._cache_key_from_fingerprint(audio_fingerprint)
        cached = self._cache.get(key)
        if cached is None:
            return None
        if self._is_valid(cached, expected_tokens):
            return cached
        logger.warning(
            "ARK-ASR pre-LM cache entry %s failed validation "
            "(shape=%s, dtype=%s); discarding it if unchanged before re-encoding",
            key,
            getattr(cached, "shape", None),
            getattr(cached, "dtype", None),
        )
        self._cache.remove_if_same(key, cached)
        return None

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            cache_lookups = self._hits + self._misses
            return {
                "hits": self._hits,
                "early_hits": self._early_hits,
                "misses": self._misses,
                "merged": self._merged,
                "failed": self._failed,
                "submitted": self._submitted,
                "pending": self._pending,
                "queue_full_waits": self._queue_full_waits,
                "queue_depth_max": self._queue_depth_max,
                "cache_hit_rate": (
                    self._hits / cache_lookups if cache_lookups else 0.0
                ),
                "batches": self._batch_count,
                "items": self._item_count,
                "queue_depth": self._queue.qsize(),
                "queue_wait_avg_s": (
                    self._queue_wait_total_s / self._queue_wait_count
                    if self._queue_wait_count
                    else 0.0
                ),
                "queue_wait_max_s": self._queue_wait_max_s,
                "encoder_time_s": self._encoder_time_s,
                "cache_entries": len(self._cache),
                "cache_bytes": self._cache.current_bytes,
                "cache_evictions": self._cache.eviction_count,
            }

    def _cache_key(self, item: Any) -> str | None:
        return self._cache_key_from_fingerprint(
            getattr(item, "audio_fingerprint", None)
        )

    def _cache_key_from_fingerprint(self, audio_fingerprint: str | None) -> str | None:
        if audio_fingerprint is None:
            return None
        return f"{self._namespace}:{audio_fingerprint}"

    def _is_valid(self, embedding: Any, expected_tokens: int) -> bool:
        return (
            isinstance(embedding, torch.Tensor)
            and embedding.dim() == 2
            and embedding.shape[0] == expected_tokens
            and embedding.shape[1] == self._hidden_size
            and embedding.dtype == self._dtype
        )

    def attach_embedding(self, item: Any, embedding: torch.Tensor) -> None:
        embedding = embedding.to(self._device, non_blocking=True)
        if self._stream is not None and embedding.is_cuda:
            # the batch path allocates on the private stream while the LM
            # consumes on the default stream; register the consumer so the
            # allocator cannot recycle the block for a later batch while LM
            # reads are still queued.
            embedding.record_stream(torch.cuda.default_stream(self._device))
        item.precomputed_embeddings = embedding
        item.feature = None
        item.format = MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    def _drain_batch(
        self,
    ) -> tuple[list[QueueEntry[Any]], bool]:
        # the default window is 0 (greedy drain): items that queued while the
        # previous batch encoded are taken instantly, so groups still form
        # under load, and an idle-arrival request never pays a batching wait --
        # at concurrency 1 a window is pure latency.
        first = self._queue.get()
        if first is _SHUTDOWN:
            return [], True
        batch = [cast(QueueEntry[Any], first)]
        deadline = time.monotonic() + self._max_batch_wait_s
        shutdown = False
        while len(batch) < self._max_batch_size:
            try:
                remaining = deadline - time.monotonic()
                queued = (
                    self._queue.get(timeout=remaining)
                    if remaining > 0
                    else self._queue.get_nowait()
                )
            except queue.Empty:
                break
            if queued is _SHUTDOWN:
                shutdown = True
                break
            batch.append(cast(QueueEntry[Any], queued))
        return batch, shutdown

    def _next_batch(self) -> tuple[list[QueueEntry[Any]], bool]:
        return self._drain_batch()

    @contextlib.contextmanager
    def _batch_context(self) -> Iterator[None]:
        with torch.inference_mode():
            if self._stream is None:
                yield
            else:
                with torch.cuda.stream(self._stream):
                    yield

    def encode_batch(self, items: list[Any]) -> torch.Tensor:
        return self._model.get_audio_feature(items)

    def split_embeddings(
        self,
        items: list[Any],
        embedding: torch.Tensor,
    ) -> list[torch.Tensor]:
        token_counts = []
        for item in items:
            expected = _expected_audio_tokens(item)
            if expected is None:
                raise RuntimeError(
                    "ARK-ASR pre-LM encode item is missing its audio token count"
                )
            token_counts.append(expected)
        # get_audio_feature concatenates each item's [tokens_i, hidden] block
        # along the token axis, so the result is already flat.
        if (
            embedding.dim() != 2
            or embedding.shape[0] != sum(token_counts)
            or embedding.shape[1] != self._hidden_size
            or embedding.dtype != self._dtype
        ):
            raise RuntimeError(
                f"ARK-ASR encoder output {tuple(embedding.shape)} "
                f"({embedding.dtype}) != expected rows "
                f"{sum(token_counts)}x{self._hidden_size} ({self._dtype})"
            )
        parts = torch.split(embedding, token_counts, dim=0)
        return [part.clone() for part in parts]

    def synchronize_batch(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()

    def cache_embedding(
        self,
        item: Any,
        embedding: torch.Tensor,
        host_copy: torch.Tensor | None = None,
    ) -> None:
        del host_copy
        key = self._cache_key(item)
        if key is not None:
            self._cache.put(key, embedding)

    def _retry_batch(self, batch: list[QueueEntry[Any]], _exc: Exception) -> bool:
        return len(batch) > 1

    def _handle_batch_failure(
        self,
        batch: list[QueueEntry[Any]],
        exc: Exception,
    ) -> Exception:
        failure = self._detach_failure(exc)
        if len(batch) == 1:
            logger.error(
                "ARK-ASR audio encode failed:\n%s",
                failure.formatted_traceback,
            )
        else:
            logger.error(
                "ARK-ASR batched audio encode failed for %d items; "
                "retrying per item:\n%s",
                len(batch),
                failure.formatted_traceback,
            )
        self._recover_after_failure(failure.exception)
        return failure.exception

    def _handle_item_failure(
        self,
        _entry: QueueEntry[Any],
        exc: Exception,
    ) -> Exception:
        failure = self._detach_failure(exc)
        logger.error(
            "ARK-ASR per-item audio encode retry failed:\n%s",
            failure.formatted_traceback,
        )
        self._recover_after_failure(failure.exception)
        return failure.exception

    @staticmethod
    def _detach_failure(exc: Exception) -> _DetachedFailure:
        # keep the formatted traceback for logs but drop frame references from
        # the propagated exception; a future holding a traceback would pin
        # encoder tensors after an OOM.
        formatted_traceback = "".join(traceback.format_exception(exc)).rstrip()
        message = str(exc)
        traceback.clear_frames(exc.__traceback__)
        exc.__traceback__ = None
        exc.__cause__ = None
        exc.__context__ = None
        if isinstance(exc, torch.OutOfMemoryError):
            detached: Exception = torch.OutOfMemoryError(message)
        elif isinstance(exc, ValueError):
            detached = ValueError(message)
        else:
            detached = RuntimeError(f"{type(exc).__name__}: {message}")
        return _DetachedFailure(
            exception=detached,
            formatted_traceback=formatted_traceback,
        )

    def _recover_after_failure(self, exc: Exception) -> None:
        if not isinstance(exc, torch.OutOfMemoryError):
            return
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except Exception:
                logger.warning(
                    "ARK-ASR encoder stream cleanup failed after OOM",
                    exc_info=True,
                )
        try:
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
        except Exception:
            logger.warning("ARK-ASR CUDA cache cleanup failed after OOM", exc_info=True)

    def _on_batch_start(self, batch: list[QueueEntry[Any]]) -> None:
        dequeue_time = time.perf_counter()
        queue_waits = [
            dequeue_time - entry.enqueued_at
            for entry in batch
            if entry.enqueued_at is not None
        ]
        with self._lock:
            self._queue_wait_count += len(queue_waits)
            self._queue_wait_total_s += sum(queue_waits)
            self._queue_wait_max_s = max(
                self._queue_wait_max_s,
                max(queue_waits, default=0.0),
            )

    def _on_batch_finished(
        self,
        batch: list[QueueEntry[Any]],
        batch_exc: Exception | None,
        retry_recovered: int | None,
        elapsed_s: float,
    ) -> None:
        with self._lock:
            self._encoder_time_s += elapsed_s
            if batch_exc is not None:
                if retry_recovered is not None:
                    # retried items are single-item batches.
                    self._batch_count += retry_recovered
                    self._item_count += retry_recovered
                return
            self._batch_count += 1
            self._item_count += len(batch)
            batch_count = self._batch_count
            item_count = self._item_count
        if batch_count % 50 == 1:
            logger.info(
                f"ARK-ASR pre-LM encoder stage: {batch_count} batches, "
                f"{item_count} items (avg "
                f"{item_count / batch_count:.2f} items/batch, "
                f"last batch: {len(batch)}), cache: {self.stats()}"
            )


__all__ = [
    "ArkasrPreLMEncoderService",
    "build_cache_namespace",
]
