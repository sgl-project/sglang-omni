# SPDX-License-Identifier: Apache-2.0
"""Precompute and cache complete LM-ready Fun-ASR audio embeddings."""

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

_FRONTEND_CONFIG_FIELDS = (
    "feature_size",
    "sampling_rate",
    "frame_length",
    "frame_shift",
    "num_frames_lfr",
    "stride_lfr",
    "window",
)


@dataclass(frozen=True)
class DetachedFailure:
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
    payload = {
        "model_path": model_path,
        "model_config": model_config,
        "frontend": {
            field: getattr(feature_extractor, field, None)
            for field in _FRONTEND_CONFIG_FIELDS
        },
        "dtype": str(next(model.audio_tower.parameters()).dtype),
        "mm_attention_backend": mm_attention_backend or "default",
        "device_type": next(model.audio_tower.parameters()).device.type,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def expected_audio_tokens(item: Any) -> int | None:
    """Audio placeholder token count for an item (rows the LM expects)."""
    num_tokens = getattr(item, "num_audio_tokens", None)
    return int(num_tokens) if num_tokens is not None else None


class FunASRPreLMEncoderService(PreLMEncoderService[Any, torch.Tensor, torch.Tensor]):
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
        max_batch_wait_ms: int = 4,
    ) -> None:
        self.model = model
        reference = next(model.audio_tower.parameters())
        self.device = reference.device
        self.dtype = reference.dtype
        self.hidden_size = int(model.config.text_config.hidden_size)
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.cache = StageOutputCache(
            max_size=cache_max_entries,
            max_bytes=cache_max_bytes,
            cache_device="cpu",
        )
        self.namespace = cache_namespace
        self.max_batch_size = max(int(max_batch_size), 1)
        self.max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self.lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.closed = False
        self.inflight: dict[str, concurrent.futures.Future[torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0
        self.merged = 0
        self.failed = 0
        self.batch_count = 0
        self.item_count = 0
        self.queue_wait_count = 0
        self.queue_wait_total_s = 0.0
        self.queue_wait_max_s = 0.0
        self.encoder_time_s = 0.0
        super().__init__(worker_name="fun-asr-audio-encode")

    def close(self) -> None:
        """Stop the encoder worker after all queued requests finish."""
        with self.lifecycle_lock:
            if self.closed:
                return
            self.closed = True
            self.queue.put(_SHUTDOWN)
        self.thread.join(timeout=5)

    def enqueue(
        self,
        item: Any,
        future: concurrent.futures.Future[torch.Tensor],
    ) -> None:
        with self.lifecycle_lock:
            if self.closed:
                raise RuntimeError("Fun-ASR pre-LM encoder service is closed")
            self.queue.put(
                QueueEntry(
                    item=item,
                    future=future,
                    enqueued_at=time.perf_counter(),
                )
            )

    def encode_item(self, item: Any) -> None:
        """Block until ``item.precomputed_embeddings`` holds the LM embedding.

        On success ``item.feature`` is cleared to release the CPU fbank/LFR
        tensor. Raises on encode failure; the request must not be admitted
        without the complete embedding.
        """
        expected_tokens = expected_audio_tokens(item)
        if expected_tokens is None:
            raise RuntimeError(
                "Fun-ASR pre-LM encode requires the item's num_audio_tokens"
            )
        key = self.cache_key(item)

        if key is None:
            future = self.submit(item)
            future.result(timeout=self.ENCODE_TIMEOUT_S)
            return

        cached = self.cache.get(key)
        if cached is not None:
            if self.is_valid(cached, expected_tokens):
                with self.lock:
                    self.hits += 1
                self.attach_embedding(item, cached)
                return
            logger.warning(
                f"Fun-ASR pre-LM cache entry {key} failed validation "
                f"(shape={tuple(cached.shape)}, dtype={cached.dtype}); "
                f"discarding it if unchanged before re-encoding"
            )
            self.cache.remove_if_same(key, cached)
            cached = None

        leader = False
        with self.lock:
            future = self.inflight.get(key)
            if future is None:
                # Note (Akazaakane): Re-check under the single-flight lock so a
                # stale miss cannot start work after the prior leader cached.
                cached = self.cache.get(key)
                if cached is not None and self.is_valid(cached, expected_tokens):
                    self.hits += 1
                else:
                    cached = None
                    future = concurrent.futures.Future()
                    self.inflight[key] = future
                    leader = True
                    self.misses += 1
                    try:
                        self.submit(item, future)
                    except Exception:
                        del self.inflight[key]
                        raise
            else:
                self.merged += 1
        if cached is not None:
            self.attach_embedding(item, cached)
            return
        try:
            embedding = future.result(timeout=self.ENCODE_TIMEOUT_S)
        except Exception:
            with self.lock:
                self.failed += 1
            raise
        finally:
            if leader:
                with self.lock:
                    if self.inflight.get(key) is future:
                        del self.inflight[key]
        if leader:
            return
        if not self.is_valid(embedding, expected_tokens):
            with self.lock:
                self.failed += 1
            raise RuntimeError(
                f"Fun-ASR pre-LM encode leader for {key} returned an invalid "
                f"embedding"
            )
        self.attach_embedding(item, embedding)

    def stats(self) -> dict[str, int | float]:
        with self.lock:
            cache_lookups = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "merged": self.merged,
                "failed": self.failed,
                "cache_hit_rate": (self.hits / cache_lookups if cache_lookups else 0.0),
                "batches": self.batch_count,
                "items": self.item_count,
                "queue_depth": self.queue.qsize(),
                "queue_wait_avg_s": (
                    self.queue_wait_total_s / self.queue_wait_count
                    if self.queue_wait_count
                    else 0.0
                ),
                "queue_wait_max_s": self.queue_wait_max_s,
                "encoder_time_s": self.encoder_time_s,
                "cache_entries": len(self.cache),
                "cache_bytes": self.cache.current_bytes,
                "cache_evictions": self.cache.eviction_count,
            }

    def cache_key(self, item: Any) -> str | None:
        item_hash = getattr(item, "audio_fingerprint", None)
        if item_hash is None:
            return None
        return f"{self.namespace}:{item_hash}"

    def is_valid(self, embedding: Any, expected_tokens: int) -> bool:
        return (
            isinstance(embedding, torch.Tensor)
            and embedding.dim() == 2
            and embedding.shape[0] == expected_tokens
            and embedding.shape[1] == self.hidden_size
            and embedding.dtype == self.dtype
        )

    def attach_embedding(self, item: Any, embedding: torch.Tensor) -> None:
        item.precomputed_embeddings = embedding.to(self.device, non_blocking=True)
        item.feature = None
        item.format = MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    def drain_batch(
        self,
    ) -> tuple[list[QueueEntry[Any]], bool]:
        first = self.queue.get()
        if first is _SHUTDOWN:
            return [], True
        batch = [cast(QueueEntry[Any], first)]
        deadline = time.monotonic() + self.max_batch_wait_s
        shutdown = False
        while len(batch) < self.max_batch_size:
            try:
                remaining = deadline - time.monotonic()
                queued = (
                    self.queue.get(timeout=remaining)
                    if remaining > 0
                    else self.queue.get_nowait()
                )
            except queue.Empty:
                break
            if queued is _SHUTDOWN:
                shutdown = True
                break
            batch.append(cast(QueueEntry[Any], queued))
        return batch, shutdown

    def next_batch(self) -> tuple[list[QueueEntry[Any]], bool]:
        return self.drain_batch()

    @contextlib.contextmanager
    def batch_context(self) -> Iterator[None]:
        with torch.inference_mode():
            if self.stream is None:
                yield
            else:
                with torch.cuda.stream(self.stream):
                    yield

    def encode_batch(self, items: list[Any]) -> torch.Tensor:
        return self.model.get_audio_feature(items)

    def split_embeddings(
        self,
        items: list[Any],
        embedding: torch.Tensor,
    ) -> list[torch.Tensor]:
        token_counts = []
        for item in items:
            expected = expected_audio_tokens(item)
            if expected is None:
                raise RuntimeError(
                    "Fun-ASR pre-LM encode item is missing its audio token count"
                )
            token_counts.append(expected)
        if (
            embedding.dim() != 2
            or embedding.shape[0] != sum(token_counts)
            or embedding.shape[1] != self.hidden_size
            or embedding.dtype != self.dtype
        ):
            raise RuntimeError(
                f"Fun-ASR encoder output {tuple(embedding.shape)} "
                f"({embedding.dtype}) != expected rows "
                f"{sum(token_counts)}x{self.hidden_size} ({self.dtype})"
            )
        parts = torch.split(embedding, token_counts, dim=0)
        return [part.clone() for part in parts]

    def attach_before_synchronize(self) -> bool:
        return False

    def synchronize_batch(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()

    def cache_embedding(
        self,
        item: Any,
        embedding: torch.Tensor,
        host_copy: torch.Tensor | None = None,
    ) -> None:
        del host_copy
        key = self.cache_key(item)
        if key is not None:
            self.cache.put(key, embedding)

    @staticmethod
    def detach_failure(exc: Exception) -> DetachedFailure:
        formatted_traceback = "".join(traceback.format_exception(exc)).rstrip()
        message = str(exc)
        traceback.clear_frames(exc.__traceback__)
        exc.__traceback__ = None
        exc.__cause__ = None
        exc.__context__ = None
        if isinstance(exc, torch.OutOfMemoryError):
            detached = torch.OutOfMemoryError(message)
        elif isinstance(exc, ValueError):
            detached = ValueError(message)
        else:
            detached = RuntimeError(f"{type(exc).__name__}: {message}")
        return DetachedFailure(
            exception=detached,
            formatted_traceback=formatted_traceback,
        )

    def recover_after_failure(self, exc: Exception) -> None:
        if (
            not isinstance(exc, torch.OutOfMemoryError)
            or self.stream is None
            or self.device.type != "cuda"
        ):
            return
        try:
            self.stream.synchronize()
        except Exception as cleanup_exc:
            failure = self.detach_failure(cleanup_exc)
            logger.warning(
                "Fun-ASR encoder stream cleanup failed after OOM:\n%s",
                failure.formatted_traceback,
            )
        try:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        except Exception as cleanup_exc:
            failure = self.detach_failure(cleanup_exc)
            logger.warning(
                "Fun-ASR CUDA cache cleanup failed after OOM:\n%s",
                failure.formatted_traceback,
            )

    def handle_batch_failure(
        self,
        batch: list[QueueEntry[Any]],
        exc: Exception,
    ) -> Exception:
        failure = self.detach_failure(exc)
        if len(batch) == 1:
            logger.error(
                "Fun-ASR audio encode failed:\n%s",
                failure.formatted_traceback,
            )
        else:
            logger.error(
                "Fun-ASR batched audio encode failed for %d items; "
                "retrying per item:\n%s",
                len(batch),
                failure.formatted_traceback,
            )
        self.recover_after_failure(failure.exception)
        return failure.exception

    def handle_item_failure(
        self,
        _entry: QueueEntry[Any],
        exc: Exception,
    ) -> Exception:
        failure = self.detach_failure(exc)
        logger.error(
            "Fun-ASR per-item audio encode retry failed:\n%s",
            failure.formatted_traceback,
        )
        self.recover_after_failure(failure.exception)
        return failure.exception

    def retry_batch(self, batch: list[QueueEntry[Any]], _exc: Exception) -> bool:
        return len(batch) > 1

    def on_batch_start(self, batch: list[QueueEntry[Any]]) -> None:
        dequeue_time = time.perf_counter()
        queue_waits = [
            dequeue_time - entry.enqueued_at
            for entry in batch
            if entry.enqueued_at is not None
        ]
        with self.lock:
            self.queue_wait_count += len(queue_waits)
            self.queue_wait_total_s += sum(queue_waits)
            self.queue_wait_max_s = max(
                self.queue_wait_max_s,
                max(queue_waits, default=0.0),
            )

    def on_batch_finished(
        self,
        batch: list[QueueEntry[Any]],
        batch_exc: Exception | None,
        retry_recovered: int | None,
        elapsed_s: float,
    ) -> None:
        with self.lock:
            self.encoder_time_s += elapsed_s
            if batch_exc is not None:
                if retry_recovered is not None:
                    # Note (Akazaakane): Retried items are single-item batches.
                    self.batch_count += retry_recovered
                    self.item_count += retry_recovered
                return
            self.batch_count += 1
            self.item_count += len(batch)
            batch_count = self.batch_count
            item_count = self.item_count
        if batch_count % 50 == 1:
            logger.info(
                f"Fun-ASR pre-LM encoder stage: {batch_count} batches, "
                f"{item_count} items (avg "
                f"{item_count / batch_count:.2f} items/batch, "
                f"last batch: {len(batch)}), cache: {self.stats()}"
            )


__all__ = [
    "FunASRPreLMEncoderService",
    "build_cache_namespace",
]
