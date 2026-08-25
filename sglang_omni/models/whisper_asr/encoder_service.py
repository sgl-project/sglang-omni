# SPDX-License-Identifier: Apache-2.0
"""Precompute and cache Whisper encoder hidden states before LM admission."""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import logging
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any, cast

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputFormat

from sglang_omni.scheduling.pre_lm_encoder import PreLMEncoderService, QueueEntry
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

# Note(Jeffro): Whisper encoder states are fixed-size ([encoder_token_count, d_model] in the
# encoder dtype), so the cache is sized in entries and the byte budget is
# derived from that: 1024 entries is ~3.9 GB of host memory for large-v3
# (3.84 MB per entry) and ~1.5 GB for base.
_CACHE_MAX_ENTRIES = 1024
_SHUTDOWN = object()


def build_cache_namespace(
    model: Any,
    *,
    model_path: str,
    feature_extractor: Any,
) -> str:
    """Digest identifying this process's encoder pipeline for cache keying."""
    config = model.config
    try:
        model_config: Any = config.to_dict()
    except AttributeError:
        model_config = repr(config)
    reference = next(model.model.encoder.parameters())
    payload = {
        "model_path": model_path,
        "model_config": model_config,
        "frontend": {
            "feature_size": feature_extractor.feature_size,
            "sampling_rate": feature_extractor.sampling_rate,
            "hop_length": feature_extractor.hop_length,
            "chunk_length": feature_extractor.chunk_length,
            "n_fft": feature_extractor.n_fft,
            "nb_max_frames": feature_extractor.nb_max_frames,
            "padding_value": feature_extractor.padding_value,
        },
        "dtype": str(reference.dtype),
        "device_type": reference.device.type,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _expected_audio_tokens(item: Any) -> int | None:
    num_tokens = item.num_audio_tokens
    return int(num_tokens) if num_tokens is not None else None


class WhisperPreLMEncoderService(PreLMEncoderService[Any, torch.Tensor, torch.Tensor]):
    """Encode before admission with single-flight deduplication and a CPU LRU."""

    ENCODE_TIMEOUT_S = 300.0

    def __init__(
        self,
        model: Any,
        *,
        cache_namespace: str,
        encoder_token_count: int,
        cache_max_entries: int = _CACHE_MAX_ENTRIES,
        cache_max_bytes: int | None = None,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 0,
        pin_host_memory: bool = True,
    ) -> None:
        if encoder_token_count < 1:
            raise ValueError(
                f"encoder_token_count must be >= 1, got {encoder_token_count}"
            )
        if cache_max_entries < 0:
            raise ValueError(f"cache_max_entries must be >= 0, got {cache_max_entries}")
        if cache_max_bytes is not None and cache_max_bytes < 0:
            raise ValueError(f"cache_max_bytes must be >= 0, got {cache_max_bytes}")
        self._model = model
        reference = next(model.model.encoder.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._hidden_size = int(model.config.d_model)
        self._stream = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )

        self._encoder_token_count = int(encoder_token_count)
        self._entry_bytes = (
            self._encoder_token_count * self._hidden_size * self._dtype.itemsize
        )
        derived_bytes = int(cache_max_entries) * self._entry_bytes
        self._cache_max_bytes = (
            derived_bytes
            if cache_max_bytes is None
            else min(derived_bytes, int(cache_max_bytes))
        )
        self._cache_capacity_entries = min(
            int(cache_max_entries), self._cache_max_bytes // self._entry_bytes
        )
        self._pin_host_memory = pin_host_memory and self._device.type == "cuda"
        self._cache = StageOutputCache(
            max_size=cache_max_entries,
            max_bytes=self._cache_max_bytes,
            cache_device="cpu",
            pin_memory=self._pin_host_memory,
        )
        self._pin_failures = 0
        self._prewarm_s = 0.0
        if self._pin_host_memory:
            self._prewarm_pinned_pool(self._cache_capacity_entries)
        self._namespace = cache_namespace
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._inflight: dict[str, concurrent.futures.Future[torch.Tensor]] = {}
        self._hits = 0
        self._misses = 0
        self._merged = 0
        self._failed = 0
        self._batch_count = 0
        self._item_count = 0
        self._queue_wait_count = 0
        self._queue_wait_total_s = 0.0
        self._queue_wait_max_s = 0.0
        self._encoder_time_s = 0.0
        super().__init__(worker_name="whisper-asr-audio-encode")

    @property
    def entry_bytes(self) -> int:
        """Bytes of one cached encoder state."""
        return self._entry_bytes

    @property
    def cache_max_bytes(self) -> int:
        """Effective byte budget after applying the optional cap."""
        return self._cache_max_bytes

    @property
    def cache_capacity_entries(self) -> int:
        """How many encoder states the cache can hold at once."""
        return self._cache_capacity_entries

    @property
    def pin_host_memory(self) -> bool:
        """Whether cached states are held in page-locked host memory."""
        return self._pin_host_memory

    def _new_pinned_host(self, tokens: int) -> torch.Tensor:
        return torch.empty(
            (tokens, self._hidden_size),
            dtype=self._dtype,
            device="cpu",
            pin_memory=True,
        )

    def _prewarm_pinned_pool(self, entries: int) -> None:
        """Pay the one-off cudaHostAlloc cost for the whole cache at start-up.

        Freed pinned blocks stay in PyTorch's caching host allocator, so
        allocating the full capacity once and dropping it means the first
        cache fill reuses warm blocks instead of paying ~1 ms per entry on
        the encoder worker.
        """
        if entries <= 0:
            return
        started = time.perf_counter()
        blocks: list[torch.Tensor] = []
        try:
            for _ in range(int(entries)):
                blocks.append(self._new_pinned_host(self._encoder_token_count))
        except RuntimeError as exc:
            logger.warning(
                "Whisper pre-LM cache: pinned host pool prewarm stopped after "
                "%d/%d entries (%s); remaining entries will pin lazily or fall "
                "back to pageable memory",
                len(blocks),
                entries,
                exc,
            )
        finally:
            warmed = len(blocks)
            blocks.clear()
            self._prewarm_s = time.perf_counter() - started
        logger.info(
            "Whisper pre-LM cache: prewarmed %d pinned host entries "
            "(%.1f MB) in %.2fs",
            warmed,
            warmed * self._entry_bytes / 1e6,
            self._prewarm_s,
        )

    def _disable_pinning(self, exc: Exception) -> None:
        if not self._pin_host_memory:
            return
        self._pin_host_memory = False
        self._cache.pin_memory = False
        logger.warning(
            "Whisper pre-LM cache: pinned host allocation failed (%s); "
            "switching this cache to pageable host memory",
            exc,
        )

    def close(self) -> None:
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
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Whisper ASR pre-LM encoder service is closed")
            self._queue.put(
                QueueEntry(
                    item=item,
                    future=future,
                    enqueued_at=time.perf_counter(),
                )
            )

    def encode_item(self, item: Any) -> None:
        """Block until item.precomputed_embeddings holds encoder states."""
        expected_tokens = _expected_audio_tokens(item)
        if expected_tokens is None:
            raise RuntimeError(
                "Whisper pre-LM encode requires the item's num_audio_tokens"
            )
        key = self._cache_key(item)

        if key is None:
            future = self._submit(item)
            future.result(timeout=self.ENCODE_TIMEOUT_S)
            return

        cached = self.lookup_cached_embedding(item.audio_fingerprint, expected_tokens)
        if cached is not None:
            self.attach_embedding(item, cached)
            return

        leader = False
        with self._lock:
            future = self._inflight.get(key)
            if future is None:
                cached = self._cache.get(key)
                if cached is not None and self._is_valid(cached, expected_tokens):
                    self._hits += 1
                else:
                    cached = None
                    future = concurrent.futures.Future()
                    self._inflight[key] = future
                    leader = True
                    self._misses += 1
                    try:
                        self._submit(item, future)
                    except Exception:
                        del self._inflight[key]
                        raise
            else:
                self._merged += 1
        if cached is not None:
            self.attach_embedding(item, cached)
            return
        try:
            embedding = future.result(timeout=self.ENCODE_TIMEOUT_S)
        except Exception:
            with self._lock:
                self._failed += 1
            raise
        finally:
            if leader:
                with self._lock:
                    if self._inflight.get(key) is future:
                        del self._inflight[key]
        if leader:
            return
        if not self._is_valid(embedding, expected_tokens):
            with self._lock:
                self._failed += 1
            raise RuntimeError(
                f"Whisper pre-LM encode leader for {key} returned an invalid "
                f"embedding"
            )
        self.attach_embedding(item, embedding)

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
            with self._lock:
                self._hits += 1
            return cached
        logger.warning(
            f"Whisper pre-LM cache entry {key} failed validation "
            f"(shape={tuple(cached.shape)}, dtype={cached.dtype}); "
            f"discarding it if unchanged before re-encoding"
        )
        self._cache.remove_if_same(key, cached)
        return None

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            cache_lookups = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "failed": self._failed,
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
                "cache_capacity_entries": self._cache_capacity_entries,
                "cache_bytes": self._cache.current_bytes,
                "cache_evictions": self._cache.eviction_count,
                "pin_host_memory": self._pin_host_memory,
                "pin_failures": self._pin_failures,
                "pin_prewarm_s": self._prewarm_s,
            }

    def _cache_key(self, item: Any) -> str | None:
        return self._cache_key_from_fingerprint(item.audio_fingerprint)

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
            embedding.record_stream(torch.cuda.default_stream(self._device))
        item.precomputed_embeddings = embedding
        item.feature = None
        item.format = MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    def _drain_batch(self) -> tuple[list[QueueEntry[Any]], bool]:
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
        return self._model.encode_audio_features(items)

    def split_embeddings(
        self,
        items: list[Any],
        encoded: torch.Tensor,
    ) -> list[torch.Tensor]:
        if encoded.dim() != 3:
            raise RuntimeError(
                f"Whisper encoder output rank {encoded.dim()} != 3 "
                f"(expected [B, T, H])"
            )
        if encoded.shape[0] != len(items):
            raise RuntimeError(
                f"Whisper encoder batch {encoded.shape[0]} != {len(items)} items"
            )
        if encoded.shape[2] != self._hidden_size or encoded.dtype != self._dtype:
            raise RuntimeError(
                f"Whisper encoder output {tuple(encoded.shape)} ({encoded.dtype}) "
                f"!= expected hidden {self._hidden_size} ({self._dtype})"
            )
        parts: list[torch.Tensor] = []
        for index, item in enumerate(items):
            expected = _expected_audio_tokens(item)
            if expected is None:
                raise RuntimeError(
                    "Whisper pre-LM encode item is missing its audio token count"
                )
            if encoded.shape[1] < expected:
                raise RuntimeError(
                    f"Whisper encoder T={encoded.shape[1]} < expected {expected}"
                )
            parts.append(encoded[index, :expected].clone())
        return parts

    def stage_host_copy(
        self, item: Any, embedding: torch.Tensor
    ) -> torch.Tensor | None:
        """Copy embedding GPU->CPU into a pinned buffer without waiting.

        We are still on the encoder stream here, so the copy runs on the GPU
        right after the encoder kernels and the CPU thread returns at once.
        synchronize_batch (called next by the base class) waits for it, so by
        the time cache_embedding gets this tensor the data is complete.
        Returns None when pinning is off or the item has no cache key.
        """
        if not self._pin_host_memory or self._cache_key(item) is None:
            return None
        try:
            host = self._new_pinned_host(int(embedding.shape[0]))
        except RuntimeError as exc:
            self._pin_failures += 1
            self._disable_pinning(exc)
            return None
        host.copy_(embedding, non_blocking=True)
        return host

    def synchronize_batch(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()

    def cache_embedding(
        self,
        item: Any,
        embedding: torch.Tensor,
        host_copy: torch.Tensor | None = None,
    ) -> None:
        key = self._cache_key(item)
        if key is None:
            return
        # note (Jeffro): host_copy is complete here (synchronize_batch ran in
        # between) and already pinned, so the cache stores it without another copy.
        self._cache.put(key, host_copy if host_copy is not None else embedding)

    def _retry_batch(self, batch: list[QueueEntry[Any]], _exc: Exception) -> bool:
        if len(batch) == 1:
            return False
        logger.exception(
            "Whisper batched audio encode failed for %d items; retrying per item",
            len(batch),
        )
        return True

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
                    self._batch_count += retry_recovered
                    self._item_count += retry_recovered
                return
            self._batch_count += 1
            self._item_count += len(batch)
            batch_count = self._batch_count
            item_count = self._item_count
        if batch_count % 50 == 1:
            logger.info(
                f"Whisper pre-LM encoder stage: {batch_count} batches, "
                f"{item_count} items (avg "
                f"{item_count / batch_count:.2f} items/batch, "
                f"last batch: {len(batch)}), cache: {self.stats()}"
            )


__all__ = [
    "WhisperPreLMEncoderService",
    "build_cache_namespace",
]
