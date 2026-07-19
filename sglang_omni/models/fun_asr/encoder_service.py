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
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputFormat

from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = 4096
_CACHE_MAX_BYTES = 2 * 1024**3

_FRONTEND_CONFIG_FIELDS = (
    "feature_size",
    "sampling_rate",
    "frame_length",
    "frame_shift",
    "lfr_m",
    "lfr_n",
    "window",
)


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


def _expected_audio_tokens(item: Any) -> int | None:
    """Audio placeholder token count for an item (rows the LM expects)."""
    num_tokens = getattr(item, "num_audio_tokens", None)
    if num_tokens is not None:
        return int(num_tokens)
    mask = getattr(item, "feature_attention_mask", None)
    if mask is not None:
        return max(int(fun_asr_low_frame_rate_length(int(mask.sum()))), 1)
    feature = getattr(item, "feature", None)
    if isinstance(feature, torch.Tensor) and feature.dim() >= 1:
        return max(int(fun_asr_low_frame_rate_length(int(feature.shape[-1]))), 1)
    return None


class FunAsrPreLMEncoderService:
    """Encode before admission with single-flight deduplication and a CPU LRU."""

    ENCODE_TIMEOUT_S = 300.0

    def __init__(
        self,
        model: Any,
        *,
        cache_namespace: str,
        cache_max_entries: int = _CACHE_MAX_ENTRIES,
        cache_max_bytes: int = _CACHE_MAX_BYTES,
    ) -> None:
        self._model = model
        reference = next(model.audio_tower.parameters())
        self._device = reference.device
        self._dtype = reference.dtype
        self._hidden_size = int(model.config.text_config.hidden_size)
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
        self._queue: queue.Queue[
            tuple[Any, concurrent.futures.Future[torch.Tensor], float]
        ] = queue.Queue()
        self._lock = threading.Lock()
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
        self._thread = threading.Thread(
            target=self._worker, name="fun-asr-audio-encode", daemon=True
        )
        self._thread.start()

    def encode_item(self, item: Any) -> None:
        """Block until ``item.precomputed_embeddings`` holds the LM embedding.

        On success ``item.feature`` is cleared to release the CPU fbank/LFR
        tensor. Raises on encode failure; the request must not be admitted
        without the complete embedding.
        """
        expected_tokens = _expected_audio_tokens(item)
        if expected_tokens is None:
            raise RuntimeError(
                "Fun-ASR pre-LM encode requires the item's audio token count "
                "(num_audio_tokens or feature_attention_mask)"
            )
        key = self._cache_key(item)

        if key is None:
            future: concurrent.futures.Future[torch.Tensor] = (
                concurrent.futures.Future()
            )
            self._queue.put((item, future, time.perf_counter()))
            future.result(timeout=self.ENCODE_TIMEOUT_S)
            return

        cached = self._cache.get(key)
        if cached is not None:
            if self._is_valid(cached, expected_tokens):
                with self._lock:
                    self._hits += 1
                self._attach(item, cached)
                return
            logger.warning(
                f"Fun-ASR pre-LM cache entry {key} failed validation "
                f"(shape={tuple(cached.shape)}, dtype={cached.dtype}); "
                f"evicting and re-encoding"
            )
            self._cache.remove_if(lambda candidate: candidate == key)

        leader = False
        with self._lock:
            future = self._inflight.get(key)
            if future is None:
                future = concurrent.futures.Future()
                self._inflight[key] = future
                leader = True
                self._misses += 1
                self._queue.put((item, future, time.perf_counter()))
            else:
                self._merged += 1
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
                f"Fun-ASR pre-LM encode leader for {key} returned an invalid "
                f"embedding"
            )
        self._attach(item, embedding)

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
                "cache_bytes": self._cache.current_bytes,
                "cache_evictions": self._cache.eviction_count,
            }

    def _cache_key(self, item: Any) -> str | None:
        item_hash = getattr(item, "audio_fingerprint", None)
        if item_hash is None:
            item_hash = getattr(item, "hash", None)
        if item_hash is None:
            return None
        return f"{self._namespace}:{item_hash}"

    def _is_valid(self, embedding: Any, expected_tokens: int) -> bool:
        return (
            isinstance(embedding, torch.Tensor)
            and embedding.dim() == 2
            and embedding.shape[0] == expected_tokens
            and embedding.shape[1] == self._hidden_size
            and embedding.dtype == self._dtype
        )

    def _attach(self, item: Any, embedding: torch.Tensor) -> None:
        item.precomputed_embeddings = embedding.to(self._device, non_blocking=True)
        item.feature = None
        item.format = MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    def _drain_batch(
        self,
    ) -> list[tuple[Any, concurrent.futures.Future[torch.Tensor], float]]:
        # Note (Akazaakane): Never wait here: this service overlaps individual
        # encodes today, while true batched inference belongs to F-PR5.
        batch = [self._queue.get()]
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _worker(self) -> None:
        while True:
            batch = self._drain_batch()
            dequeue_time = time.perf_counter()
            queue_waits = [dequeue_time - enqueued_at for _, _, enqueued_at in batch]
            with self._lock:
                self._queue_wait_count += len(queue_waits)
                self._queue_wait_total_s += sum(queue_waits)
                self._queue_wait_max_s = max(
                    self._queue_wait_max_s, max(queue_waits, default=0.0)
                )
            items = [item for item, _, _ in batch]
            encode_start = time.perf_counter()
            try:
                embeddings = self._encode_batch(items)
            except Exception as batch_exc:
                if len(batch) == 1:
                    batch[0][1].set_exception(batch_exc)
                    with self._lock:
                        self._encoder_time_s += time.perf_counter() - encode_start
                    continue
                logger.exception(
                    f"Fun-ASR batched audio encode failed for {len(items)} "
                    f"items; retrying per item"
                )
                recovered = 0
                for item, future, _ in batch:
                    try:
                        embedding = self._encode_batch([item])[0]
                        future.set_result(embedding)
                        recovered += 1
                    except Exception as item_exc:
                        future.set_exception(item_exc)
                with self._lock:
                    self._encoder_time_s += time.perf_counter() - encode_start
                    # Note (Akazaakane): Each recovered item re-encoded as its
                    # own single-item batch; count it as one so the items/batch
                    # telemetry stays honest for the F-PR5 gate.
                    self._batch_count += recovered
                    self._item_count += recovered
                continue
            for (_, future, _), embedding in zip(batch, embeddings):
                future.set_result(embedding)
            with self._lock:
                self._encoder_time_s += time.perf_counter() - encode_start
                self._batch_count += 1
                self._item_count += len(items)
                batch_count = self._batch_count
                item_count = self._item_count
            if batch_count % 50 == 1:
                logger.info(
                    f"Fun-ASR pre-LM encoder stage: {batch_count} batches, "
                    f"{item_count} items (avg "
                    f"{item_count / batch_count:.2f} items/batch, "
                    f"last batch: {len(items)}), cache: {self.stats()}"
                )

    def _encode_batch(self, items: list[Any]) -> list[torch.Tensor]:
        stream_context = (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), stream_context:
            embedding = self._model._get_audio_feature_uncached(items, None)
            token_counts = []
            for item in items:
                expected = _expected_audio_tokens(item)
                if expected is None:
                    raise RuntimeError(
                        "Fun-ASR pre-LM encode item is missing its audio token count"
                    )
                token_counts.append(expected)
            if (
                embedding.dim() != 2
                or embedding.shape[0] != sum(token_counts)
                or embedding.shape[1] != self._hidden_size
                or embedding.dtype != self._dtype
            ):
                raise RuntimeError(
                    f"Fun-ASR encoder output {tuple(embedding.shape)} "
                    f"({embedding.dtype}) != expected rows "
                    f"{sum(token_counts)}x{self._hidden_size} ({self._dtype})"
                )
            parts = torch.split(embedding, token_counts, dim=0)
            embeddings = [part.clone() for part in parts]
            for item, part in zip(items, embeddings):
                self._attach(item, part)
        if self._stream is not None:
            self._stream.synchronize()
        for item, part in zip(items, embeddings):
            key = self._cache_key(item)
            if key is not None:
                self._cache.put(key, part)
        return embeddings


__all__ = [
    "FunAsrPreLMEncoderService",
    "build_cache_namespace",
]
