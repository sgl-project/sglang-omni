# SPDX-License-Identifier: Apache-2.0
"""Encode Nemotron perception embeddings before decoder admission.

A long waveform used to run inside ``general_mm_embed_routine`` during the
decoder step, so one upload stalled every running request. This service owns
that encode on a worker thread and a side device stream. The request is
admitted only after ``precomputed_embeddings`` is attached, which makes the
decoder scatter those rows instead of calling the encoder.
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
from typing import TYPE_CHECKING, TypeGuard

import torch
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputFormat
from sglang.srt.utils import create_device_stream, device_stream_context

from sglang_omni.scheduling.pre_lm_encoder import (
    PreLMEncoderService,
    QueueEntry,
    QueueSignal,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache

if TYPE_CHECKING:
    from sglang_omni.models.nemotron_transcribe.sglang_model import (
        NemotronTranscribeForConditionalGeneration,
    )
else:
    pass

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = 128
_CACHE_MAX_BYTES = 2 * 1024**3
# Clips longer than this are encoded alone. Perception pads a batch to its
# longest waveform, so a 30-minute upload must not stretch short requests.
SOLO_WAVEFORM_SECONDS = 60
_SAMPLE_RATE = 16000
_LENGTH_RATIO_LIMIT = 4


@dataclass(frozen=True)
class DetachedFailure:
    exception: Exception
    formatted_traceback: str


def build_cache_namespace(
    model: NemotronTranscribeForConditionalGeneration, *, model_path: str
) -> str:
    """Digest identifying this process's perception pipeline for cache keying."""
    config = getattr(model, "config", None)
    reference = next(model.perception.parameters())
    payload = {
        "model_path": model_path,
        "perception": getattr(config, "perception", None),
        "pe_encoder_config": getattr(config, "pe_encoder_config", None),
        "audio_token_estimator": getattr(config, "audio_token_estimator", None),
        "dtype": str(reference.dtype),
        "device_type": reference.device.type,
        "hidden_size": int(model.perception.proj.out_features),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def expected_audio_tokens(item: MultimodalDataItem) -> int | None:
    num_tokens = getattr(item, "num_audio_tokens", None)
    return int(num_tokens) if num_tokens is not None else None


def waveform_samples(item: MultimodalDataItem) -> int:
    feature = item.feature
    if isinstance(feature, torch.Tensor):
        return int(feature.numel())
    else:
        return 0


def clips_share_a_batch(first_samples: int, candidate_samples: int) -> bool:
    solo_samples = SOLO_WAVEFORM_SECONDS * _SAMPLE_RATE
    if first_samples > solo_samples or candidate_samples > solo_samples:
        return False
    else:
        pass
    if first_samples <= 0 or candidate_samples <= 0:
        return True
    else:
        pass
    return max(first_samples, candidate_samples) <= (
        min(first_samples, candidate_samples) * _LENGTH_RATIO_LIMIT
    )


class NemotronTranscribePreLMEncoderService(
    PreLMEncoderService[MultimodalDataItem, torch.Tensor, torch.Tensor]
):
    """Encode before admission with single-flight deduplication and a CPU LRU."""

    ENCODE_TIMEOUT_S = 3600.0

    def __init__(
        self,
        model: NemotronTranscribeForConditionalGeneration,
        *,
        cache_namespace: str,
        cache_max_entries: int = _CACHE_MAX_ENTRIES,
        cache_max_bytes: int = _CACHE_MAX_BYTES,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 0,
    ) -> None:
        if cache_max_entries < 1:
            raise ValueError(
                f"cache_max_entries must be >= 1, got {cache_max_entries}"
            )
        else:
            pass
        if cache_max_bytes < 1:
            raise ValueError(f"cache_max_bytes must be >= 1, got {cache_max_bytes}")
        else:
            pass
        self.model = model
        reference = next(model.perception.parameters())
        self.device = reference.device
        self.dtype = reference.dtype
        self.hidden_size = int(model.perception.proj.out_features)
        self.stream = (
            create_device_stream(self.device)
            if self.device.type in {"cuda", "npu"}
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
        self.held: QueueEntry[MultimodalDataItem, torch.Tensor] | None = None
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
        super().__init__(worker_name="nemotron-transcribe-audio-encode")

    def close(self) -> None:
        """Stop the encoder worker after queued requests finish."""
        with self.lifecycle_lock:
            if self.closed:
                return
            else:
                pass
            self.closed = True
            self.queue.put(QueueSignal.SHUTDOWN)
        self.thread.join(timeout=5)

    def enqueue(
        self,
        item: MultimodalDataItem,
        future: concurrent.futures.Future[torch.Tensor],
    ) -> None:
        with self.lifecycle_lock:
            if self.closed:
                raise RuntimeError(
                    "Nemotron Transcribe pre-LM encoder service is closed"
                )
            else:
                pass
            self.queue.put(
                QueueEntry(
                    item=item,
                    future=future,
                    enqueued_at=time.perf_counter(),
                )
            )

    def submit_item(
        self, item: MultimodalDataItem
    ) -> concurrent.futures.Future[torch.Tensor]:
        """Queue the item and return the future completed after attach."""
        expected_tokens = expected_audio_tokens(item)
        if expected_tokens is None:
            raise RuntimeError(
                "Nemotron Transcribe pre-LM encode requires num_audio_tokens"
            )
        else:
            pass
        key = self.cache_key(item)
        if key is None:
            return self.count_failed(self.submit(item))
        else:
            pass

        cached = self.lookup_cached_embedding(
            getattr(item, "audio_fingerprint", None), expected_tokens
        )
        if cached is not None:
            self.attach_embedding(item, cached)
            done: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
            done.set_result(cached)
            return done
        else:
            pass

        follower_of: concurrent.futures.Future[torch.Tensor] | None = None
        leader = False
        with self.lock:
            future = self.inflight.get(key)
            if future is None:
                cached = self.cache.get(key)
                if cached is not None and self.is_valid(cached, expected_tokens):
                    self.hits += 1
                else:
                    cached = None
                    future = concurrent.futures.Future()
                    self.inflight[key] = future
                    leader = True
                    self.misses += 1
            else:
                self.merged += 1
                follower_of = future
        if cached is not None:
            self.attach_embedding(item, cached)
            completed: concurrent.futures.Future[torch.Tensor] = (
                concurrent.futures.Future()
            )
            completed.set_result(cached)
            return completed
        else:
            pass
        if leader:
            future.add_done_callback(
                lambda done, cache_key=key: self.clear_inflight(cache_key, done)
            )
            self.count_failed(future)
            try:
                self.submit(item, future)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                else:
                    pass
                raise
            return future
        else:
            pass

        item.feature = None
        completion: concurrent.futures.Future[torch.Tensor] = (
            concurrent.futures.Future()
        )

        def attach_follower(done: concurrent.futures.Future[torch.Tensor]) -> None:
            try:
                embedding = done.result()
                if not self.is_valid(embedding, expected_tokens):
                    raise RuntimeError(
                        f"Nemotron Transcribe pre-LM encode leader for {key} "
                        "returned an invalid embedding"
                    )
                else:
                    pass
                self.attach_embedding(item, embedding)
                completion.set_result(embedding)
            except Exception as exc:
                completion.set_exception(exc)

        assert follower_of is not None
        follower_of.add_done_callback(attach_follower)
        return self.count_failed(completion)

    def encode_item(self, item: MultimodalDataItem) -> None:
        """Block until the item holds the language-model embedding."""
        self.submit_item(item).result(timeout=self.ENCODE_TIMEOUT_S)

    def count_failed(
        self, future: concurrent.futures.Future[torch.Tensor]
    ) -> concurrent.futures.Future[torch.Tensor]:
        def finish(done: concurrent.futures.Future[torch.Tensor]) -> None:
            try:
                failed = done.exception() is not None
            except concurrent.futures.CancelledError:
                failed = True
            if failed:
                with self.lock:
                    self.failed += 1
            else:
                pass

        future.add_done_callback(finish)
        return future

    def clear_inflight(
        self, key: str, future: concurrent.futures.Future[torch.Tensor]
    ) -> None:
        with self.lock:
            if self.inflight.get(key) is future:
                del self.inflight[key]
            else:
                pass

    def lookup_cached_embedding(
        self, audio_fingerprint: str | None, expected_tokens: int
    ) -> torch.Tensor | None:
        key = self.cache_key_from_fingerprint(audio_fingerprint)
        cached = self.cache.get(key)
        if cached is None:
            return None
        else:
            pass
        if self.is_valid(cached, expected_tokens):
            with self.lock:
                self.hits += 1
            return cached
        else:
            pass
        logger.warning(
            "Nemotron Transcribe pre-LM cache entry %s failed validation "
            "(shape=%s, dtype=%s); discarding it if unchanged",
            key,
            getattr(cached, "shape", None),
            getattr(cached, "dtype", None),
        )
        self.cache.remove_if_same(key, cached)
        return None

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

    def cache_key(self, item: MultimodalDataItem) -> str | None:
        return self.cache_key_from_fingerprint(getattr(item, "audio_fingerprint", None))

    def cache_key_from_fingerprint(self, audio_fingerprint: str | None) -> str | None:
        if audio_fingerprint is None:
            return None
        else:
            pass
        return f"{self.namespace}:{audio_fingerprint}"

    def is_valid(
        self, embedding: object, expected_tokens: int
    ) -> TypeGuard[torch.Tensor]:
        return (
            isinstance(embedding, torch.Tensor)
            and embedding.dim() == 2
            and embedding.shape[0] == expected_tokens
            and embedding.shape[1] == self.hidden_size
            and embedding.dtype == self.dtype
        )

    def attach_embedding(
        self, item: MultimodalDataItem, embedding: torch.Tensor
    ) -> None:
        embedding = embedding.to(self.device, non_blocking=True)
        if self.stream is not None:
            device_module = torch.get_device_module(embedding.device)
            embedding.record_stream(device_module.default_stream(embedding.device))
        else:
            pass
        item.precomputed_embeddings = embedding
        item.feature = None
        item.format = MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    def next_batch(
        self,
    ) -> tuple[list[QueueEntry[MultimodalDataItem, torch.Tensor]], bool]:
        first = self.held if self.held is not None else self.queue.get()
        self.held = None
        if first is QueueSignal.SHUTDOWN:
            return [], True
        else:
            pass
        batch = [first]
        solo_samples = SOLO_WAVEFORM_SECONDS * _SAMPLE_RATE
        if waveform_samples(first.item) > solo_samples:
            # A long clip is its own batch. Leave later arrivals queued so a
            # 30-minute upload does not pad every short clip to its length.
            return batch, False
        else:
            pass
        deadline = time.monotonic() + self.max_batch_wait_s
        shutdown = False
        first_samples = waveform_samples(first.item)
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
            if queued is QueueSignal.SHUTDOWN:
                shutdown = True
                break
            else:
                pass
            if not clips_share_a_batch(first_samples, waveform_samples(queued.item)):
                self.held = queued
                break
            else:
                pass
            batch.append(queued)
        return batch, shutdown

    @contextlib.contextmanager
    def batch_context(self) -> Iterator[None]:
        with torch.inference_mode():
            if self.stream is None:
                yield
            else:
                with device_stream_context(self.stream):
                    yield

    def encode_batch(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self.model.get_audio_feature(items)

    def split_embeddings(
        self, items: list[MultimodalDataItem], embedding: torch.Tensor
    ) -> list[torch.Tensor]:
        token_counts = []
        for item in items:
            expected = expected_audio_tokens(item)
            if expected is None:
                raise RuntimeError(
                    "Nemotron Transcribe pre-LM encode item is missing its "
                    "audio token count"
                )
            else:
                pass
            token_counts.append(expected)
        if embedding.dim() == 3 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        else:
            pass
        if (
            embedding.dim() != 2
            or embedding.shape[0] != sum(token_counts)
            or embedding.shape[1] != self.hidden_size
            or embedding.dtype != self.dtype
        ):
            raise RuntimeError(
                f"Nemotron Transcribe encoder output {tuple(embedding.shape)} "
                f"({embedding.dtype}) != expected rows "
                f"{sum(token_counts)}x{self.hidden_size} ({self.dtype})"
            )
        else:
            pass
        return [part.clone() for part in torch.split(embedding, token_counts, dim=0)]

    def synchronize_batch(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        else:
            pass

    def cache_embedding(
        self,
        item: MultimodalDataItem,
        embedding: torch.Tensor,
        host_copy: torch.Tensor | None = None,
    ) -> None:
        del host_copy
        key = self.cache_key(item)
        if key is not None:
            self.cache.put(key, embedding)
        else:
            pass

    def retry_batch(
        self,
        batch: list[QueueEntry[MultimodalDataItem, torch.Tensor]],
        _exc: Exception,
    ) -> bool:
        return len(batch) > 1

    def handle_batch_failure(
        self,
        batch: list[QueueEntry[MultimodalDataItem, torch.Tensor]],
        exc: Exception,
    ) -> Exception:
        failure = self.detach_failure(exc)
        if len(batch) == 1:
            logger.error(
                "Nemotron Transcribe audio encode failed:\n%s",
                failure.formatted_traceback,
            )
        else:
            logger.error(
                "Nemotron Transcribe batched audio encode failed for %d items; "
                "retrying per item:\n%s",
                len(batch),
                failure.formatted_traceback,
            )
        self.recover_after_failure(failure.exception)
        return failure.exception

    def handle_item_failure(
        self,
        _entry: QueueEntry[MultimodalDataItem, torch.Tensor],
        exc: Exception,
    ) -> Exception:
        failure = self.detach_failure(exc)
        logger.error(
            "Nemotron Transcribe per-item audio encode retry failed:\n%s",
            failure.formatted_traceback,
        )
        self.recover_after_failure(failure.exception)
        return failure.exception

    @staticmethod
    def detach_failure(exc: Exception) -> DetachedFailure:
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
        return DetachedFailure(
            exception=detached, formatted_traceback=formatted_traceback
        )

    def recover_after_failure(self, exc: Exception) -> None:
        if not isinstance(exc, torch.OutOfMemoryError):
            return
        else:
            pass
        if self.stream is not None:
            try:
                self.stream.synchronize()
            except Exception:
                logger.warning(
                    "Nemotron Transcribe encoder stream cleanup failed after OOM",
                    exc_info=True,
                )
        else:
            pass
        try:
            device_module = torch.get_device_module(self.device)
            with device_module.device(self.device):
                device_module.empty_cache()
        except Exception:
            logger.warning(
                "Nemotron Transcribe device cache cleanup failed after OOM",
                exc_info=True,
            )

    def on_batch_start(
        self, batch: list[QueueEntry[MultimodalDataItem, torch.Tensor]]
    ) -> None:
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
                self.queue_wait_max_s, max(queue_waits, default=0.0)
            )

    def on_batch_finished(
        self,
        batch: list[QueueEntry[MultimodalDataItem, torch.Tensor]],
        batch_exc: Exception | None,
        retry_recovered: int | None,
        elapsed_s: float,
    ) -> None:
        with self.lock:
            self.encoder_time_s += elapsed_s
            if batch_exc is not None:
                if retry_recovered is not None:
                    self.batch_count += retry_recovered
                    self.item_count += retry_recovered
                else:
                    pass
                return
            else:
                pass
            self.batch_count += 1
            self.item_count += len(batch)
            batch_count = self.batch_count
            item_count = self.item_count
        if batch_count % 50 == 1:
            logger.info(
                "Nemotron Transcribe pre-LM encoder stage: %d batches, "
                "%d items (avg %.2f items/batch, last batch: %d), cache: %s",
                batch_count,
                item_count,
                item_count / batch_count,
                len(batch),
                self.stats(),
            )
        else:
            pass


__all__ = [
    "NemotronTranscribePreLMEncoderService",
    "build_cache_namespace",
    "clips_share_a_batch",
]
