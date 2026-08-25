"""Pre-LM audio-encoder service for MOSS-Transcribe-Diarize.

Encoding inside the LM forward stalls every running request at each prefill;
encoding at request-build time on a dedicated thread/stream lets the
compute-bound encoder overlap the memory-bound decode on the same GPU.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import traceback
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.scheduling.pre_lm_encoder import PreLMEncoderService, QueueEntry
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = 4096
_CACHE_MAX_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class _DetachedFailure:
    exception: Exception
    formatted_traceback: str


class BatchedAudioEncoderService(PreLMEncoderService[Any, torch.Tensor, torch.Tensor]):
    ENCODE_TIMEOUT_S = 300.0

    def __init__(self, model: Any, *, max_batch_size: int = 2) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self._model = model
        self._max_batch_size = int(max_batch_size)
        self._device = next(model.whisper_encoder.parameters()).device
        adaptor_reference = next(model.vq_adaptor.parameters())
        self._dtype = adaptor_reference.dtype
        self._hidden_size = int(model.config.text_config.hidden_size)
        self._stream = torch.cuda.Stream(device=self._device)
        self._cache = StageOutputCache(
            max_size=_CACHE_MAX_ENTRIES,
            max_bytes=_CACHE_MAX_BYTES,
            cache_device="cpu",
        )
        self._batch_count = 0
        self._item_count = 0
        super().__init__(worker_name="moss-td-audio-encode")

    def encode_item(self, item: Any) -> None:
        """Blocks until item.precomputed_embeddings is attached."""
        feature_lengths = getattr(item, "audio_feature_lengths", None)
        if feature_lengths is None:
            future = self._submit(item)
            future.result(timeout=self.ENCODE_TIMEOUT_S)
            return
        expected_tokens = int(feature_lengths.sum())
        key = self._cache_key(item)
        cached = self._lookup_cached_embedding(key, expected_tokens)
        if cached is not None:
            self.attach_embedding(item, cached)
            return
        future = self._submit(item)
        future.result(timeout=self.ENCODE_TIMEOUT_S)

    def lookup_cached_embedding(
        self,
        audio_fingerprint: str,
        expected_tokens: int,
    ) -> torch.Tensor | None:
        """Return a validated cached embedding without starting an encode."""
        return self._lookup_cached_embedding(str(audio_fingerprint), expected_tokens)

    def _lookup_cached_embedding(
        self,
        key: str | None,
        expected_tokens: int,
    ) -> torch.Tensor | None:
        cached = self._cache.get(key)
        if cached is None:
            return None
        if self._is_valid(cached, expected_tokens):
            return cached
        logger.warning(
            "MOSS-TD pre-LM cache entry %s failed validation "
            "(shape=%s, dtype=%s); discarding it if unchanged before re-encoding",
            key,
            getattr(cached, "shape", None),
            getattr(cached, "dtype", None),
        )
        self._cache.remove_if_same(key, cached)
        return None

    def _cache_key(self, item: Any) -> str | None:
        fingerprint = getattr(item, "audio_fingerprint", None)
        if fingerprint is None:
            fingerprint = getattr(item, "hash", None)
        return None if fingerprint is None else str(fingerprint)

    def _is_valid(self, embedding: Any, expected_tokens: int) -> bool:
        return (
            isinstance(embedding, torch.Tensor)
            and embedding.dim() == 2
            and embedding.shape[0] == expected_tokens
            and embedding.shape[1] == self._hidden_size
            and embedding.dtype == self._dtype
        )

    def _drain_batch(self) -> list[QueueEntry[Any]]:
        # note (yichi): never wait — a window costs 8~16ms at low concurrency, buys <=5ms at high.
        batch = [self._queue.get()]
        for _ in range(self._max_batch_size - 1):
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _next_batch(self) -> tuple[list[QueueEntry[Any]], bool]:
        return self._drain_batch(), False

    def _batch_context(self) -> contextlib.AbstractContextManager[Any]:
        return torch.cuda.stream(self._stream)

    def encode_batch(self, items: list[Any]) -> torch.Tensor:
        return self._model._get_audio_feature_uncached(items, None)

    def split_embeddings(
        self,
        items: list[Any],
        embedding: torch.Tensor,
    ) -> list[torch.Tensor]:
        token_counts = [
            int(getattr(item, "audio_feature_lengths").sum()) for item in items
        ]
        if embedding.shape[0] != sum(token_counts):
            raise RuntimeError(
                f"encoder output rows {embedding.shape[0]} != expected "
                f"{sum(token_counts)}"
            )
        return [
            part.contiguous() for part in torch.split(embedding, token_counts, dim=0)
        ]

    def attach_embedding(self, item: Any, embedding: torch.Tensor) -> None:
        item.precomputed_embeddings = embedding.to(self._device, non_blocking=True)
        item.feature = None

    def attach_before_synchronize(self) -> bool:
        return False

    def synchronize_batch(self) -> None:
        self._stream.synchronize()

    def cache_embedding(
        self,
        item: Any,
        embedding: torch.Tensor,
        host_copy: torch.Tensor | None = None,
    ) -> None:
        del host_copy
        self._cache.put(self._cache_key(item), embedding)

    def _handle_batch_failure(
        self,
        batch: list[QueueEntry[Any]],
        exc: Exception,
    ) -> Exception:
        failure = self._detach_failure(exc)
        if len(batch) == 1:
            logger.error(
                "MOSS-TD audio encode failed:\n%s",
                failure.formatted_traceback,
            )
        else:
            logger.error(
                "MOSS-TD batched audio encode failed for %d items; "
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
            "MOSS-TD per-item audio encode retry failed:\n%s",
            failure.formatted_traceback,
        )
        self._recover_after_failure(failure.exception)
        return failure.exception

    def _retry_batch(self, batch: list[QueueEntry[Any]], _exc: Exception) -> bool:
        return len(batch) > 1

    def _future_result(self, _embedding: torch.Tensor) -> None:
        return None

    def _on_batch_finished(
        self,
        batch: list[QueueEntry[Any]],
        batch_exc: Exception | None,
        retry_recovered: int | None,
        _elapsed_s: float,
    ) -> None:
        if batch_exc is None:
            self._record_success(len(batch))
            return
        for _ in range(retry_recovered or 0):
            self._record_success(1)

    @staticmethod
    def _detach_failure(exc: Exception) -> _DetachedFailure:
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
        return _DetachedFailure(
            exception=detached,
            formatted_traceback=formatted_traceback,
        )

    def _recover_after_failure(self, exc: Exception) -> None:
        if not isinstance(exc, torch.OutOfMemoryError):
            return
        try:
            self._stream.synchronize()
        except Exception:
            logger.warning(
                "MOSS-TD encoder stream cleanup failed after OOM", exc_info=True
            )
        try:
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
        except Exception:
            logger.warning("MOSS-TD CUDA cache cleanup failed after OOM", exc_info=True)

    def _record_success(self, item_count: int) -> None:
        self._batch_count += 1
        self._item_count += item_count
        if self._batch_count % 50 == 1:
            logger.info(
                f"MOSS-TD pre-LM encoder stage: {self._batch_count} batches, "
                f"{self._item_count} items (avg "
                f"{self._item_count / self._batch_count:.2f} items/batch, "
                f"last batch: {item_count})"
            )
