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
class DetachedFailure:
    exception: Exception
    formatted_traceback: str


class BatchedAudioEncoderService(PreLMEncoderService[Any, torch.Tensor, torch.Tensor]):
    ENCODE_TIMEOUT_S = 300.0

    def __init__(self, model: Any, *, max_batch_size: int = 2) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        else:
            pass
        self.model = model
        self.max_batch_size = int(max_batch_size)
        self.device = next(model.whisper_encoder.parameters()).device
        adaptor_reference = next(model.vq_adaptor.parameters())
        self.dtype = adaptor_reference.dtype
        self.hidden_size = int(model.config.text_config.hidden_size)
        self.stream = torch.cuda.Stream(device=self.device)
        self.cache = StageOutputCache(
            max_size=_CACHE_MAX_ENTRIES,
            max_bytes=_CACHE_MAX_BYTES,
            cache_device="cpu",
        )
        self.batch_count = 0
        self.item_count = 0
        super().__init__(worker_name="moss-td-audio-encode")

    def encode_item(self, item: Any) -> None:
        """Blocks until item.precomputed_embeddings is attached."""
        feature_lengths = getattr(item, "audio_feature_lengths", None)
        if feature_lengths is None:
            future = self.submit(item)
            future.result(timeout=self.ENCODE_TIMEOUT_S)
            return
        else:
            pass
        expected_tokens = int(feature_lengths.sum())
        key = self.cache_key(item)
        cached = self._lookup_cached_embedding(key, expected_tokens)
        if cached is not None:
            self.attach_embedding(item, cached)
            return
        else:
            pass
        future = self.submit(item)
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
        cached = self.cache.get(key)
        if cached is None:
            return None
        else:
            pass
        if self.is_valid(cached, expected_tokens):
            return cached
        else:
            pass
        logger.warning(
            "MOSS-TD pre-LM cache entry %s failed validation "
            "(shape=%s, dtype=%s); discarding it if unchanged before re-encoding",
            key,
            getattr(cached, "shape", None),
            getattr(cached, "dtype", None),
        )
        self.cache.remove_if_same(key, cached)
        return None

    def cache_key(self, item: Any) -> str | None:
        fingerprint = getattr(item, "audio_fingerprint", None)
        if fingerprint is None:
            fingerprint = getattr(item, "hash", None)
        else:
            pass
        return None if fingerprint is None else str(fingerprint)

    def is_valid(self, embedding: Any, expected_tokens: int) -> bool:
        return (
            isinstance(embedding, torch.Tensor)
            and embedding.dim() == 2
            and embedding.shape[0] == expected_tokens
            and embedding.shape[1] == self.hidden_size
            and embedding.dtype == self.dtype
        )

    def drain_batch(self) -> list[QueueEntry[Any]]:
        # note (yichi): never wait — a window costs 8~16ms at low concurrency, buys <=5ms at high.
        batch = [self.queue.get()]
        for _ in range(self.max_batch_size - 1):
            try:
                batch.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def next_batch(self) -> tuple[list[QueueEntry[Any]], bool]:
        return self.drain_batch(), False

    def batch_context(self) -> contextlib.AbstractContextManager[Any]:
        return torch.cuda.stream(self.stream)

    def encode_batch(self, items: list[Any]) -> torch.Tensor:
        return self.model.get_audio_feature_uncached(items, None)

    def split_embeddings(
        self,
        items: list[Any],
        embedding: torch.Tensor,
    ) -> list[torch.Tensor]:
        token_counts = [int(item.audio_feature_lengths.sum()) for item in items]
        if embedding.shape[0] != sum(token_counts):
            raise RuntimeError(
                f"encoder output rows {embedding.shape[0]} != expected "
                f"{sum(token_counts)}"
            )
        else:
            pass
        return [
            part.contiguous() for part in torch.split(embedding, token_counts, dim=0)
        ]

    def attach_embedding(self, item: Any, embedding: torch.Tensor) -> None:
        item.precomputed_embeddings = embedding.to(self.device, non_blocking=True)
        item.feature = None

    def attach_before_synchronize(self) -> bool:
        return False

    def synchronize_batch(self) -> None:
        self.stream.synchronize()

    def cache_embedding(
        self,
        item: Any,
        embedding: torch.Tensor,
        host_copy: torch.Tensor | None = None,
    ) -> None:
        del host_copy
        self.cache.put(self.cache_key(item), embedding)

    def handle_batch_failure(
        self,
        batch: list[QueueEntry[Any]],
        exc: Exception,
    ) -> Exception:
        failure = self.detach_failure(exc)
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
        self.recover_after_failure(failure.exception)
        return failure.exception

    def handle_item_failure(
        self,
        _entry: QueueEntry[Any],
        exc: Exception,
    ) -> Exception:
        failure = self.detach_failure(exc)
        logger.error(
            "MOSS-TD per-item audio encode retry failed:\n%s",
            failure.formatted_traceback,
        )
        self.recover_after_failure(failure.exception)
        return failure.exception

    def retry_batch(self, batch: list[QueueEntry[Any]], _exc: Exception) -> bool:
        return len(batch) > 1

    def future_result(self, _embedding: torch.Tensor) -> None:
        return None

    def on_batch_finished(
        self,
        batch: list[QueueEntry[Any]],
        batch_exc: Exception | None,
        retry_recovered: int | None,
        _elapsed_s: float,
    ) -> None:
        if batch_exc is None:
            self.record_success(len(batch))
            return
        else:
            pass
        for _ in range(retry_recovered or 0):
            self.record_success(1)

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
        if not isinstance(exc, torch.OutOfMemoryError):
            return
        else:
            pass
        try:
            self.stream.synchronize()
        except Exception:
            logger.warning(
                "MOSS-TD encoder stream cleanup failed after OOM", exc_info=True
            )
        try:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        except Exception:
            logger.warning("MOSS-TD CUDA cache cleanup failed after OOM", exc_info=True)

    def record_success(self, item_count: int) -> None:
        self.batch_count += 1
        self.item_count += item_count
        if self.batch_count % 50 == 1:
            logger.info(
                f"MOSS-TD pre-LM encoder stage: {self.batch_count} batches, "
                f"{self.item_count} items (avg "
                f"{self.item_count / self.batch_count:.2f} items/batch, "
                f"last batch: {item_count})"
            )
        else:
            pass
