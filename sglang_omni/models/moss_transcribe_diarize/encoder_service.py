"""Pre-LM audio-encoder service for MOSS-Transcribe-Diarize.

Encoding inside the LM forward stalls every running request at each prefill;
encoding at request-build time on a dedicated thread/stream lets the
compute-bound encoder overlap the memory-bound decode on the same GPU.
"""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = 4096
_CACHE_MAX_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class _DetachedFailure:
    exception: Exception
    formatted_traceback: str


class BatchedAudioEncoderService:
    ENCODE_TIMEOUT_S = 300.0

    def __init__(self, model: Any, *, max_batch_size: int = 2) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self._model = model
        self._max_batch_size = int(max_batch_size)
        self._device = next(model.whisper_encoder.parameters()).device
        self._stream = torch.cuda.Stream(device=self._device)
        self._cache = StageOutputCache(
            max_size=_CACHE_MAX_ENTRIES,
            max_bytes=_CACHE_MAX_BYTES,
            cache_device="cpu",
        )
        self._queue: queue.Queue[tuple[Any, concurrent.futures.Future]] = queue.Queue()
        self._batch_count = 0
        self._item_count = 0
        self._thread = threading.Thread(
            target=self._worker, name="moss-td-audio-encode", daemon=True
        )
        self._thread.start()

    def encode_item(self, item: Any) -> None:
        """Blocks until item.precomputed_embeddings is attached."""
        if item.hash is not None:
            cached = self._cache.get(str(item.hash))
        else:
            cached = None
        if cached is not None:
            item.precomputed_embeddings = cached.to(self._device, non_blocking=True)
            item.feature = None
            return
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((item, future))
        future.result(timeout=self.ENCODE_TIMEOUT_S)

    def _drain_batch(self) -> list[tuple[Any, concurrent.futures.Future]]:
        # note (yichi): never wait — a window costs 8~16ms at low concurrency, buys <=5ms at high.
        batch = [self._queue.get()]
        for _ in range(self._max_batch_size - 1):
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _worker(self) -> None:
        while True:
            self._process_batch(self._drain_batch())

    def _process_batch(
        self, batch: list[tuple[Any, concurrent.futures.Future]]
    ) -> None:
        items = [item for item, _ in batch]
        try:
            self._encode_batch(items)
        except Exception as batch_exc:
            if len(batch) == 1:
                failure = self._detach_failure(batch_exc)
                logger.error(
                    "MOSS-TD audio encode failed:\n%s",
                    failure.formatted_traceback,
                )
                self._recover_after_failure(failure.exception)
                batch[0][1].set_exception(failure.exception)
                return

            failure = self._detach_failure(batch_exc)
            logger.error(
                "MOSS-TD batched audio encode failed for %d items; "
                "retrying per item:\n%s",
                len(items),
                failure.formatted_traceback,
            )
            self._recover_after_failure(failure.exception)
            for item, future in batch:
                try:
                    self._encode_batch([item])
                except Exception as item_exc:
                    failure = self._detach_failure(item_exc)
                    logger.error(
                        "MOSS-TD per-item audio encode retry failed:\n%s",
                        failure.formatted_traceback,
                    )
                    self._recover_after_failure(failure.exception)
                    future.set_exception(failure.exception)
                else:
                    self._record_success(1)
                    future.set_result(None)
            return

        self._record_success(len(items))
        for _, future in batch:
            future.set_result(None)

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

    def _encode_batch(self, items: list[Any]) -> None:
        with torch.cuda.stream(self._stream):
            embedding = self._model._get_audio_feature_uncached(items, None)
            token_counts = [
                int(getattr(item, "audio_feature_lengths").sum()) for item in items
            ]
            if embedding.shape[0] != sum(token_counts):
                raise RuntimeError(
                    f"encoder output rows {embedding.shape[0]} != expected "
                    f"{sum(token_counts)}"
                )
            parts = tuple(
                part.contiguous()
                for part in torch.split(embedding, token_counts, dim=0)
            )
        self._stream.synchronize()
        for item, part in zip(items, parts):
            item.precomputed_embeddings = part
            item.feature = None
        for item in items:
            if item.hash is not None:
                self._cache.put(str(item.hash), item.precomputed_embeddings)
