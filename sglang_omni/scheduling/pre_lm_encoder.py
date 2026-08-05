# SPDX-License-Identifier: Apache-2.0
"""Shared mechanics for pre-LM encoder services."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

ItemT = TypeVar("ItemT")
EncodedT = TypeVar("EncodedT")
EmbeddingT = TypeVar("EmbeddingT")

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueEntry(Generic[ItemT]):
    item: ItemT
    future: concurrent.futures.Future[Any]
    enqueued_at: float | None = None


class PreLMEncoderService(ABC, Generic[ItemT, EncodedT, EmbeddingT]):
    """Run model-owned encoder hooks on a queue-backed worker thread."""

    def __init__(self, *, worker_name: str) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._worker_state_lock = threading.Lock()
        self._worker_error: Exception | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name=worker_name,
            daemon=True,
        )
        self._thread.start()

    def _enqueue(
        self,
        item: ItemT,
        future: concurrent.futures.Future[Any],
    ) -> None:
        self._queue.put(QueueEntry(item=item, future=future))

    def _submit(
        self,
        item: ItemT,
        future: concurrent.futures.Future[Any] | None = None,
    ) -> concurrent.futures.Future[Any]:
        if future is None:
            future = concurrent.futures.Future()
        with self._worker_state_lock:
            if self._worker_error is not None:
                raise RuntimeError(
                    "pre-LM encoder worker has failed"
                ) from self._worker_error
            self._enqueue(item, future)
        return future

    @abstractmethod
    def _next_batch(self) -> tuple[list[QueueEntry[ItemT]], bool]:
        raise NotImplementedError

    def _batch_context(self) -> AbstractContextManager[Any]:
        return contextlib.nullcontext()

    @abstractmethod
    def encode_batch(self, items: list[ItemT]) -> EncodedT:
        raise NotImplementedError

    @abstractmethod
    def split_embeddings(
        self,
        items: list[ItemT],
        encoded: EncodedT,
    ) -> list[EmbeddingT]:
        raise NotImplementedError

    @abstractmethod
    def attach_embedding(self, item: ItemT, embedding: EmbeddingT) -> None:
        raise NotImplementedError

    def attach_before_synchronize(self) -> bool:
        return True

    def synchronize_batch(self) -> None:
        pass

    def cache_embedding(self, item: ItemT, embedding: EmbeddingT) -> None:
        pass

    def _execute_batch(self, items: list[ItemT]) -> list[EmbeddingT]:
        attach_before_synchronize = self.attach_before_synchronize()
        with self._batch_context():
            encoded = self.encode_batch(items)
            embeddings = self.split_embeddings(items, encoded)
            if len(embeddings) != len(items):
                raise RuntimeError(
                    f"split_embeddings returned {len(embeddings)} embeddings "
                    f"for {len(items)} items"
                )
            if attach_before_synchronize:
                for item, embedding in zip(items, embeddings):
                    self.attach_embedding(item, embedding)
        self.synchronize_batch()
        if not attach_before_synchronize:
            for item, embedding in zip(items, embeddings):
                self.attach_embedding(item, embedding)
        for item, embedding in zip(items, embeddings):
            self.cache_embedding(item, embedding)
        return embeddings

    def _handle_batch_failure(
        self,
        batch: list[QueueEntry[ItemT]],
        exc: Exception,
    ) -> Exception:
        return exc

    def _handle_item_failure(
        self,
        entry: QueueEntry[ItemT],
        exc: Exception,
    ) -> Exception:
        return exc

    def _retry_batch(
        self,
        batch: list[QueueEntry[ItemT]],
        exc: Exception,
    ) -> bool:
        return False

    def _future_result(self, embedding: EmbeddingT) -> Any:
        return embedding

    def _on_batch_start(self, batch: list[QueueEntry[ItemT]]) -> None:
        pass

    def _on_batch_finished(
        self,
        batch: list[QueueEntry[ItemT]],
        batch_exc: Exception | None,
        retry_recovered: int | None,
        elapsed_s: float,
    ) -> None:
        pass

    @staticmethod
    def _set_exception(entry: QueueEntry[ItemT], exc: Exception) -> None:
        if entry.future.done():
            return
        try:
            entry.future.set_exception(exc)
        except concurrent.futures.InvalidStateError:
            logger.warning("pre-LM encoder future completed before exception dispatch")

    def _set_result(self, entry: QueueEntry[ItemT], embedding: EmbeddingT) -> None:
        if entry.future.done():
            return
        try:
            result = self._future_result(embedding)
            entry.future.set_result(result)
        except concurrent.futures.InvalidStateError:
            logger.warning("pre-LM encoder future completed before result dispatch")
        except Exception as exc:
            self._set_exception(entry, exc)

    def _notify_batch_start(self, batch: list[QueueEntry[ItemT]]) -> None:
        try:
            self._on_batch_start(batch)
        except Exception:
            logger.exception("pre-LM encoder batch-start hook failed")

    def _notify_batch_finished(
        self,
        batch: list[QueueEntry[ItemT]],
        batch_exc: Exception | None,
        retry_recovered: int | None,
        elapsed_s: float,
    ) -> None:
        try:
            self._on_batch_finished(
                batch,
                batch_exc,
                retry_recovered,
                elapsed_s,
            )
        except Exception:
            logger.exception("pre-LM encoder batch-finished hook failed")

    def _fail_worker(
        self,
        exc: Exception,
        current_batch: list[QueueEntry[ItemT]],
    ) -> None:
        with self._worker_state_lock:
            self._worker_error = exc
            pending = list(current_batch)
            while True:
                try:
                    queued = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(queued, QueueEntry):
                    pending.append(queued)
        for entry in pending:
            self._set_exception(entry, exc)

    def _worker(self) -> None:
        batch: list[QueueEntry[ItemT]] = []
        try:
            while True:
                batch, shutdown = self._next_batch()
                if not batch:
                    return
                self._notify_batch_start(batch)
                items = [entry.item for entry in batch]
                encode_start = time.perf_counter()
                try:
                    embeddings = self._execute_batch(items)
                except Exception as batch_exc:
                    batch_exc = self._handle_batch_failure(batch, batch_exc)
                    if not self._retry_batch(batch, batch_exc):
                        for entry in batch:
                            self._set_exception(entry, batch_exc)
                        self._notify_batch_finished(
                            batch,
                            batch_exc,
                            None,
                            time.perf_counter() - encode_start,
                        )
                        if shutdown:
                            return
                        batch = []
                        continue
                    recovered = 0
                    for entry in batch:
                        try:
                            embedding = self._execute_batch([entry.item])[0]
                            self._set_result(entry, embedding)
                            recovered += 1
                        except Exception as item_exc:
                            item_exc = self._handle_item_failure(entry, item_exc)
                            self._set_exception(entry, item_exc)
                    self._notify_batch_finished(
                        batch,
                        batch_exc,
                        recovered,
                        time.perf_counter() - encode_start,
                    )
                    if shutdown:
                        return
                    batch = []
                    continue
                for entry, embedding in zip(batch, embeddings):
                    self._set_result(entry, embedding)
                self._notify_batch_finished(
                    batch,
                    None,
                    None,
                    time.perf_counter() - encode_start,
                )
                if shutdown:
                    return
                batch = []
        except Exception as worker_exc:
            logger.exception("pre-LM encoder worker failed")
            self._fail_worker(worker_exc, batch)


__all__ = ["PreLMEncoderService", "QueueEntry"]
