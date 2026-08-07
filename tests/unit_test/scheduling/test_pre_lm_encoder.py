# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator

import pytest

from sglang_omni.scheduling.pre_lm_encoder import PreLMEncoderService, QueueEntry

_STOP = object()


class _Service(PreLMEncoderService[int, list[int], int]):
    def __init__(self, *, controlled_drain: bool = False) -> None:
        self.attachments: list[tuple[int, int]] = []
        self.cached: list[tuple[int, int]] = []
        self.context_events: list[str] = []
        self.fail_multi = False
        self.fail_items: set[int] = set()
        self.retry = False
        self.retry_hook_error = False
        self.start_hook_error = False
        self.split_mode = "exact"
        self.drain_gate = threading.Event() if controlled_drain else None
        super().__init__(worker_name="test-pre-lm-encoder")

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout=2)

    def _next_batch(self) -> tuple[list[QueueEntry[int]], bool]:
        first = self._queue.get()
        if first is _STOP:
            return [], True
        if self.drain_gate is not None:
            assert self.drain_gate.wait(timeout=2)
        batch = [first]
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch, False

    @contextlib.contextmanager
    def _batch_context(self) -> Iterator[None]:
        self.context_events.append("enter")
        try:
            yield
        finally:
            self.context_events.append("exit")

    def encode_batch(self, items: list[int]) -> list[int]:
        if self.fail_multi and len(items) > 1:
            raise RuntimeError("batch encode failed")
        if any(item in self.fail_items for item in items):
            raise RuntimeError("item encode failed")
        return [item * 2 for item in items]

    def split_embeddings(self, items: list[int], encoded: list[int]) -> list[int]:
        if self.split_mode == "too_few":
            return encoded[:-1]
        if self.split_mode == "too_many":
            return [*encoded, -1]
        return encoded

    def attach_embedding(self, item: int, embedding: int) -> None:
        self.attachments.append((item, embedding))

    def cache_embedding(self, item: int, embedding: int) -> None:
        self.cached.append((item, embedding))

    def _retry_batch(self, batch, exc):  # noqa: ANN001, ANN202
        if self.retry_hook_error:
            raise RuntimeError("retry policy failed")
        return self.retry

    def _on_batch_start(self, batch):  # noqa: ANN001, ANN202
        if self.start_hook_error:
            raise RuntimeError("start hook failed")


def test_successful_dispatch_attaches_and_caches() -> None:
    service = _Service()
    try:
        future = service._submit(3)

        assert future.result(timeout=2) == 6
        assert service.attachments == [(3, 6)]
        assert service.cached == [(3, 6)]
    finally:
        service.close()


def test_batch_failure_recovers_each_item() -> None:
    service = _Service(controlled_drain=True)
    service.fail_multi = True
    service.retry = True
    try:
        first = service._submit(1)
        second = service._submit(2)
        service.drain_gate.set()

        assert first.result(timeout=2) == 2
        assert second.result(timeout=2) == 4
    finally:
        service.close()


@pytest.mark.parametrize("split_mode", ["too_few", "too_many"])
def test_wrong_embedding_cardinality_fails_future(split_mode: str) -> None:
    service = _Service()
    service.split_mode = split_mode
    try:
        future = service._submit(1)

        with pytest.raises(RuntimeError, match="split_embeddings returned"):
            future.result(timeout=2)
        assert service.context_events == ["enter", "exit"]
    finally:
        service.close()


def test_statistics_hook_failure_does_not_kill_worker() -> None:
    service = _Service()
    service.start_hook_error = True
    try:
        assert service._submit(1).result(timeout=2) == 2
        service.start_hook_error = False
        assert service._submit(2).result(timeout=2) == 4
    finally:
        service.close()


def test_fatal_policy_failure_completes_current_future_and_rejects_submits() -> None:
    service = _Service(controlled_drain=True)
    service.fail_items.add(1)
    service.retry_hook_error = True
    first = service._submit(1)
    second = service._submit(2)
    service.drain_gate.set()

    with pytest.raises(RuntimeError, match="retry policy failed"):
        first.result(timeout=2)
    with pytest.raises(RuntimeError, match="retry policy failed"):
        second.result(timeout=2)
    with pytest.raises(RuntimeError, match="worker has failed"):
        service._submit(3)

    service.close()
