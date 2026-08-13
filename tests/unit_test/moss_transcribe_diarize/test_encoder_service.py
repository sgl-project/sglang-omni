# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import queue
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_transcribe_diarize import encoder_service
from sglang_omni.models.moss_transcribe_diarize.encoder_service import (
    BatchedAudioEncoderService,
)
from sglang_omni.scheduling.pre_lm_encoder import QueueEntry
from sglang_omni.scheduling.stage_cache import StageOutputCache


def test_drain_batch_respects_gpu_microbatch_limit() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._max_batch_size = 2
    service._queue = queue.Queue()
    entries = [
        QueueEntry(object(), concurrent.futures.Future())
        for _ in range(service._max_batch_size + 2)
    ]
    for entry in entries:
        service._queue.put(entry)

    assert service._drain_batch() == entries[:2]
    assert service._queue.qsize() == 2


def test_encoder_microbatch_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
        BatchedAudioEncoderService(object(), max_batch_size=0)


class _FailingStream:
    def synchronize(self) -> None:
        raise torch.OutOfMemoryError("test encoder OOM")


class _EncoderIntermediate:
    pass


class _StopWorker(BaseException):
    pass


def test_encode_batch_commits_item_state_only_after_stream_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._stream = _FailingStream()
    service._model = SimpleNamespace(
        _get_audio_feature_uncached=lambda items, forward_batch: torch.ones(2, 3)
    )
    monkeypatch.setattr(
        encoder_service.torch.cuda,
        "stream",
        lambda stream: contextlib.nullcontext(),
    )
    features = [torch.ones(1), torch.ones(1)]
    items = [
        SimpleNamespace(
            feature=feature,
            audio_feature_lengths=torch.tensor([1]),
            hash=None,
        )
        for feature in features
    ]

    with pytest.raises(torch.OutOfMemoryError, match="test encoder OOM"):
        service._execute_batch(items)

    for item, feature in zip(items, features):
        assert item.feature is feature
        assert not hasattr(item, "precomputed_embeddings")


def test_singleton_oom_is_request_scoped_and_worker_processes_next_item(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._max_batch_size = 1
    service._queue = queue.Queue()
    service._worker_state_lock = threading.Lock()
    service._worker_error = None
    service._batch_count = 0
    service._item_count = 0
    service._device = "cuda:7"
    cleanup_steps: list[str] = []
    selected_devices: list[str] = []
    calls: list[list[object]] = []
    retained_intermediates: list[weakref.ReferenceType[_EncoderIntermediate]] = []
    poisoned = False
    failed_item = SimpleNamespace(hash=None, feature=object())
    healthy_item = SimpleNamespace(hash=None, feature=object())
    stop_item = object()

    def _execute_batch(items: list[object]) -> list[object]:
        nonlocal poisoned
        if items == [stop_item]:
            raise _StopWorker
        calls.append(items)
        if items == [failed_item]:
            intermediate = _EncoderIntermediate()
            retained_intermediates.append(weakref.ref(intermediate))
            poisoned = True
            raise torch.OutOfMemoryError("test encoder OOM")
        if poisoned:
            raise RuntimeError("allocator remained poisoned after OOM")
        items[0].precomputed_embeddings = object()
        items[0].feature = None
        return [items[0].precomputed_embeddings]

    def _cuda_device(device: str) -> contextlib.AbstractContextManager:
        selected_devices.append(device)
        return contextlib.nullcontext()

    def _empty_cache() -> None:
        nonlocal poisoned
        cleanup_steps.append("empty_cache")
        poisoned = False

    service._stream = SimpleNamespace(
        synchronize=lambda: cleanup_steps.append("synchronize")
    )
    monkeypatch.setattr(service, "_execute_batch", _execute_batch)
    monkeypatch.setattr(encoder_service.torch.cuda, "device", _cuda_device)
    monkeypatch.setattr(encoder_service.torch.cuda, "empty_cache", _empty_cache)

    def _run_worker() -> None:
        try:
            service._worker()
        except _StopWorker:
            pass

    service._thread = threading.Thread(target=_run_worker, daemon=True)
    service._thread.start()

    try:
        with pytest.raises(torch.OutOfMemoryError, match="test encoder OOM"):
            service.encode_item(failed_item)
        service.encode_item(healthy_item)
    finally:
        service._queue.put(QueueEntry(stop_item, concurrent.futures.Future()))
        service._thread.join(timeout=1)

    gc.collect()
    assert not service._thread.is_alive()
    assert calls == [[failed_item], [healthy_item]]
    assert cleanup_steps == ["synchronize", "empty_cache"]
    assert selected_devices == ["cuda:7"]
    assert healthy_item.feature is None
    assert service._batch_count == 1
    assert service._item_count == 1
    assert retained_intermediates[0]() is None
    assert all(record.exc_info is None for record in caplog.records)
    assert all(
        not any(isinstance(arg, BaseException) for arg in record.args)
        for record in caplog.records
    )


def test_batched_oom_falls_back_to_per_item_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._batch_count = 0
    service._item_count = 0
    service._worker_state_lock = threading.Lock()
    service._worker_error = None
    service._device = "cuda:5"
    cleanup_steps: list[str] = []
    selected_devices: list[str] = []
    service._stream = SimpleNamespace(
        synchronize=lambda: cleanup_steps.append("synchronize")
    )
    poisoned = False
    calls: list[list[object]] = []
    items = [object(), object()]

    def _execute_batch(batch: list[object]) -> list[object]:
        nonlocal poisoned
        calls.append(batch)
        if len(batch) > 1:
            poisoned = True
            raise torch.OutOfMemoryError("aggregate batch is too large")
        if poisoned:
            raise RuntimeError("allocator remained poisoned after OOM")
        return [object()]

    def _cuda_device(device: str) -> contextlib.AbstractContextManager:
        selected_devices.append(device)
        return contextlib.nullcontext()

    def _empty_cache() -> None:
        nonlocal poisoned
        cleanup_steps.append("empty_cache")
        poisoned = False

    monkeypatch.setattr(service, "_execute_batch", _execute_batch)
    monkeypatch.setattr(encoder_service.torch.cuda, "device", _cuda_device)
    monkeypatch.setattr(encoder_service.torch.cuda, "empty_cache", _empty_cache)
    entries = [QueueEntry(item, concurrent.futures.Future()) for item in items]
    batches = iter([(entries, False), ([], True)])
    service._next_batch = lambda: next(batches)

    service._worker()

    assert [entry.future.result() for entry in entries] == [None, None]
    assert calls == [items, [items[0]], [items[1]]]
    assert service._batch_count == 2
    assert cleanup_steps == ["synchronize", "empty_cache"]
    assert selected_devices == ["cuda:5"]
    assert service._item_count == 2


def test_non_oom_failure_logs_traceback_without_retaining_exception_state(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._worker_state_lock = threading.Lock()
    service._worker_error = None
    retained_intermediates: list[weakref.ReferenceType[_EncoderIntermediate]] = []

    def _raise_non_oom_encoder_failure(_items: list[object]) -> list[object]:
        intermediate = _EncoderIntermediate()
        retained_intermediates.append(weakref.ref(intermediate))
        raise ValueError("unexpected encoder shape")

    monkeypatch.setattr(service, "_execute_batch", _raise_non_oom_encoder_failure)
    entry = QueueEntry(object(), concurrent.futures.Future())
    batches = iter([([entry], False), ([], True)])
    service._next_batch = lambda: next(batches)

    service._worker()

    failure = entry.future.exception()
    assert isinstance(failure, ValueError)
    assert failure.__traceback__ is None
    assert failure.__cause__ is None
    assert failure.__context__ is None

    gc.collect()
    assert retained_intermediates[0]() is None
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "Traceback (most recent call last):" in message
    assert "_raise_non_oom_encoder_failure" in message
    assert "ValueError: unexpected encoder shape" in message
    assert all(record.exc_info is None for record in caplog.records)
    assert all(
        not any(isinstance(arg, BaseException) for arg in record.args)
        for record in caplog.records
    )


def test_encode_item_rechecks_cache_after_preprocessing() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._device = torch.device("cpu")
    service._dtype = torch.float32
    service._hidden_size = 3
    service._cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    cached = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    service._cache.put("fingerprint", cached)
    item = SimpleNamespace(
        hash=7,
        audio_fingerprint="fingerprint",
        audio_feature_lengths=torch.tensor([2]),
        feature=object(),
        precomputed_embeddings=None,
    )
    service._submit = lambda item: pytest.fail("cached item must not be submitted")

    service.encode_item(item)

    assert torch.equal(item.precomputed_embeddings, cached)
    assert item.feature is None


@pytest.mark.parametrize(
    "cached",
    [
        torch.ones(3, 3),
        torch.ones(2, 4),
        torch.ones(2, 3, dtype=torch.float64),
    ],
)
def test_lookup_cached_embedding_evicts_invalid_entries(cached: torch.Tensor) -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._dtype = torch.float32
    service._hidden_size = 3
    service._cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    service._cache.put("fingerprint", cached)

    assert service.lookup_cached_embedding("fingerprint", 2) is None
    assert len(service._cache) == 0


def test_lookup_cached_embedding_returns_valid_entry() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._dtype = torch.float32
    service._hidden_size = 3
    service._cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    cached = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    service._cache.put("fingerprint", cached)

    result = service.lookup_cached_embedding("fingerprint", 2)

    assert result is not None
    assert torch.equal(result, cached)


def test_batch_failure_retries_moss_items_with_failure_isolation() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._batch_count = 0
    service._item_count = 0
    service._worker_state_lock = threading.Lock()
    service._worker_error = None
    service._device = torch.device("cpu")
    service._cache = StageOutputCache(max_size=4, max_bytes=1024, cache_device="cpu")
    synchronized: list[None] = []
    service._stream = SimpleNamespace(synchronize=lambda: synchronized.append(None))

    good = SimpleNamespace(
        hash=1,
        audio_feature_lengths=torch.tensor([2]),
        feature=object(),
        precomputed_embeddings=None,
        fail=False,
    )
    bad = SimpleNamespace(
        hash=2,
        audio_feature_lengths=torch.tensor([1]),
        feature=object(),
        precomputed_embeddings=None,
        fail=True,
    )

    def encode(items, _unused):  # noqa: ANN001, ANN202
        if len(items) > 1:
            raise RuntimeError("batch failed")
        if items[0].fail:
            raise RuntimeError("item failed")
        rows = int(items[0].audio_feature_lengths.sum())
        return torch.ones(rows, 3)

    service._model = SimpleNamespace(_get_audio_feature_uncached=encode)
    service._batch_context = contextlib.nullcontext
    good_entry = QueueEntry(good, concurrent.futures.Future())
    bad_entry = QueueEntry(bad, concurrent.futures.Future())
    batches = iter([([good_entry, bad_entry], False), ([], True)])
    service._next_batch = lambda: next(batches)

    service._worker()

    assert good_entry.future.result(timeout=0) is None
    with pytest.raises(RuntimeError, match="item failed"):
        bad_entry.future.result(timeout=0)
    assert good.precomputed_embeddings.shape == (2, 3)
    assert good.feature is None
    assert bad.precomputed_embeddings is None
    assert torch.equal(service._cache.get("1"), good.precomputed_embeddings)
    assert service._cache.get("2") is None
    assert len(synchronized) == 1
