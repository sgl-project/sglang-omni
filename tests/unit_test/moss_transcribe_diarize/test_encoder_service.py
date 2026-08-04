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


def test_drain_batch_respects_gpu_microbatch_limit() -> None:
    service = object.__new__(BatchedAudioEncoderService)
    service._max_batch_size = 2
    service._queue = queue.Queue()
    entries = [
        (object(), concurrent.futures.Future())
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
        service._encode_batch(items)

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

    def _encode_batch(items: list[object]) -> None:
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
    monkeypatch.setattr(service, "_encode_batch", _encode_batch)
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
        service._queue.put((stop_item, concurrent.futures.Future()))
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
    service._device = "cuda:5"
    cleanup_steps: list[str] = []
    selected_devices: list[str] = []
    service._stream = SimpleNamespace(
        synchronize=lambda: cleanup_steps.append("synchronize")
    )
    poisoned = False
    calls: list[list[object]] = []
    items = [object(), object()]

    def _encode_batch(batch: list[object]) -> None:
        nonlocal poisoned
        calls.append(batch)
        if len(batch) > 1:
            poisoned = True
            raise torch.OutOfMemoryError("aggregate batch is too large")
        if poisoned:
            raise RuntimeError("allocator remained poisoned after OOM")

    def _cuda_device(device: str) -> contextlib.AbstractContextManager:
        selected_devices.append(device)
        return contextlib.nullcontext()

    def _empty_cache() -> None:
        nonlocal poisoned
        cleanup_steps.append("empty_cache")
        poisoned = False

    monkeypatch.setattr(service, "_encode_batch", _encode_batch)
    monkeypatch.setattr(encoder_service.torch.cuda, "device", _cuda_device)
    monkeypatch.setattr(encoder_service.torch.cuda, "empty_cache", _empty_cache)
    futures = [concurrent.futures.Future(), concurrent.futures.Future()]

    service._process_batch(list(zip(items, futures)))

    assert [future.result() for future in futures] == [None, None]
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
    retained_intermediates: list[weakref.ReferenceType[_EncoderIntermediate]] = []

    def _raise_non_oom_encoder_failure(_items: list[object]) -> None:
        intermediate = _EncoderIntermediate()
        retained_intermediates.append(weakref.ref(intermediate))
        raise ValueError("unexpected encoder shape")

    monkeypatch.setattr(service, "_encode_batch", _raise_non_oom_encoder_failure)
    future: concurrent.futures.Future = concurrent.futures.Future()

    service._process_batch([(object(), future)])

    failure = future.exception()
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
