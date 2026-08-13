# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import gc
import threading
import time
import weakref
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_asr import encoder_service
from sglang_omni.models.fun_asr.encoder_service import (
    FunASRPreLMEncoderService,
    _expected_audio_tokens,
    build_cache_namespace,
)

_HIDDEN_SIZE = 4
_NAMESPACE = "testns"
_SERVICES: list[FunASRPreLMEncoderService] = []


@pytest.fixture(autouse=True)
def _close_services() -> Iterator[None]:
    yield
    for service in _SERVICES:
        service.close()
    _SERVICES.clear()


class _StubModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.audio_tower = torch.nn.Linear(2, 2).to(dtype)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE)
        )
        self.dtype = dtype
        self.encode_calls = 0
        self.fail = False
        self.fail_oom_once = False
        self.fail_multi_item = False
        self.encode_gate: threading.Event | None = None
        self.row_offset = 0
        self.encode_delay_s = 0.0
        self.grad_enabled_during_encode: bool | None = None

    def get_audio_feature(self, items):  # noqa: ANN001
        self.grad_enabled_during_encode = torch.is_grad_enabled()
        self.encode_calls += 1
        gate = self.encode_gate
        if gate is not None:
            self.encode_gate = None
            gate.wait(timeout=10)
        if self.encode_delay_s:
            time.sleep(self.encode_delay_s)
        if self.fail:
            raise RuntimeError("boom")
        if self.fail_oom_once:
            self.fail_oom_once = False
            raise torch.OutOfMemoryError("test encoder OOM")
        if self.fail_multi_item and len(items) > 1:
            raise RuntimeError("multi-item boom")
        parts = []
        for item in items:
            rows = _expected_audio_tokens(item) + self.row_offset
            fill = float((getattr(item, "hash", None) or 0) % 97 + 1)
            parts.append(torch.full((rows, _HIDDEN_SIZE), fill, dtype=self.dtype))
        return torch.cat(parts, dim=0)


def _make_service(
    model: _StubModel | None = None,
    *,
    cache_max_entries: int = 16,
    cache_max_bytes: int = 1 << 20,
) -> FunASRPreLMEncoderService:
    service = FunASRPreLMEncoderService(
        model or _StubModel(),
        cache_namespace=_NAMESPACE,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
    )
    _SERVICES.append(service)
    return service


def _item(
    audio_hash: int | None,
    num_audio_tokens: int,
    *,
    with_feature: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        hash=audio_hash,
        audio_fingerprint=str(audio_hash) if audio_hash is not None else None,
        num_audio_tokens=num_audio_tokens,
        feature=torch.zeros(1, 560, 8) if with_feature else None,
        precomputed_embeddings=None,
    )


def test_encode_attaches_lm_ready_embedding_and_clears_feature() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _item(7, 3)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
    assert item.precomputed_embeddings.dtype == model.dtype
    assert (
        item.precomputed_embeddings.device
        == next(model.audio_tower.parameters()).device
    )
    assert item.feature is None
    assert item.format.name == "PRECOMPUTED_EMBEDDING"
    assert model.encode_calls == 1
    assert model.grad_enabled_during_encode is False
    assert service.stats()["misses"] == 1


def test_close_stops_worker() -> None:
    service = _make_service()

    service.close()

    assert not service._thread.is_alive()


def test_batch_context_unwinds_inference_mode_when_stream_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(FunASRPreLMEncoderService)
    service._stream = object()

    def fail_stream(_stream):  # noqa: ANN001, ANN202
        raise RuntimeError("stream context failed")

    monkeypatch.setattr(torch.cuda, "stream", fail_stream)

    assert not torch.is_inference_mode_enabled()
    with pytest.raises(RuntimeError, match="stream context failed"):
        with service._batch_context():
            pass
    assert not torch.is_inference_mode_enabled()


def test_cache_hit_skips_reencode() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(11, 3)
    second = _item(11, 3)
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 1
    assert torch.equal(first.precomputed_embeddings, second.precomputed_embeddings)
    assert second.feature is None
    assert service.stats()["hits"] == 1


def test_extended_audio_never_reuses_prefix_embedding() -> None:
    model = _StubModel()
    service = _make_service(model)

    short = _item(111, 3)
    extended = _item(222, 5)
    service.encode_item(short)
    service.encode_item(extended)

    assert model.encode_calls == 2
    assert extended.precomputed_embeddings.shape == (5, _HIDDEN_SIZE)
    assert len(service._cache) == 2
    assert not torch.equal(
        short.precomputed_embeddings[0], extended.precomputed_embeddings[0]
    )


def test_cache_key_prefers_full_waveform_fingerprint() -> None:
    model = _StubModel()
    service = _make_service(model)
    first = _item(7, 3)
    second = _item(7, 3)
    first.audio_fingerprint = "full-hash-a"
    second.audio_fingerprint = "full-hash-b"

    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert len(service._cache) == 2


def test_concurrent_identical_requests_encode_once() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.05
    service = _make_service(model)
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    items = [_item(123, 3) for _ in range(n_threads)]
    errors: list[BaseException] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert model.encode_calls == 1
    for item in items:
        assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
        assert torch.equal(item.precomputed_embeddings, items[0].precomputed_embeddings)
    stats = service.stats()
    assert stats["merged"] + stats["hits"] == n_threads - 1


def test_stale_cache_miss_rechecks_before_starting_duplicate_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel()
    service = _make_service(model)
    stale_miss = threading.Event()
    release_stale_reader = threading.Event()
    original_get = service._cache.get

    def controlled_get(key: str | None):  # noqa: ANN202
        cached = original_get(key)
        if (
            threading.current_thread().name == "stale-cache-reader"
            and not stale_miss.is_set()
        ):
            assert cached is None
            stale_miss.set()
            assert release_stale_reader.wait(timeout=10)
        return cached

    monkeypatch.setattr(service._cache, "get", controlled_get)
    follower_item = _item(123, 3)
    errors: list[BaseException] = []

    def follower() -> None:
        try:
            service.encode_item(follower_item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=follower, name="stale-cache-reader")
    thread.start()
    assert stale_miss.wait(timeout=10)

    leader_item = _item(123, 3)
    service.encode_item(leader_item)
    release_stale_reader.set()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert not errors, errors
    assert model.encode_calls == 1
    assert torch.equal(
        leader_item.precomputed_embeddings,
        follower_item.precomputed_embeddings,
    )
    assert service.stats()["hits"] == 1


def test_concurrent_identical_requests_deduplicate_without_cache() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.05
    service = _make_service(model, cache_max_entries=0)
    barrier = threading.Barrier(2)
    items = [_item(123, 3) for _ in range(2)]
    errors: list[BaseException] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert model.encode_calls == 1
    assert len(service._cache) == 0
    assert torch.equal(items[0].precomputed_embeddings, items[1].precomputed_embeddings)


def test_encode_failure_propagates_without_poisoning_cache() -> None:
    model = _StubModel()
    model.fail = True
    model.encode_delay_s = 0.05
    service = _make_service(model)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(_item(55, 3))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(errors) == 2
    assert all(isinstance(exc, RuntimeError) and "boom" in str(exc) for exc in errors)
    assert len(service._cache) == 0
    assert service.stats()["failed"] == 2

    model.fail = False
    item = _item(55, 3)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


class _FailingStream:
    def synchronize(self) -> None:
        raise torch.OutOfMemoryError("test encoder sync OOM")


class _EncoderIntermediate:
    pass


def test_execute_batch_commits_item_state_only_after_stream_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(FunASRPreLMEncoderService)
    service._model = _StubModel()
    service._stream = _FailingStream()
    service._hidden_size = _HIDDEN_SIZE
    service._dtype = torch.float32
    service._device = torch.device("cpu")
    monkeypatch.setattr(
        encoder_service.torch.cuda,
        "stream",
        lambda stream: contextlib.nullcontext(),
    )
    items = [_item(701, 2), _item(702, 3)]
    features = [item.feature for item in items]

    with pytest.raises(torch.OutOfMemoryError, match="test encoder sync OOM"):
        service._execute_batch(items)

    for item, feature in zip(items, features):
        assert item.feature is feature
        assert item.precomputed_embeddings is None
        assert not hasattr(item, "format")


def test_singleton_oom_is_not_retried_and_next_request_succeeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _StubModel()
    model.fail_oom_once = True
    service = _make_service(model)
    failed_item = _item(801, 3)
    healthy_item = _item(802, 3)

    with pytest.raises(torch.OutOfMemoryError, match="test encoder OOM"):
        service.encode_item(failed_item)
    service.encode_item(healthy_item)

    assert model.encode_calls == 2
    assert failed_item.precomputed_embeddings is None
    assert failed_item.feature is not None
    assert healthy_item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
    assert healthy_item.feature is None
    assert all(record.exc_info is None for record in caplog.records)
    assert all(
        not any(isinstance(arg, BaseException) for arg in record.args)
        for record in caplog.records
    )


def test_oom_recovery_synchronizes_and_clears_selected_encoder_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(FunASRPreLMEncoderService)
    service._device = torch.device("cuda:7")
    cleanup_steps: list[str] = []
    selected_devices: list[torch.device] = []
    service._stream = SimpleNamespace(
        synchronize=lambda: cleanup_steps.append("synchronize")
    )

    def cuda_device(device: torch.device):  # noqa: ANN202
        selected_devices.append(device)
        return contextlib.nullcontext()

    monkeypatch.setattr(encoder_service.torch.cuda, "device", cuda_device)
    monkeypatch.setattr(
        encoder_service.torch.cuda,
        "empty_cache",
        lambda: cleanup_steps.append("empty_cache"),
    )

    service._recover_after_failure(torch.OutOfMemoryError("test encoder OOM"))

    assert cleanup_steps == ["synchronize", "empty_cache"]
    assert selected_devices == [torch.device("cuda:7")]


def test_non_oom_failure_is_detached_before_future_and_logging(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = _make_service()
    retained_intermediates: list[weakref.ReferenceType[_EncoderIntermediate]] = []

    def raise_non_oom(_items: list[object]) -> torch.Tensor:
        intermediate = _EncoderIntermediate()
        retained_intermediates.append(weakref.ref(intermediate))
        raise ValueError("unexpected encoder shape")

    monkeypatch.setattr(service, "encode_batch", raise_non_oom)
    future = service._submit(object())

    failure = future.exception(timeout=2)
    assert isinstance(failure, ValueError)
    assert failure.__traceback__ is None
    assert failure.__cause__ is None
    assert failure.__context__ is None
    gc.collect()
    assert retained_intermediates[0]() is None
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "Traceback (most recent call last):" in message
    assert "raise_non_oom" in message
    assert "ValueError: unexpected encoder shape" in message
    assert all(record.exc_info is None for record in caplog.records)
    assert all(
        not any(isinstance(arg, BaseException) for arg in record.args)
        for record in caplog.records
    )


@pytest.mark.parametrize("error_type", [ValueError, torch.OutOfMemoryError])
def test_failure_arguments_do_not_retain_tensors_in_future(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    service = _make_service()
    retained_tensors: list[weakref.ReferenceType[torch.Tensor]] = []

    def raise_with_tensor(_items: list[object]) -> torch.Tensor:
        failed_tensor = torch.ones(1)
        retained_tensors.append(weakref.ref(failed_tensor))
        raise error_type("failed tensor", failed_tensor)

    monkeypatch.setattr(service, "encode_batch", raise_with_tensor)
    future = service._submit(object())

    failure = future.exception(timeout=2)
    assert isinstance(failure, error_type)
    assert all(not isinstance(arg, torch.Tensor) for arg in failure.args)
    gc.collect()
    assert retained_tensors[0]() is None


def test_batched_oom_recovers_before_per_item_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _make_service()
    service._max_batch_wait_s = 1.0
    calls: list[list[SimpleNamespace]] = []
    cleanup_steps: list[str] = []
    poisoned = False
    items = [_item(None, 1), _item(None, 1)]

    def encode_batch(batch: list[SimpleNamespace]) -> torch.Tensor:
        nonlocal poisoned
        calls.append(list(batch))
        if len(batch) > 1:
            poisoned = True
            raise torch.OutOfMemoryError("aggregate batch is too large")
        if poisoned:
            raise RuntimeError("allocator remained poisoned after OOM")
        return torch.ones(1, _HIDDEN_SIZE)

    def recover(exc: Exception) -> None:
        nonlocal poisoned
        assert isinstance(exc, torch.OutOfMemoryError)
        cleanup_steps.append("recover")
        poisoned = False

    monkeypatch.setattr(service, "encode_batch", encode_batch)
    monkeypatch.setattr(service, "_recover_after_failure", recover)
    futures = [service._submit(item) for item in items]
    results = [future.result(timeout=5) for future in futures]

    assert all(torch.equal(result, torch.ones(1, _HIDDEN_SIZE)) for result in results)
    assert [len(batch) for batch in calls] == [2, 1, 1]
    assert calls[0][0] is items[0]
    assert calls[0][1] is items[1]
    assert calls[1][0] is items[0]
    assert calls[2][0] is items[1]
    assert cleanup_steps == ["recover"]
    assert service.stats()["batches"] == 2
    assert service.stats()["items"] == 2


def test_merged_follower_token_mismatch_raises_and_counts_failed() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.2
    service = _make_service(model)
    leader_item = _item(321, 3)
    follower_item = _item(321, 5)
    errors: list[BaseException] = []

    def leader() -> None:
        try:
            service.encode_item(leader_item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=leader)
    thread.start()
    deadline = time.monotonic() + 5
    while not service._inflight and time.monotonic() < deadline:
        time.sleep(0.005)
    assert service._inflight, "leader never registered in-flight"

    with pytest.raises(RuntimeError, match="returned an invalid"):
        service.encode_item(follower_item)
    thread.join(timeout=30)

    assert not errors, errors
    assert leader_item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
    assert follower_item.precomputed_embeddings is None
    stats = service.stats()
    assert stats["merged"] == 1
    assert stats["failed"] == 1


def test_multi_item_batch_failure_retries_per_item_and_counts_stats() -> None:
    model = _StubModel()
    model.fail_multi_item = True
    gate = threading.Event()
    model.encode_gate = gate
    service = _make_service(model)
    items = [_item(31, 3), _item(32, 3), _item(33, 4)]
    errors: list[BaseException] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    # Note (Akazaakane): Queue every leader before releasing the gate so the
    # next drain exercises the multi-item retry path.
    deadline = time.monotonic() + 5
    while len(service._inflight) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(service._inflight) == 3, "items never queued"
    gate.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    for item in items:
        assert item.precomputed_embeddings.shape == (
            item.num_audio_tokens,
            _HIDDEN_SIZE,
        )
    stats = service.stats()
    assert stats["failed"] == 0
    assert stats["items"] == 3
    assert stats["batches"] == 3
    assert model.encode_calls == 4
    assert len(service._cache) == 3


def test_eviction_under_byte_budget_triggers_reencode() -> None:
    model = _StubModel()
    service = _make_service(model, cache_max_bytes=100)

    for audio_hash in (1, 2, 3):
        service.encode_item(_item(audio_hash, 3))
    assert model.encode_calls == 3
    assert service._cache.eviction_count >= 1
    assert len(service._cache) == 2

    service.encode_item(_item(1, 3))
    assert model.encode_calls == 4


def test_invalid_cache_entry_is_evicted_and_reencoded() -> None:
    model = _StubModel()
    service = _make_service(model)
    probe = _item(42, 3)
    service.encode_item(probe)
    assert model.encode_calls == 1
    key = service._cache_key(probe)

    for poison in (
        torch.zeros(5, _HIDDEN_SIZE),
        torch.zeros(3, _HIDDEN_SIZE + 1),
        torch.zeros(3, _HIDDEN_SIZE, dtype=torch.float64),
    ):
        service._cache.put(key, poison)
        item = _item(42, 3)
        service.encode_item(item)
        assert model.encode_calls == 2
        assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
        assert torch.equal(item.precomputed_embeddings, probe.precomputed_embeddings)
        model.encode_calls = 1


def test_invalid_cache_reader_preserves_a_valid_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _item(42, 3)
    key = service._cache_key(item)
    service._cache.put(key, torch.zeros(2, _HIDDEN_SIZE))
    stale_reader = threading.Event()
    release_reader = threading.Event()
    original_remove = service._cache.remove_if_same

    def controlled_remove(key, expected):  # noqa: ANN001, ANN202
        stale_reader.set()
        assert release_reader.wait(timeout=10)
        return original_remove(key, expected)

    monkeypatch.setattr(service._cache, "remove_if_same", controlled_remove)
    errors: list[BaseException] = []

    def encode() -> None:
        try:
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=encode)
    thread.start()
    assert stale_reader.wait(timeout=10)
    replacement = torch.full((3, _HIDDEN_SIZE), 7.0)
    service._cache.put(key, replacement)
    release_reader.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert not errors, errors
    assert model.encode_calls == 0
    assert torch.equal(item.precomputed_embeddings, replacement)
    assert torch.equal(service._cache.get(key), replacement)


def test_token_count_mismatch_fails_loudly() -> None:
    model = _StubModel()
    model.row_offset = 1
    service = _make_service(model)
    item = _item(9, 3)

    with pytest.raises(RuntimeError, match="!= expected rows"):
        service.encode_item(item)

    assert item.precomputed_embeddings is None
    assert len(service._cache) == 0


def test_missing_token_count_raises() -> None:
    service = _make_service()
    item = SimpleNamespace(hash=1, feature=None, precomputed_embeddings=None)

    with pytest.raises(RuntimeError, match="num_audio_tokens"):
        service.encode_item(item)


def test_item_without_fingerprint_encodes_without_caching() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(1, 2)
    second = _item(1, 2)
    first.audio_fingerprint = None
    second.audio_fingerprint = None
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert first.feature is None
    assert first.precomputed_embeddings.shape == (2, _HIDDEN_SIZE)
    assert len(service._cache) == 0


def test_expected_audio_tokens_uses_request_metadata() -> None:
    explicit = SimpleNamespace(num_audio_tokens=5, feature=torch.zeros(1, 560, 17))
    assert _expected_audio_tokens(explicit) == 5
    assert _expected_audio_tokens(SimpleNamespace()) is None


def test_build_cache_namespace_is_stable_and_scoped() -> None:
    model = _StubModel()
    frontend = SimpleNamespace(
        feature_size=80,
        sampling_rate=16000,
        frame_length=25,
        frame_shift=10,
        lfr_m=7,
        lfr_n=6,
        window="hamming",
    )
    base = dict(
        model_path="FunAudioLLM/Fun-ASR-Nano-2512-hf",
        feature_extractor=frontend,
        mm_attention_backend=None,
    )

    namespace = build_cache_namespace(model, **base)
    assert namespace == build_cache_namespace(model, **base)
    assert namespace != build_cache_namespace(
        model, **{**base, "model_path": "other/revision"}
    )
    assert namespace != build_cache_namespace(
        model, **{**base, "mm_attention_backend": "triton_attn"}
    )
    assert namespace != build_cache_namespace(_StubModel(dtype=torch.bfloat16), **base)
    changed_frontend = SimpleNamespace(**{**vars(frontend), "lfr_m": 5})
    assert namespace != build_cache_namespace(
        model, **{**base, "feature_extractor": changed_frontend}
    )
    changed_config = _StubModel()
    changed_config.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE), marker="other"
    )
    assert namespace != build_cache_namespace(changed_config, **base)
