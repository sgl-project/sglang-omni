# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_asr.encoder_service import (
    Qwen3ASRPreLMEncoderService,
    _expected_audio_tokens,
    build_cache_namespace,
)

_HIDDEN_SIZE = 4
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    _DEVICE = "xpu"
else:
    _DEVICE = "cpu"
requires_accelerator = pytest.mark.skipif(
    _DEVICE == "cpu",
    reason="requires cuda or xpu",
)
_NAMESPACE = "testns"
_SERVICES: list[Qwen3ASRPreLMEncoderService] = []


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
            thinker_config=SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE)
            )
        )
        self.dtype = dtype
        self.encode_calls = 0
        self.encode_batch_sizes: list[int] = []
        self.fail = False
        self.fail_oom = False
        self.fail_multi_item = False
        self.packed_3d_output = True
        self.encode_gate: threading.Event | None = None
        self.encode_started: threading.Event | None = None
        self.row_offset = 0
        self.encode_delay_s = 0.0
        self.grad_enabled_during_encode: bool | None = None

    def get_audio_feature(self, items):  # noqa: ANN001
        self.grad_enabled_during_encode = torch.is_grad_enabled()
        self.encode_calls += 1
        self.encode_batch_sizes.append(len(items))
        if self.encode_started is not None:
            self.encode_started.set()
        gate = self.encode_gate
        if gate is not None:
            self.encode_gate = None
            gate.wait(timeout=10)
        if self.encode_delay_s:
            time.sleep(self.encode_delay_s)
        if self.fail_oom:
            raise torch.OutOfMemoryError("encoder OOM")
        if self.fail:
            raise RuntimeError("boom")
        if self.fail_multi_item and len(items) > 1:
            raise RuntimeError("multi-item boom")
        parts = []
        for item in items:
            rows = _expected_audio_tokens(item) + self.row_offset
            fill = float((getattr(item, "hash", None) or 0) % 97 + 1)
            parts.append(torch.full((rows, _HIDDEN_SIZE), fill, dtype=self.dtype))
        packed = torch.cat(parts, dim=0)
        if self.packed_3d_output:
            # Mirrors the real audio tower: one packed frame stream in, so
            # last_hidden_state is [1, total_tokens, hidden].
            return packed.unsqueeze(0)
        return packed


def _make_service(
    model: _StubModel | None = None,
    *,
    cache_max_entries: int = 16,
    cache_max_bytes: int = 1 << 20,
    max_batch_size: int = 8,
) -> Qwen3ASRPreLMEncoderService:
    service = Qwen3ASRPreLMEncoderService(
        model or _StubModel(),
        cache_namespace=_NAMESPACE,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
        max_batch_size=max_batch_size,
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
        feature=torch.zeros(1, 128, 300) if with_feature else None,
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


def test_submit_returns_before_encoding_completes() -> None:
    model = _StubModel()
    gate = threading.Event()
    model.encode_gate = gate
    service = _make_service(model)
    item = _item(7, 3)

    future = service.submit_item(item)

    assert not future.done()
    assert item.precomputed_embeddings is None
    gate.set()
    future.result(timeout=2)
    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


def test_async_submissions_form_full_batch_without_blocked_callers() -> None:
    model = _StubModel()
    gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = gate
    model.encode_started = encode_started
    service = _make_service(model, max_batch_size=8)
    items = [_item(audio_hash, 3) for audio_hash in range(9)]

    futures = [service.submit_item(items[0])]
    assert encode_started.wait(timeout=2)
    assert model.encode_calls == 1
    futures.extend(service.submit_item(item) for item in items[1:])
    gate.set()
    for future in futures:
        future.result(timeout=2)

    assert model.encode_batch_sizes == [1, 8]


def test_async_single_flight_completes_each_item_future() -> None:
    model = _StubModel()
    gate = threading.Event()
    model.encode_gate = gate
    service = _make_service(model)
    items = [_item(123, 3) for _ in range(3)]

    futures = [service.submit_item(item) for item in items]
    assert all(not future.done() for future in futures)
    gate.set()
    for future in futures:
        future.result(timeout=2)

    assert model.encode_calls == 1
    assert service.stats()["merged"] == 2
    assert all(item.precomputed_embeddings is not None for item in items)


def test_close_stops_worker() -> None:
    service = _make_service()

    service.close()

    assert not service._thread.is_alive()


def test_batch_context_unwinds_inference_mode_when_stream_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(Qwen3ASRPreLMEncoderService)
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


def test_weight_update_context_clears_cached_embeddings() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(11, 3)
    service.encode_item(first)
    assert model.encode_calls == 1

    with service.weight_update_context():
        pass

    second = _item(11, 3)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert service.stats()["cache_entries"] == 1


def test_weight_update_context_clears_cache_after_update_failure() -> None:
    model = _StubModel()
    service = _make_service(model)
    service.encode_item(_item(12, 3))

    with pytest.raises(RuntimeError, match="update failed"):
        with service.weight_update_context():
            raise RuntimeError("update failed")

    service.encode_item(_item(12, 3))
    assert model.encode_calls == 2


def test_weight_update_context_waits_for_accepted_encode() -> None:
    model = _StubModel()
    encode_gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = encode_gate
    model.encode_started = encode_started
    service = _make_service(model)

    future = service.submit_item(_item(41, 3))
    assert encode_started.wait(timeout=2)

    update_started = threading.Event()

    def update() -> None:
        with service.weight_update_context():
            assert future.done()
            update_started.set()

    thread = threading.Thread(target=update)
    thread.start()
    assert not update_started.wait(timeout=0.05)

    encode_gate.set()
    assert update_started.wait(timeout=2)
    thread.join(timeout=2)

    assert not thread.is_alive()
    future.result(timeout=2)


def test_weight_update_holds_queued_batch_for_new_weights() -> None:
    model = _StubModel()
    encode_gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = encode_gate
    model.encode_started = encode_started
    service = _make_service(model, max_batch_size=1)

    first = service.submit_item(_item(41, 3))
    assert encode_started.wait(timeout=2)
    second = service.submit_item(_item(42, 3))

    update_started = threading.Event()
    release_update = threading.Event()

    def update() -> None:
        with service.weight_update_context():
            update_started.set()
            assert release_update.wait(timeout=2)

    thread = threading.Thread(target=update)
    thread.start()
    encode_gate.set()
    assert update_started.wait(timeout=2)

    first.result(timeout=2)
    assert not second.done()
    assert model.encode_calls == 1

    release_update.set()
    second.result(timeout=2)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert model.encode_calls == 2


def test_cancelled_future_does_not_finish_encoder_work_early() -> None:
    model = _StubModel()
    encode_gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = encode_gate
    model.encode_started = encode_started
    service = _make_service(model)

    future = service.submit_item(_item(43, 3))
    assert encode_started.wait(timeout=2)
    assert future.cancel()

    update_started = threading.Event()

    def update() -> None:
        with service.weight_update_context():
            update_started.set()

    thread = threading.Thread(target=update)
    thread.start()
    assert not update_started.wait(timeout=0.05)

    encode_gate.set()
    assert update_started.wait(timeout=2)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert model.encode_calls == 1


def test_weight_update_timeout_restores_submissions() -> None:
    model = _StubModel()
    encode_gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = encode_gate
    model.encode_started = encode_started
    service = _make_service(model)
    service.WEIGHT_UPDATE_DRAIN_TIMEOUT_S = 0.01

    future = service.submit_item(_item(44, 3))
    assert encode_started.wait(timeout=2)

    with pytest.raises(TimeoutError, match="active Qwen3-ASR encoder work"):
        with service.weight_update_context():
            pytest.fail("weight update started before encoder work drained")

    encode_gate.set()
    future.result(timeout=2)
    service.submit_item(_item(45, 3)).result(timeout=2)


def test_omni_scheduler_admin_update_waits_for_real_inflight_encode() -> None:
    """Integration check: OmniScheduler._admin_update_weights_from_disk must
    not call the model_worker's update_weights_from_disk until a REAL
    Qwen3ASRPreLMEncoderService confirms no encode is in flight.

    test_weight_update_context_waits_for_accepted_encode (above) already
    proves weight_update_context() blocks in isolation. This proves the
    scheduler's admin lifecycle actually composes with that real gate
    end-to-end -- the other admin-control tests only stub
    _weight_update_context with a bare contextlib.nullcontext, which cannot
    catch a regression where the scheduler stopped calling it, called it too
    late, or called it against the wrong encoder instance.
    """
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    model = _StubModel()
    encode_gate = threading.Event()
    encode_started = threading.Event()
    model.encode_gate = encode_gate
    model.encode_started = encode_started
    service = _make_service(model)

    # Simulate a request admitted moments before the weight update arrives:
    # its encode is already running on the encoder's own worker thread.
    future = service.submit_item(_item(91, 3))
    assert encode_started.wait(timeout=2)

    update_fn_called = threading.Event()

    def update_weights_from_disk(payload: dict) -> tuple[bool, str]:
        del payload
        # If this runs before the in-flight encode resolves, the weight copy
        # could land mid-forward-pass through audio_tower.
        assert future.done(), "update_fn ran while an encode was still in flight"
        update_fn_called.set()
        return True, "ok"

    scheduler = object.__new__(OmniScheduler)
    scheduler.model_worker = SimpleNamespace(
        update_weights_from_disk=update_weights_from_disk
    )
    scheduler._admin_lock = threading.Lock()
    scheduler._engine_paused = False
    scheduler._last_pause_mode = None
    scheduler._async_pending = None
    scheduler._resolve_pending_async = lambda: None
    scheduler._active_request_ids = lambda: []
    scheduler._empty_torch_cache = lambda: None
    scheduler.flush_cache = lambda: True
    scheduler._weight_update_context = service.weight_update_context

    result_box: dict[str, dict] = {}

    def run_admin() -> None:
        result_box["result"] = OmniScheduler._admin_update_weights_from_disk(
            scheduler, {"model_path": "/tmp/new-model"}
        )

    admin_thread = threading.Thread(target=run_admin)
    admin_thread.start()

    # The admin call must genuinely block here: the gate is still held, so
    # update_fn must not have run yet.
    assert not update_fn_called.wait(timeout=0.2)
    assert not future.done()

    encode_gate.set()  # let the in-flight encode finish
    future.result(timeout=2)

    assert update_fn_called.wait(timeout=2)
    admin_thread.join(timeout=2)
    assert not admin_thread.is_alive()
    assert result_box["result"]["success"] is True

    # weight_update_context()'s own finally clears the cache: a later
    # request for the same audio must miss and re-encode, not silently reuse
    # the embedding computed before the update.
    service.encode_item(_item(91, 3))
    assert model.encode_calls == 2


def test_lookup_cached_embedding_returns_only_valid_entries() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _item(11, 3)
    service.encode_item(item)

    cached = service.lookup_cached_embedding(item.audio_fingerprint, 3)

    assert cached is not None
    assert torch.equal(cached, item.precomputed_embeddings.cpu())
    assert service.stats()["hits"] == 1

    assert service.lookup_cached_embedding(item.audio_fingerprint, 4) is None
    assert len(service._cache) == 0


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
    errors: list[Exception] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except Exception as exc:
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
    errors: list[Exception] = []

    def follower() -> None:
        try:
            service.encode_item(follower_item)
        except Exception as exc:
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
    errors: list[Exception] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except Exception as exc:
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


def test_submit_item_failure_counts_failed() -> None:
    model = _StubModel()
    model.fail = True
    service = _make_service(model)

    future = service.submit_item(_item(88, 3))

    with pytest.raises(RuntimeError, match="boom"):
        future.result(timeout=2)
    assert service.stats()["failed"] == 1


def test_encode_failure_propagates_without_poisoning_cache() -> None:
    model = _StubModel()
    model.fail = True
    model.encode_delay_s = 0.05
    service = _make_service(model)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(_item(55, 3))
        except Exception as exc:
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


def test_oom_failure_detaches_traceback_and_recovers() -> None:
    model = _StubModel()
    model.fail_oom = True
    service = _make_service(model)

    with pytest.raises(torch.OutOfMemoryError, match="encoder OOM") as excinfo:
        service.encode_item(_item(77, 3))

    # note (luojiaxuan): the propagated exception must not pin encoder frames.
    assert excinfo.value.__traceback__ is not None  # pytest's own raise site
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert service.stats()["failed"] == 1

    model.fail_oom = False
    item = _item(77, 3)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


def test_merged_follower_token_mismatch_raises_and_counts_failed() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.2
    service = _make_service(model)
    leader_item = _item(321, 3)
    follower_item = _item(321, 5)
    errors: list[Exception] = []

    def leader() -> None:
        try:
            service.encode_item(leader_item)
        except Exception as exc:
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
    errors: list[Exception] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            service.encode_item(item)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    # note (luojiaxuan): queue every leader before releasing the gate so the
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
    explicit = SimpleNamespace(num_audio_tokens=5, feature=torch.zeros(1, 128, 300))
    assert _expected_audio_tokens(explicit) == 5
    assert _expected_audio_tokens(SimpleNamespace()) is None


def test_build_cache_namespace_is_stable_and_scoped() -> None:
    model = _StubModel()
    frontend = SimpleNamespace(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        nb_max_frames=3000,
        padding_value=0.0,
    )
    base = dict(
        model_path="Qwen/Qwen3-ASR-1.7B",
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
    changed_frontend = SimpleNamespace(**{**vars(frontend), "hop_length": 320})
    assert namespace != build_cache_namespace(
        model, **{**base, "feature_extractor": changed_frontend}
    )
    changed_config = _StubModel()
    changed_config.config = SimpleNamespace(
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE)
        ),
        marker="other",
    )
    assert namespace != build_cache_namespace(changed_config, **base)


def test_flat_2d_encoder_output_is_also_accepted() -> None:
    model = _StubModel()
    model.packed_3d_output = False
    service = _make_service(model)
    item = _item(8, 3)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


@pytest.mark.accelerator
@requires_accelerator
def test_the_device_cache_is_really_reclaimed_after_an_oom() -> None:
    """The behaviour the fix exists for, on the live accelerator."""
    device_module = torch.get_device_module(torch.device(_DEVICE))
    service = _make_service(_StubModel().to(_DEVICE))

    with device_module.device(_DEVICE):
        device_module.synchronize()
        device_module.empty_cache()
        floor = device_module.memory_reserved()
        allocated = device_module.memory_allocated()
        cached_slack = max(0, floor - allocated)
        block = torch.empty(
            cached_slack + 64 * 1024 * 1024,
            dtype=torch.uint8,
            device=_DEVICE,
        )
        device_module.synchronize()
        reserved_with_block = device_module.memory_reserved()
        assert reserved_with_block > floor

        del block
        device_module.synchronize()
        reserved_before = device_module.memory_reserved()
        assert reserved_before > floor

        service._recover_after_failure(torch.OutOfMemoryError("encoder OOM"))

        device_module.synchronize()
        assert device_module.memory_reserved() < reserved_before
