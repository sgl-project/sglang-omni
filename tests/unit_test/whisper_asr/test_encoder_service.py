# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalInputFormat

from sglang_omni.models.whisper_asr.encoder_service import (
    WhisperPreLMEncoderService,
    _expected_audio_tokens,
    build_cache_namespace,
)

_HIDDEN_SIZE = 8
_TOKENS = 4
_NAMESPACE = "whisper-test"
_SERVICES: list[WhisperPreLMEncoderService] = []


@pytest.fixture(autouse=True)
def _close_services() -> Iterator[None]:
    yield
    for service in _SERVICES:
        service.close()
    _SERVICES.clear()


class _StubEncoder(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2).to(dtype)
        self.encode_calls = 0
        self.fail = False
        self.fail_multi_item = False
        self.encode_gate: threading.Event | None = None

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        gate = self.encode_gate
        if gate is not None:
            self.encode_gate = None
            gate.wait(timeout=10)
        if self.fail:
            raise RuntimeError("boom")
        if self.fail_multi_item and audio_features.shape[0] > 1:
            raise RuntimeError("multi-item boom")
        batch = audio_features.shape[0]
        fill = float(audio_features.reshape(batch, -1)[:, 0].mean().item() + 1.0)
        return torch.full(
            (batch, _TOKENS, _HIDDEN_SIZE),
            fill,
            dtype=audio_features.dtype,
            device=audio_features.device,
        )


class _StubModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        encoder = _StubEncoder(dtype=dtype)
        self.model = SimpleNamespace(encoder=encoder)
        self.config = SimpleNamespace(d_model=_HIDDEN_SIZE)
        self._encoder = encoder

    @property
    def encode_calls(self) -> int:
        return self._encoder.encode_calls

    def encode_audio_features(self, items: list[object]) -> torch.Tensor:
        features: list[torch.Tensor] = []
        reference = next(self._encoder.parameters())
        for item in items:
            feature = getattr(item, "feature", None)
            if feature is None:
                raise RuntimeError("missing mel features")
            if not isinstance(feature, torch.Tensor):
                feature = torch.as_tensor(feature)
            features.append(feature.to(device=reference.device, dtype=reference.dtype))
        return self._encoder(torch.cat(features, dim=0))


def _make_item(*, fingerprint: str = "fp", fill: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        feature=torch.full((1, 80, 16), fill),
        precomputed_embeddings=None,
        format=MultimodalInputFormat.NORMAL,
        model_specific_data={
            "audio_fingerprint": fingerprint,
            "num_audio_tokens": _TOKENS,
        },
        audio_fingerprint=fingerprint,
        num_audio_tokens=_TOKENS,
    )


def _make_service(
    model: _StubModel | None = None,
    *,
    cache_max_entries: int = 16,
    cache_max_bytes: int | None = None,
    max_batch_size: int = 8,
    pin_host_memory: bool = True,
) -> WhisperPreLMEncoderService:
    service = WhisperPreLMEncoderService(
        model or _StubModel(),
        cache_namespace=_NAMESPACE,
        encoder_token_count=_TOKENS,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
        max_batch_size=max_batch_size,
        pin_host_memory=pin_host_memory,
    )
    _SERVICES.append(service)
    return service


_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinned host memory needs a CUDA context"
)


def _cuda_model() -> _StubModel:
    return _StubModel(dtype=torch.float16).to("cuda")


def _cached_entry(
    service: WhisperPreLMEncoderService, fingerprint: str
) -> torch.Tensor:
    cached = service.lookup_cached_embedding(fingerprint, _TOKENS)
    assert cached is not None
    return cached


def test_pinning_is_off_without_cuda_device() -> None:
    # note (Jeffro): the stub model lives on CPU: there is nothing to DMA against, so the
    # service must not try to page-lock anything.
    service = _make_service()
    assert service.pin_host_memory is False
    assert service._cache.pin_memory is False
    assert service.stats()["pin_prewarm_s"] == 0.0


@pytest.mark.accelerator
@_requires_cuda
def test_cuda_cache_entries_are_pinned_and_match_device_result() -> None:
    model = _cuda_model()
    service = _make_service(model)
    assert service.pin_host_memory is True
    item = _make_item(fingerprint="pinned", fill=2.0)
    service.encode_item(item)
    cached = _cached_entry(service, "pinned")
    assert cached.device.type == "cpu"
    assert cached.is_pinned()
    assert cached.dtype == torch.float16
    torch.cuda.synchronize()
    assert torch.equal(cached.to("cuda"), item.precomputed_embeddings)
    # note (Jeffro): no fingerprint means nothing will be cached, so nothing is staged.
    anonymous = _make_item(fingerprint="ignored")
    anonymous.audio_fingerprint = None
    assert service.stage_host_copy(anonymous, item.precomputed_embeddings) is None


@pytest.mark.accelerator
@_requires_cuda
def test_cuda_hit_attaches_device_tensor_from_pinned_entry() -> None:
    model = _cuda_model()
    service = _make_service(model)
    first = _make_item(fingerprint="hit", fill=1.5)
    second = _make_item(fingerprint="hit", fill=7.0)
    service.encode_item(first)
    service.encode_item(second)
    torch.cuda.synchronize()
    assert model.encode_calls == 1
    assert second.precomputed_embeddings.is_cuda
    assert torch.equal(first.precomputed_embeddings, second.precomputed_embeddings)


@pytest.mark.accelerator
@_requires_cuda
def test_pin_host_memory_flag_disables_pinning() -> None:
    service = _make_service(_cuda_model(), pin_host_memory=False)
    assert service.pin_host_memory is False
    item = _make_item(fingerprint="pageable")
    service.encode_item(item)
    cached = _cached_entry(service, "pageable")
    assert not cached.is_pinned()
    assert service.stats()["pin_prewarm_s"] == 0.0


@pytest.mark.accelerator
@_requires_cuda
def test_pinned_allocation_failure_falls_back_to_pageable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _make_service(_cuda_model())
    assert service.pin_host_memory is True

    def _boom(_tokens: int) -> torch.Tensor:
        raise RuntimeError("cudaHostAlloc failed")

    monkeypatch.setattr(service, "_new_pinned_host", _boom)
    item = _make_item(fingerprint="fallback", fill=4.0)
    service.encode_item(item)
    # note (Jeffro): the entry is still cached, just in pageable memory, and the service
    # stops trying to pin so the failure is paid once, not per request.
    cached = _cached_entry(service, "fallback")
    assert not cached.is_pinned()
    assert service.pin_host_memory is False
    assert service._cache.pin_memory is False
    assert service.stats()["pin_failures"] == 1
    torch.cuda.synchronize()
    assert torch.equal(cached.to("cuda"), item.precomputed_embeddings)


@pytest.mark.accelerator
@_requires_cuda
def test_prewarm_covers_cache_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    allocations: list[int] = []
    original = WhisperPreLMEncoderService._new_pinned_host

    def _counting(self: WhisperPreLMEncoderService, tokens: int) -> torch.Tensor:
        allocations.append(tokens)
        return original(self, tokens)

    monkeypatch.setattr(WhisperPreLMEncoderService, "_new_pinned_host", _counting)
    service = _make_service(_cuda_model(), cache_max_entries=5)
    assert allocations == [_TOKENS] * 5
    assert service.stats()["pin_prewarm_s"] >= 0.0
    # Zero capacity means nothing to warm.
    allocations.clear()
    _make_service(_cuda_model(), cache_max_entries=0)
    assert allocations == []


def test_cache_budget_is_derived_from_entry_count() -> None:
    service = _make_service(cache_max_entries=16)
    entry_bytes = _TOKENS * _HIDDEN_SIZE * torch.float32.itemsize

    assert service.entry_bytes == entry_bytes
    assert service.cache_max_bytes == 16 * entry_bytes
    assert service.cache_capacity_entries == 16
    assert service.stats()["cache_capacity_entries"] == 16


def test_cache_budget_respects_explicit_byte_cap() -> None:
    entry_bytes = _TOKENS * _HIDDEN_SIZE * torch.float32.itemsize

    capped = _make_service(cache_max_entries=16, cache_max_bytes=3 * entry_bytes)
    assert capped.cache_max_bytes == 3 * entry_bytes
    assert capped.cache_capacity_entries == 3

    # A cap above the derived budget never raises it.
    loose = _make_service(cache_max_entries=16, cache_max_bytes=100 * entry_bytes)
    assert loose.cache_max_bytes == 16 * entry_bytes
    assert loose.cache_capacity_entries == 16


def test_cache_capacity_matches_lru_behavior() -> None:
    # Two entries fit; a third insert must evict the least recently used one.
    service = _make_service(cache_max_entries=2)
    for index in range(3):
        service.encode_item(_make_item(fingerprint=f"fp{index}", fill=float(index)))

    stats = service.stats()
    assert stats["cache_capacity_entries"] == 2
    assert stats["cache_entries"] == 2
    assert stats["cache_evictions"] == 1


def test_zero_entries_disables_cache() -> None:
    service = _make_service(cache_max_entries=0)
    assert service.cache_capacity_entries == 0
    item = _make_item(fingerprint="fp0")
    service.encode_item(item)
    assert item.precomputed_embeddings is not None
    assert service.stats()["cache_entries"] == 0


def test_expected_audio_tokens() -> None:
    assert _expected_audio_tokens(SimpleNamespace(num_audio_tokens=12)) == 12
    assert _expected_audio_tokens(SimpleNamespace(num_audio_tokens=None)) is None


def test_build_cache_namespace_is_stable() -> None:
    model = _StubModel()
    fe = SimpleNamespace(
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        nb_max_frames=3000,
        padding_value=0.0,
    )
    a = build_cache_namespace(model, model_path="m", feature_extractor=fe)
    b = build_cache_namespace(model, model_path="m", feature_extractor=fe)
    assert a == b
    assert len(a) == 16


def test_encode_item_attaches_and_clears_feature() -> None:
    service = _make_service()
    item = _make_item(fill=3.0)
    service.encode_item(item)
    assert item.feature is None
    assert item.precomputed_embeddings is not None
    assert tuple(item.precomputed_embeddings.shape) == (_TOKENS, _HIDDEN_SIZE)
    assert item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
    assert service.stats()["misses"] == 1
    assert service.stats()["batches"] == 1


def test_encode_item_hits_cache_on_second_call() -> None:
    model = _StubModel()
    service = _make_service(model)
    first = _make_item(fingerprint="same", fill=2.0)
    second = _make_item(fingerprint="same", fill=9.0)
    service.encode_item(first)
    service.encode_item(second)
    assert model.encode_calls == 1
    assert service.stats()["hits"] == 1
    assert torch.equal(first.precomputed_embeddings, second.precomputed_embeddings)


def test_encode_item_reuses_lookup_cached_embedding() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _make_item(fingerprint="lookup-reuse")
    service.encode_item(item)
    again = _make_item(fingerprint="lookup-reuse", fill=0.0)
    service.encode_item(again)
    assert model.encode_calls == 1
    assert again.feature is None


def test_lookup_cached_embedding_skips_worker() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _make_item(fingerprint="lookup")
    service.encode_item(item)
    cached = service.lookup_cached_embedding("lookup", _TOKENS)
    assert cached is not None
    assert model.encode_calls == 1
    assert service.stats()["hits"] >= 1


def test_no_fingerprint_does_not_cache() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _make_item(fingerprint="ignored")
    item.audio_fingerprint = None
    item.model_specific_data["audio_fingerprint"] = None
    service.encode_item(item)
    assert item.precomputed_embeddings is not None
    assert service.stats()["misses"] == 0
    assert service.stats()["hits"] == 0
    assert len(service._cache) == 0


def test_close_rejects_new_encodes() -> None:
    service = _make_service()
    service.close()
    with pytest.raises(RuntimeError, match="closed"):
        service.encode_item(_make_item())


def test_batched_encode_retries_per_item_on_multi_failure() -> None:
    model = _StubModel()
    model._encoder.fail_multi_item = True
    service = _make_service(model, max_batch_size=4)
    items = [_make_item(fingerprint=f"b{i}", fill=float(i + 1)) for i in range(3)]
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(service.encode_item, items))
    assert all(item.precomputed_embeddings is not None for item in items)
    assert model.encode_calls >= 3
