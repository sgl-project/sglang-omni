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
    max_batch_size: int = 8,
) -> WhisperPreLMEncoderService:
    service = WhisperPreLMEncoderService(
        model or _StubModel(),
        cache_namespace=_NAMESPACE,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=1 << 20,
        max_batch_size=max_batch_size,
    )
    _SERVICES.append(service)
    return service


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
