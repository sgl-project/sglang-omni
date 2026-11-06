# SPDX-License-Identifier: Apache-2.0
"""Tests for ARK-ASR request building and embedding-cache admission."""

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from transformers import WhisperFeatureExtractor

from sglang_omni.models.arkasr.request_builders import make_arkasr_scheduler_adapters
from sglang_omni.preprocessing import transcription
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import DeferredAdmission

AUDIO_TOKEN_ID = 151663


class FakeTokenizer:
    eos_token_id = 99
    vocab_size = 200000
    all_special_ids = [99]

    def __len__(self) -> int:
        return self.vocab_size

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def __call__(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert not add_special_tokens
        audio_token_count = text.count("<|audio|>")
        return SimpleNamespace(
            input_ids=[1, *([AUDIO_TOKEN_ID] * audio_token_count), 2]
        )


class UnexpectedFeatureExtractor:
    hop_length = 160
    nb_max_frames = 3000

    def __call__(
        self,
        audio: np.ndarray,
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
        padding: str,
        truncation: bool,
    ) -> SimpleNamespace:
        raise AssertionError("feature extractor should not run on an early cache hit")


class FeatureExtractor:
    hop_length = 160
    nb_max_frames = 3000

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        audio: np.ndarray,
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
        padding: str,
        truncation: bool,
    ) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            input_features=torch.zeros((1, 128, 100)),
            attention_mask=torch.ones((1, 100), dtype=torch.long),
        )


class HitEncoderService:
    def __init__(self, expected_tokens: int) -> None:
        self.lookup: tuple[str, int] | None = None
        self.embedding = torch.zeros((expected_tokens, 4))

    def lookup_cached_embedding(
        self, audio_fingerprint: str, expected_tokens: int
    ) -> torch.Tensor | None:
        self.lookup = (audio_fingerprint, expected_tokens)
        return self.embedding

    def submit_cached_item(
        self,
        item: MultimodalDataItem,
        embedding: torch.Tensor,
    ) -> concurrent.futures.Future[torch.Tensor]:
        item.precomputed_embeddings = embedding
        item.feature = None
        future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
        future.set_result(embedding)
        return future

    def submit_item(
        self, item: MultimodalDataItem
    ) -> concurrent.futures.Future[torch.Tensor]:
        raise AssertionError("encoder should not run on an early cache hit")


class MissEncoderService:
    def __init__(self) -> None:
        self.lookup: tuple[str, int] | None = None
        self.encoded_feature: torch.Tensor | None = None
        self.encoded_attention_mask: torch.Tensor | None = None
        self.encoded_tokens: int | None = None

    def lookup_cached_embedding(
        self, audio_fingerprint: str, expected_tokens: int
    ) -> torch.Tensor | None:
        self.lookup = (audio_fingerprint, expected_tokens)
        return None

    def submit_cached_item(
        self,
        item: MultimodalDataItem,
        embedding: torch.Tensor,
    ) -> concurrent.futures.Future[torch.Tensor]:
        raise AssertionError("a cache miss must not attach a cached embedding")

    def submit_item(
        self, item: MultimodalDataItem
    ) -> concurrent.futures.Future[torch.Tensor]:
        self.encoded_feature = item.feature
        self.encoded_attention_mask = item.feature_attention_mask
        self.encoded_tokens = item.num_audio_tokens
        item.precomputed_embeddings = torch.zeros((item.num_audio_tokens, 4))
        item.feature = None
        future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
        future.set_result(item.precomputed_embeddings)
        return future


def make_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )


def test_embedding_cache_hit_skips_mel_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    service = HitEncoderService(expected_tokens=12)
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=UnexpectedFeatureExtractor(),
        audio_encoder_service=service,
    )

    result = request_builder(make_payload("early-hit"))

    assert isinstance(result, DeferredAdmission)
    result.ready.result(timeout=1)
    request_data = result.value
    item = request_data.req.multimodal_inputs.mm_items[0]
    assert service.lookup == (request_data.req.extra_key, 12)
    assert item.num_audio_tokens == 12
    assert item.feature is None
    assert item.precomputed_embeddings is service.embedding
    assert request_data.prefill_coalesce_after_build_drain_hint is True
    assert len(request_data.prompt_token_ids or []) == 14


def test_embedding_cache_hit_uses_truncated_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(481120, dtype=np.float32),
    )
    service = HitEncoderService(expected_tokens=375)
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=UnexpectedFeatureExtractor(),
        audio_encoder_service=service,
    )

    result = request_builder(make_payload("truncated-hit"))

    assert isinstance(result, DeferredAdmission)
    result.ready.result(timeout=1)
    request_data = result.value
    assert service.lookup == (request_data.req.extra_key, 375)
    assert request_data.req.multimodal_inputs.mm_items[0].num_audio_tokens == 375


@pytest.mark.parametrize("sample_count", [2399, 2400, 479999, 480000, 480001, 481120])
def test_estimated_tokens_match_real_whisper_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    sample_count: int,
) -> None:
    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    service = MissEncoderService()
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(sample_count, dtype=np.float32),
    )
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
        audio_encoder_service=service,
    )

    result = request_builder(make_payload(f"boundary-{sample_count}"))

    assert isinstance(result, DeferredAdmission)
    result.ready.result(timeout=1)
    assert service.lookup is not None
    assert service.lookup[1] == service.encoded_tokens


def test_embedding_cache_miss_preserves_mel_and_deferred_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    feature_extractor = FeatureExtractor()
    service = MissEncoderService()
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
        audio_encoder_service=service,
    )

    result = request_builder(make_payload("early-miss"))

    assert isinstance(result, DeferredAdmission)
    result.ready.result(timeout=1)
    request_data = result.value
    assert service.lookup == (request_data.req.extra_key, 12)
    assert feature_extractor.calls == 1
    assert service.encoded_feature is not None
    assert tuple(service.encoded_feature.shape) == (1, 128, 100)
    assert service.encoded_attention_mask is not None
    assert tuple(service.encoded_attention_mask.shape) == (1, 100)
    assert int(service.encoded_attention_mask.sum()) == 100
    assert service.encoded_tokens == 12
    assert request_data.prefill_coalesce_after_build_drain_hint is False
    assert request_data.req.multimodal_inputs.mm_items[0].feature is None


def test_request_without_encoder_service_disables_cache_hit_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    feature_extractor = FeatureExtractor()
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
        audio_encoder_service=None,
    )

    request_data = request_builder(make_payload("no-service"))

    assert not isinstance(request_data, DeferredAdmission)
    assert request_data.prefill_coalesce_after_build_drain_hint is False
    assert feature_extractor.calls == 1
