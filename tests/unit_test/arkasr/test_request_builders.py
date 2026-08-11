# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.arkasr.request_builders import make_arkasr_scheduler_adapters
from sglang_omni.preprocessing import transcription
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import DeferredAdmission

_AUDIO_TOKEN_ID = 151663
_HIDDEN_SIZE = 4


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 200000
    all_special_ids = [2]

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        audio_tokens = text.count("<|audio|>")
        return SimpleNamespace(input_ids=[1, *([_AUDIO_TOKEN_ID] * audio_tokens), 2])


class _FeatureExtractor:
    hop_length = 160
    nb_max_frames = 3000

    def __init__(
        self,
        mel_frames: int = 100,
        *,
        return_attention_mask: bool = True,
    ) -> None:
        self.calls = 0
        self.mel_frames = mel_frames
        self.return_attention_mask = return_attention_mask

    def __call__(self, *args, **kwargs):
        self.calls += 1
        result = SimpleNamespace(input_features=torch.zeros((1, 128, self.mel_frames)))
        if self.return_attention_mask:
            result.attention_mask = torch.ones((1, self.mel_frames), dtype=torch.long)
        return result


def _payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )


@pytest.mark.parametrize(
    ("num_samples", "expected_tokens"),
    [
        (16000, 12),
        (31 * 16000, 375),
    ],
)
def test_arkasr_embedding_cache_hit_skips_mel_extraction(
    monkeypatch: pytest.MonkeyPatch,
    num_samples: int,
    expected_tokens: int,
) -> None:
    class _EncoderService:
        def __init__(self) -> None:
            self.lookup: tuple[str, int] | None = None
            self.embedding = torch.zeros((expected_tokens, _HIDDEN_SIZE))

        def lookup_cached_embedding(
            self,
            audio_fingerprint: str,
            expected_audio_tokens: int,
        ) -> torch.Tensor:
            self.lookup = (audio_fingerprint, expected_audio_tokens)
            return self.embedding

        def attach_embedding(self, item, embedding: torch.Tensor) -> None:
            item.precomputed_embeddings = embedding
            item.feature = None

        def submit_item(self, item):
            raise AssertionError("a cache hit must not submit encoder work")

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(num_samples, dtype=np.float32),
    )
    feature_extractor = _FeatureExtractor()
    encoder_service = _EncoderService()
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
        audio_encoder_service=encoder_service,
    )

    data = request_builder(_payload(f"cache-hit-{num_samples}"))

    item = data.req.multimodal_inputs.mm_items[0]
    assert encoder_service.lookup == (data.req.extra_key, expected_tokens)
    assert feature_extractor.calls == 0
    assert item.feature is None
    assert item.precomputed_embeddings is encoder_service.embedding
    assert item.num_audio_tokens == expected_tokens


@pytest.mark.parametrize("return_attention_mask", [True, False])
def test_arkasr_embedding_cache_miss_extracts_and_encodes(
    monkeypatch: pytest.MonkeyPatch,
    return_attention_mask: bool,
) -> None:
    class _EncoderService:
        def __init__(self) -> None:
            self.lookup: tuple[str, int] | None = None
            self.encoded_tokens: int | None = None

        def lookup_cached_embedding(
            self,
            audio_fingerprint: str,
            expected_audio_tokens: int,
        ) -> None:
            self.lookup = (audio_fingerprint, expected_audio_tokens)

        def attach_embedding(self, item, embedding: torch.Tensor) -> None:
            raise AssertionError("a cache miss must not attach a cached embedding")

        def submit_item(self, item):
            self.encoded_tokens = item.num_audio_tokens
            embedding = torch.zeros((item.num_audio_tokens, _HIDDEN_SIZE))
            item.precomputed_embeddings = embedding
            item.feature = None
            future: concurrent.futures.Future[torch.Tensor] = (
                concurrent.futures.Future()
            )
            future.set_result(embedding)
            return future

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    feature_extractor = _FeatureExtractor(return_attention_mask=return_attention_mask)
    encoder_service = _EncoderService()
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
        audio_encoder_service=encoder_service,
    )

    data = request_builder(_payload("cache-miss"))

    assert isinstance(data, DeferredAdmission)
    data.ready.result()
    item = data.value.req.multimodal_inputs.mm_items[0]
    assert encoder_service.lookup == (data.value.req.extra_key, 12)
    assert feature_extractor.calls == 1
    assert encoder_service.encoded_tokens == 12
    assert item.feature is None
    assert item.precomputed_embeddings.shape == (12, _HIDDEN_SIZE)


@pytest.mark.parametrize(
    "feature_extractor",
    [
        SimpleNamespace(nb_max_frames=3000),
        SimpleNamespace(hop_length=160),
        SimpleNamespace(hop_length=0, nb_max_frames=3000),
        SimpleNamespace(hop_length=160, nb_max_frames=0),
    ],
)
def test_arkasr_embedding_cache_requires_valid_audio_length_metadata(
    monkeypatch: pytest.MonkeyPatch,
    feature_extractor: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    encoder_service = SimpleNamespace(
        lookup_cached_embedding=lambda *args: pytest.fail(
            "invalid metadata must fail before cache lookup"
        )
    )
    request_builder, _ = make_arkasr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
        audio_encoder_service=encoder_service,
    )

    with pytest.raises(ValueError, match="audio length metadata"):
        request_builder(_payload("invalid-metadata"))
