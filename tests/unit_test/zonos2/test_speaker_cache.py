# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the SpeakerEncoder content-hash cache.

The LRU mechanics live in the shared StageOutputCache (covered by its own tests);
these cover ZONOS2's model-local glue now routed through it: content-hash keying,
a cache hit avoiding a second forward, and recency-ordered eviction at the count
cap. The Qwen3 embedder is stubbed, so no
model checkpoint or GPU is needed (the module still imports torchaudio/transformers).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("torchaudio")
pytest.importorskip("transformers")

import torch  # noqa: E402

import sglang_omni.models.zonos2.components.speaker_encoder as speaker_encoder  # noqa: E402
from sglang_omni.models.zonos2.components.speaker_encoder import (  # noqa: E402
    SPEAKER_EMBEDDING_DIM,
    SpeakerEncoder,
)


class FakeAutoModel:
    def __init__(self) -> None:
        self.received_model_names_or_paths: list[str] = []

    def from_pretrained(
        self,
        model_name_or_path: str,
        *,
        trust_remote_code: bool,
    ) -> "FakeModel":
        self.received_model_names_or_paths.append(model_name_or_path)
        assert trust_remote_code is True
        return FakeModel()


class FakeModel:
    def to(self, device: str) -> "FakeModel":
        return self

    def eval(self) -> "FakeModel":
        return self


class CountingEmbedder:
    """Stand-in for Qwen3SpeakerEmbedding: returns a distinct constant [1, 2048]
    per call so callers can tell which forward produced a cached value."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        self.calls += 1
        return torch.full((1, SPEAKER_EMBEDDING_DIM), float(self.calls))


def make_encoder(max_items: int = 256):
    enc = SpeakerEncoder(device="cpu", cache_max_items=max_items)
    fake = CountingEmbedder()
    # Shadow the lazy loader so encode() never touches the real model / GPU.
    enc.get_embedder = lambda: fake  # type: ignore[assignment]
    return enc, fake


@pytest.mark.parametrize(
    ("configured_model_name_or_path", "expected_model_name_or_path"),
    [
        (None, speaker_encoder.Qwen3SpeakerEmbedding.MODEL_NAME),
        ("/models/qwen3_voice_embedding", "/models/qwen3_voice_embedding"),
    ],
)
def test_speaker_encoder_uses_the_resolved_model(
    monkeypatch: pytest.MonkeyPatch,
    configured_model_name_or_path: str | None,
    expected_model_name_or_path: str,
) -> None:
    fake_auto_model = FakeAutoModel()
    monkeypatch.setattr(speaker_encoder, "AutoModel", fake_auto_model)

    encoder = speaker_encoder.SpeakerEncoder(
        device="cpu",
        speaker_embedding_model_name_or_path=configured_model_name_or_path,
    )
    encoder.get_embedder()
    cache_key = encoder.cache_key(encoder.normalize_input((torch.zeros(1, 8), 16_000)))

    assert fake_auto_model.received_model_names_or_paths == [
        expected_model_name_or_path
    ]
    assert encoder.encoder_id == expected_model_name_or_path
    assert cache_key is not None
    assert cache_key.encoder_id == expected_model_name_or_path


@pytest.mark.parametrize("configured_model_name_or_path", ["", "   "])
def test_speaker_encoder_rejects_an_empty_model_source(
    configured_model_name_or_path: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="speaker_embedding_model_name_or_path must not be empty",
    ):
        speaker_encoder.SpeakerEncoder(
            device="cpu",
            speaker_embedding_model_name_or_path=configured_model_name_or_path,
        )


def test_repeated_reference_hits_cache() -> None:
    enc, fake = make_encoder()
    ref = (torch.zeros(1, 100), 16000)
    first, fp1 = enc.encode_with_fingerprint(ref)
    second, fp2 = enc.encode_with_fingerprint(ref)
    assert fake.calls == 1  # second call served from cache, no extra forward
    assert torch.equal(first, second)
    assert first.shape == (SPEAKER_EMBEDDING_DIM,)
    assert fp1 == fp2 and fp1.startswith("wav:")


def test_concurrent_references_keep_their_own_fingerprints() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    enc = SpeakerEncoder(device="cpu")

    class InterleavingEmbedder:
        def __call__(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
            marker = int(wav.reshape(-1)[0].item())
            if marker == 0:
                first_started.set()
                assert release_first.wait(timeout=2)
            else:
                assert first_started.wait(timeout=2)
                release_first.set()
            return torch.full((1, SPEAKER_EMBEDDING_DIM), float(marker + 1))

    enc.get_embedder = lambda: InterleavingEmbedder()  # type: ignore[assignment]
    first_ref = (torch.zeros(1, 100), 16000)
    second_ref = (torch.ones(1, 100), 16000)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(enc.encode_with_fingerprint, first_ref)
        assert first_started.wait(timeout=2)
        second_future = pool.submit(enc.encode_with_fingerprint, second_ref)
        second_embedding, second_fingerprint = second_future.result(timeout=2)
        first_embedding, first_fingerprint = first_future.result(timeout=2)

    assert torch.equal(first_embedding, torch.ones(SPEAKER_EMBEDDING_DIM))
    assert torch.equal(second_embedding, torch.full((SPEAKER_EMBEDDING_DIM,), 2.0))
    assert first_fingerprint == enc.normalize_input(first_ref).input_key
    assert second_fingerprint == enc.normalize_input(second_ref).input_key
    assert first_fingerprint != second_fingerprint


def test_distinct_references_miss() -> None:
    enc, fake = make_encoder()
    enc.encode((torch.zeros(1, 100), 16000))
    enc.encode((torch.ones(1, 100), 16000))
    assert fake.calls == 2  # different content -> different key -> recompute


def test_eviction_is_recency_ordered() -> None:
    # Distinguishes LRU recency from FIFO: a read of `a` before inserting `c`
    # must keep `a` and evict the truly-least-recently-used `b`.
    enc, fake = make_encoder(max_items=2)
    a = (torch.zeros(1, 100), 16000)
    b = (torch.ones(1, 100), 16000)
    c = (torch.full((1, 100), 2.0), 16000)
    enc.encode(a)  # calls=1, order [a]
    enc.encode(b)  # calls=2, order [a, b]
    enc.encode(a)  # HIT -> refreshes recency, order [b, a]; calls still 2
    assert fake.calls == 2
    enc.encode(c)  # calls=3, evicts LRU=b -> {a, c}
    assert fake.calls == 3
    enc.encode(a)  # a survived (recency-refreshed) -> HIT, calls still 3
    assert fake.calls == 3
    enc.encode(b)  # b was evicted -> recompute
    assert fake.calls == 4
