# SPDX-License-Identifier: Apache-2.0
"""Tests exact-length batching parity for dots.tts reference encoding.

Padding perturbs both speaker embeddings and AudioVAE latents, so batching must
group only identical lengths. These stubs are intentionally padding-sensitive so
any regression to padded batching fails loudly.
"""

from __future__ import annotations

import math
import random
import threading

import pytest
import torch

from sglang_omni.models.dots_tts.codec import DotsAudioCodec

HOP_SIZE = 8
PATCH_SIZE = 4
LATENT_DIM = 3
SAMPLES_PER_PATCH = HOP_SIZE * PATCH_SIZE


class _PaddingSensitiveSpeaker:
    """Padding-sensitive speaker stub; accepts but ignores ``audio_lengths``."""

    max_audio_seconds = 0.0
    sample_rate = 48000

    def __call__(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch = audio[:, 0]
        return torch.stack(
            [batch.mean(-1), batch.std(-1), batch.abs().max(-1).values], dim=-1
        )


class _RandomCroppingSpeaker(_PaddingSensitiveSpeaker):
    """Mirrors upstream random crop beyond the duration cap."""

    def __init__(self, max_audio_seconds: float = 1.0, sample_rate: int = 48000):
        self.max_audio_seconds = max_audio_seconds
        self.sample_rate = sample_rate

    def __call__(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        limit = round(self.sample_rate * self.max_audio_seconds)
        total = audio.shape[-1]
        if total > limit:
            start = random.randint(0, total - limit)
            audio = audio[..., start : start + limit]
        return super().__call__(audio, audio_lengths)


class _PaddingSensitiveLatents:
    """Padding-sensitive latent stub that perturbs all frames."""

    def extract_latents(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size, _, samples = audio.shape
        frames = audio.reshape(batch_size, samples // HOP_SIZE, HOP_SIZE).mean(-1)
        causal = torch.cumsum(frames, dim=1)
        causal = causal + frames.mean(dim=1, keepdim=True)
        scales = torch.arange(1, 2 * LATENT_DIM + 1, dtype=torch.float32)
        return causal.unsqueeze(1) * scales.reshape(1, -1, 1)


def _make_codec() -> DotsAudioCodec:
    codec = object.__new__(DotsAudioCodec)
    codec.speaker = _PaddingSensitiveSpeaker()
    codec.inference = _PaddingSensitiveLatents()
    codec.vocoder = None
    codec.patch_size = PATCH_SIZE
    codec.latent_dim = LATENT_DIM
    codec.sample_rate = 48000
    codec.hop_size = HOP_SIZE
    codec.device = torch.device("cpu")
    codec.lock = threading.RLock()
    return codec


def _waveform(patches: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, patches * SAMPLES_PER_PATCH, generator=generator)


def _install_waveforms(
    monkeypatch: pytest.MonkeyPatch, waveforms: dict[str, torch.Tensor]
) -> None:
    def _fake_load_audio(path: str, **_: object):
        return waveforms[path].reshape(-1).numpy()

    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.codec.load_audio", _fake_load_audio
    )


def _set_tf32(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", enabled)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", enabled)


def test_batch_of_one_is_identical_to_the_unbatched_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _make_codec()
    waveform = _waveform(6, seed=1)
    _install_waveforms(monkeypatch, {"a.wav": waveform})

    single = codec.encode_reference("a.wav")
    batched = codec._encode_waveforms([waveform])[0]

    assert torch.equal(single["speaker_embedding"], batched["speaker_embedding"])
    assert torch.equal(single["latent_distribution"], batched["latent_distribution"])


def test_equal_length_batch_is_bit_identical_to_encoding_each_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: same length in a batch changes nothing at all."""
    codec = _make_codec()
    waveforms = {
        "a.wav": _waveform(5, seed=2),
        "b.wav": _waveform(5, seed=3),
        "c.wav": _waveform(5, seed=4),
    }
    _install_waveforms(monkeypatch, waveforms)

    singles = [codec.encode_reference(name) for name in waveforms]
    batched = codec.encode_reference_batch(list(waveforms))

    assert len(batched) == len(singles)
    for single, batch_item in zip(singles, batched):
        assert torch.equal(single["speaker_embedding"], batch_item["speaker_embedding"])
        assert torch.equal(
            single["latent_distribution"], batch_item["latent_distribution"]
        )


def test_mixed_lengths_still_match_the_unbatched_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parity check: mixed-length batches must never mix lengths in a group."""
    codec = _make_codec()
    waveforms = {
        "a.wav": _waveform(3, seed=5),
        "b.wav": _waveform(11, seed=6),
        "c.wav": _waveform(3, seed=7),
        "d.wav": _waveform(7, seed=8),
        "e.wav": _waveform(11, seed=9),
    }
    _install_waveforms(monkeypatch, waveforms)

    singles = [codec.encode_reference(name) for name in waveforms]
    batched = codec.encode_reference_batch(list(waveforms))

    for (name, _), single, batch_item in zip(waveforms.items(), singles, batched):
        assert torch.equal(
            single["speaker_embedding"], batch_item["speaker_embedding"]
        ), name
        assert torch.equal(
            single["latent_distribution"], batch_item["latent_distribution"]
        ), name


def test_stub_actually_detects_padding_so_parity_tests_are_not_vacuous() -> None:
    """Guard check: padding must change stub output or parity tests are vacuous."""
    codec = _make_codec()
    short = _waveform(3, seed=10)
    padded = torch.zeros(1, 11 * SAMPLES_PER_PATCH)
    padded[0, : short.shape[-1]] = short.reshape(-1)

    alone = codec._encode_waveforms([short])[0]
    in_padded_row = codec._encode_waveforms([padded])[0]
    frames = short.shape[-1] // HOP_SIZE

    assert not torch.equal(
        alone["speaker_embedding"], in_padded_row["speaker_embedding"]
    )
    assert not torch.equal(
        alone["latent_distribution"],
        in_padded_row["latent_distribution"][..., :frames],
    )


def test_unequal_lengths_are_rejected_by_the_batched_forward() -> None:
    codec = _make_codec()
    with pytest.raises(ValueError, match="equal-length"):
        codec._encode_waveforms([_waveform(3, seed=11), _waveform(4, seed=12)])


def test_latents_have_each_items_true_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _make_codec()
    waveforms = {
        "a.wav": _waveform(2, seed=13),
        "b.wav": _waveform(9, seed=14),
        "c.wav": _waveform(2, seed=15),
    }
    _install_waveforms(monkeypatch, waveforms)

    batched = codec.encode_reference_batch(list(waveforms))

    for (name, waveform), artifact in zip(waveforms.items(), batched):
        expected_frames = waveform.shape[-1] // HOP_SIZE
        assert artifact["latent_distribution"].shape == (
            1,
            2 * LATENT_DIM,
            expected_frames,
        ), name


def test_consumed_prompt_latents_are_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _make_codec()
    waveforms = {
        "a.wav": _waveform(6, seed=16),
        "b.wav": _waveform(6, seed=17),
    }
    _install_waveforms(monkeypatch, waveforms)

    singles = [codec.encode_reference(name) for name in waveforms]
    batched = codec.encode_reference_batch(list(waveforms))

    for single, batch_item in zip(singles, batched):
        expected = codec.sample_prompt_latents(single["latent_distribution"], seed=42)
        actual = codec.sample_prompt_latents(batch_item["latent_distribution"], seed=42)
        assert torch.equal(expected, actual)


def test_length_groups_are_exact_not_bucketed() -> None:
    waveforms = [
        torch.zeros(1, 100),
        torch.zeros(1, 110),
        torch.zeros(1, 100),
        torch.zeros(1, 120),
    ]
    groups = DotsAudioCodec._length_groups(waveforms)

    assert groups == {100: [0, 2], 110: [1], 120: [3]}
    for length, indices in groups.items():
        assert all(int(waveforms[i].shape[-1]) == length for i in indices)


def test_length_groups_preserve_every_index() -> None:
    waveforms = [torch.zeros(1, n) for n in (30, 10, 30, 20, 10, 10)]
    groups = DotsAudioCodec._length_groups(waveforms)
    assert sorted(i for indices in groups.values() for i in indices) == list(range(6))


def test_empty_batch_returns_empty() -> None:
    codec = _make_codec()
    assert codec.encode_reference_batch([]) == []
    assert codec._encode_waveforms([]) == []


def test_batch_preserves_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = _make_codec()
    waveforms = {
        "long.wav": _waveform(12, seed=18),
        "short.wav": _waveform(2, seed=19),
        "mid.wav": _waveform(12, seed=20),
    }
    _install_waveforms(monkeypatch, waveforms)

    names = list(waveforms)
    batched = codec.encode_reference_batch(names)
    singles = [codec.encode_reference(name) for name in names]

    for name, single, batch_item in zip(names, singles, batched):
        assert torch.equal(
            single["speaker_embedding"], batch_item["speaker_embedding"]
        ), name


def test_hop_misaligned_latents_are_rejected() -> None:
    codec = _make_codec()

    class _WrongRate:
        def extract_latents(self, audio: torch.Tensor) -> torch.Tensor:
            frames = audio.shape[-1] // HOP_SIZE
            return torch.zeros(audio.shape[0], 2 * LATENT_DIM, frames + 1)

    codec.inference = _WrongRate()
    with pytest.raises(RuntimeError, match="hop-aligned"):
        codec._encode_waveforms([_waveform(3, seed=21)])


def test_encoder_coalesces_equal_length_references_into_one_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import concurrent.futures

    from sglang_omni.models.dots_tts.codec import DotsReferenceEncoder

    _set_tf32(monkeypatch, enabled=False)
    codec = _make_codec()
    # Same duration, different content: the case batching can actually help.
    waveforms = {
        f"ref-{index}.wav": _waveform(4, seed=30 + index) for index in range(4)
    }
    _install_waveforms(monkeypatch, waveforms)

    calls: list[int] = []
    original = codec._encode_waveforms

    def _counting(batch):
        calls.append(len(batch))
        return original(batch)

    codec._encode_waveforms = _counting  # type: ignore[method-assign]

    encoder = DotsReferenceEncoder(
        codec,
        model_id="stub",
        max_batch_size=8,
        max_batch_wait_ms=50.0,
    )
    try:
        assert encoder.service.batching_enabled is True
        names = list(waveforms)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = [pool.submit(encoder.service.get_or_encode, n) for n in names]
            artifacts = [f.result(timeout=10) for f in futures]

        assert len(artifacts) == len(names)
        assert max(calls) > 1, "equal-length references should share a forward"
        stats = encoder.service.stats()
        assert stats["misses"] == len(names)
        assert stats["batched_items"] == len(names)
    finally:
        encoder.close()


def test_encoder_handles_all_distinct_lengths_without_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worst case for grouping: every group is size one, results still correct."""
    import concurrent.futures

    from sglang_omni.models.dots_tts.codec import DotsReferenceEncoder

    _set_tf32(monkeypatch, enabled=False)
    codec = _make_codec()
    waveforms = {
        f"ref-{index}.wav": _waveform(4 + index, seed=40 + index) for index in range(4)
    }
    _install_waveforms(monkeypatch, waveforms)

    expected = {name: codec.encode_reference(name) for name in waveforms}

    encoder = DotsReferenceEncoder(
        codec, model_id="stub", max_batch_size=8, max_batch_wait_ms=50.0
    )
    try:
        names = list(waveforms)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {
                name: pool.submit(encoder.service.get_or_encode, name) for name in names
            }
            for name, future in futures.items():
                artifact = future.result(timeout=10)
                assert torch.equal(
                    expected[name]["speaker_embedding"], artifact["speaker_embedding"]
                ), name
                assert torch.equal(
                    expected[name]["latent_distribution"],
                    artifact["latent_distribution"],
                ), name
    finally:
        encoder.close()


def test_cuda_reference_batching_is_disabled_with_default_cudnn_tf32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.dots_tts.codec import DotsReferenceEncoder

    _set_tf32(monkeypatch, enabled=False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    codec = _make_codec()
    codec.device = torch.device("cuda")
    encoder = DotsReferenceEncoder(codec, model_id="stub", max_batch_size=8)
    try:
        assert encoder.service.batching_enabled is False
    finally:
        encoder.close()


def test_reference_executor_defaults_to_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.dots_tts import stages

    _set_tf32(monkeypatch, enabled=False)
    codec = _make_codec()
    monkeypatch.setattr(
        stages, "load_dots_audio_codec", lambda *_args, **_kwargs: codec
    )

    scheduler = stages.create_reference_encode_executor("stub", device="cpu")
    try:
        encoder = scheduler._fn.__self__
        assert encoder.service.batching_enabled is False
    finally:
        scheduler.stop()


def test_reference_executor_stop_closes_batch_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.dots_tts import stages

    _set_tf32(monkeypatch, enabled=False)
    codec = _make_codec()
    monkeypatch.setattr(
        stages, "load_dots_audio_codec", lambda *_args, **_kwargs: codec
    )
    scheduler = stages.create_reference_encode_executor(
        "stub", device="cpu", max_batch_size=2
    )
    encoder = scheduler._fn.__self__
    assert encoder.service._batch_thread is not None
    assert encoder.service._batch_thread.is_alive()

    scheduler.stop()

    assert not encoder.service._batch_thread.is_alive()


# --------------------------------------------------------------------------
# Deterministic speaker cropping (upstream _crop_audio non-determinism)
# --------------------------------------------------------------------------
def _cropping_codec(max_audio_seconds: float = 1.0) -> DotsAudioCodec:
    codec = _make_codec()
    codec.speaker = _RandomCroppingSpeaker(
        max_audio_seconds=max_audio_seconds, sample_rate=codec.sample_rate
    )
    return codec


def _seconds_to_patches(codec: DotsAudioCodec, seconds: float) -> int:
    samples = int(seconds * codec.sample_rate)
    return max(1, math.ceil(samples / SAMPLES_PER_PATCH))


def test_stub_reproduces_upstream_random_crop() -> None:
    """Guard check: without the fix, this stub must vary or tests become vacuous."""
    codec = _cropping_codec()
    long_ref = _waveform(_seconds_to_patches(codec, 3.0), seed=50)
    batch = long_ref.reshape(1, 1, -1)
    lengths = torch.tensor([batch.shape[-1]])

    random.seed(1)
    first = codec.speaker(batch, lengths)
    spread = max(
        (first - codec.speaker(batch, lengths)).abs().max().item() for _ in range(20)
    )
    assert spread > 0.0, "stub does not reproduce the upstream random crop"


def test_long_references_encode_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix: repeated encodes of a >max_audio_seconds reference must match."""
    codec = _cropping_codec()
    long_ref = _waveform(_seconds_to_patches(codec, 3.0), seed=51)
    _install_waveforms(monkeypatch, {"long.wav": long_ref})

    baseline = codec.encode_reference("long.wav")
    for _ in range(20):
        repeat = codec.encode_reference("long.wav")
        assert torch.equal(
            baseline["speaker_embedding"], repeat["speaker_embedding"]
        ), "speaker embedding varies across encodes of the same reference"


def test_long_reference_is_unaffected_by_global_random_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent requests reseeding `random` must not change the embedding."""
    codec = _cropping_codec()
    long_ref = _waveform(_seconds_to_patches(codec, 3.0), seed=52)
    _install_waveforms(monkeypatch, {"long.wav": long_ref})

    random.seed(0)
    baseline = codec.encode_reference("long.wav")
    for seed in (1, 7, 99, 12345):
        random.seed(seed)
        assert torch.equal(
            baseline["speaker_embedding"],
            codec.encode_reference("long.wav")["speaker_embedding"],
        )


def test_speaker_sees_the_leading_window_only() -> None:
    """Pre-crop must be the leading window, matching upstream's start=0 draw."""
    codec = _cropping_codec()
    limit = codec._speaker_sample_limit()
    assert limit == codec.sample_rate

    long_ref = _waveform(_seconds_to_patches(codec, 3.0), seed=53)
    batch = long_ref.reshape(1, 1, -1)
    lengths = torch.tensor([batch.shape[-1]])
    cropped, cropped_lengths = codec._speaker_input(batch, lengths)

    assert cropped.shape[-1] == limit
    assert torch.equal(cropped[0, 0], batch[0, 0, :limit])
    assert int(cropped_lengths[0]) == limit


def test_short_references_are_passed_through_untouched() -> None:
    """Below the cap there is nothing to crop, so the tensor must be identical."""
    codec = _cropping_codec()
    short = _waveform(_seconds_to_patches(codec, 0.5), seed=54)
    batch = short.reshape(1, 1, -1)
    lengths = torch.tensor([batch.shape[-1]])

    cropped, cropped_lengths = codec._speaker_input(batch, lengths)
    assert cropped is batch
    assert cropped_lengths is lengths


def test_audiovae_still_receives_the_full_length_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the speaker view is truncated; latents must cover the whole file."""
    codec = _cropping_codec()
    patches = _seconds_to_patches(codec, 3.0)
    long_ref = _waveform(patches, seed=55)
    _install_waveforms(monkeypatch, {"long.wav": long_ref})

    artifact = codec.encode_reference("long.wav")
    expected_frames = long_ref.shape[-1] // HOP_SIZE
    assert artifact["latent_distribution"].shape[-1] == expected_frames


def test_no_cropping_when_encoder_has_no_duration_cap() -> None:
    codec = _make_codec()  # max_audio_seconds = 0.0
    assert codec._speaker_sample_limit() is None

    audio = _waveform(8, seed=56).reshape(1, 1, -1)
    lengths = torch.tensor([audio.shape[-1]])
    cropped, cropped_lengths = codec._speaker_input(audio, lengths)
    assert cropped is audio
    assert cropped_lengths is lengths


def test_cropping_limit_uses_the_encoders_own_sample_rate() -> None:
    """`_crop_audio` measures against the encoder's rate, not the codec's."""
    codec = _cropping_codec()
    codec.speaker.sample_rate = 16000
    codec.speaker.max_audio_seconds = 10.0
    assert codec._speaker_sample_limit() == 160000
    assert codec.sample_rate == 48000


def test_batched_long_references_are_also_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _cropping_codec()
    patches = _seconds_to_patches(codec, 3.0)
    waveforms = {
        "a.wav": _waveform(patches, seed=57),
        "b.wav": _waveform(patches, seed=58),
    }
    _install_waveforms(monkeypatch, waveforms)

    baseline = codec.encode_reference_batch(list(waveforms))
    for seed in (3, 17, 2024):
        random.seed(seed)
        repeat = codec.encode_reference_batch(list(waveforms))
        for one, two in zip(baseline, repeat):
            assert torch.equal(one["speaker_embedding"], two["speaker_embedding"])
