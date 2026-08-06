# SPDX-License-Identifier: Apache-2.0
"""Reference-cache and schedule contracts for dots.tts."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest
import torch

from sglang_omni.models.dots_tts.payload_types import DotsTtsState
from sglang_omni.models.dots_tts.prompt_builder import normalize_prompt_text
from sglang_omni.models.dots_tts.reference_encode import (
    DotsTtsPromptFeatureHook,
    DotsTtsReferenceEncoder,
)
from sglang_omni.proto import OmniRequest, StagePayload

SAMPLE_RATE = 48000
PATCH_SIZE = 4
LATENT_DIM = 128
SAMPLES_PER_PATCH = PATCH_SIZE * 960


class _FakeTokenizer:
    """Minimal stand-in: one id per character, fixed special-token ids."""

    _SPECIAL_TOKEN_IDS = {
        "<|audio_gen_start|>": 90001,
        "<|audio_gen_span|>": 90002,
        "<|audio_gen_end|>": 90003,
        "<|text_cond_end|>": 90004,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL_TOKEN_IDS.get(token, -1)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(chr(token_id) for token_id in token_ids if token_id < 90000)


class _FakeVocoderInference:
    """Returns a deterministic AudioVAE posterior for the padded waveform."""

    def extract_latents(self, waveform: torch.Tensor) -> torch.Tensor:
        frames = int(waveform.shape[-1]) // (SAMPLES_PER_PATCH // PATCH_SIZE)
        mean = torch.full((1, LATENT_DIM, frames), 0.25)
        log_std = torch.full((1, LATENT_DIM, frames), -10.0)
        return torch.cat([mean, log_std], dim=1)


def _fake_speaker_encoder(waveform: torch.Tensor) -> torch.Tensor:
    del waveform
    return torch.ones((1, 512))


def _write_wav(path: Path, samples: bytes) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples)
    return path


def _make_payload(state: DotsTtsState) -> StagePayload:
    return StagePayload(
        request_id="req",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )


def _make_hook(model_identity: str = "dots.tts-mf") -> DotsTtsPromptFeatureHook:
    return DotsTtsPromptFeatureHook(
        model_identity=model_identity,
        vocoder_inference=_FakeVocoderInference(),
        speaker_encoder=_fake_speaker_encoder,
        sample_rate=SAMPLE_RATE,
        samples_per_patch=SAMPLES_PER_PATCH,
        device=torch.device("cpu"),
    )


def _make_encoder(tmp_path: Path) -> DotsTtsReferenceEncoder:
    return DotsTtsReferenceEncoder(
        tokenizer=_FakeTokenizer(),
        vocoder_inference=_FakeVocoderInference(),
        speaker_encoder=_fake_speaker_encoder,
        sample_rate=SAMPLE_RATE,
        patch_size=PATCH_SIZE,
        samples_per_patch=SAMPLES_PER_PATCH,
        model_identity=str(tmp_path),
        device=torch.device("cpu"),
    )


def test_cache_key_differs_for_same_size_different_content(tmp_path: Path) -> None:
    hook = _make_hook()
    first = _write_wav(tmp_path / "a.wav", b"\x01\x00" * 24000)
    second = _write_wav(tmp_path / "b.wav", b"\x02\x00" * 24000)

    assert first.stat().st_size == second.stat().st_size
    assert hook.input_key(str(first)) != hook.input_key(str(second))
    assert hook.cache_key(str(first)) != hook.cache_key(str(second))


def test_cache_key_is_none_for_missing_reference(tmp_path: Path) -> None:
    hook = _make_hook()

    assert hook.input_key(str(tmp_path / "missing.wav")) is None
    assert hook.cache_key(str(tmp_path / "missing.wav")) is None


def test_encode_payload_attaches_prompt_conditioning_and_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_wav(tmp_path / "ref.wav", b"\x01\x00" * 24000)
    prompt_audio = torch.zeros((1, SAMPLES_PER_PATCH * 3), dtype=torch.float32)
    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.reference_encode.load_reference_waveform",
        lambda path, *, target_sample_rate: prompt_audio,
    )
    payload = _make_payload(
        DotsTtsState(
            text="target",
            prompt_text="reference",
            ref_audio_path=str(reference),
            max_audio_patches=16,
        )
    )

    stored = _make_encoder(tmp_path).encode_payload(payload)
    state = DotsTtsState.from_dict(stored.data)

    # The AR loop regenerates the final prompt patch, so conditioning stops one
    # patch short of the reference audio.
    assert state.prompt_patch_count == 2
    assert state.prompt_latent.shape == (1, 2 * PATCH_SIZE, LATENT_DIM)
    assert state.spk_emb.shape == (1, 512)
    assert state.generation_schedule_ids is not None
    assert state.generation_schedule_ids.count(90002) == 16
    assert state.prompt_tokens == len(state.generation_schedule_ids)


def test_encode_payload_rejects_prompt_longer_than_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_wav(tmp_path / "long.wav", b"\x01\x00" * 24000)
    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.reference_encode.load_reference_waveform",
        lambda path, *, target_sample_rate: torch.zeros(
            (1, SAMPLES_PER_PATCH * 4), dtype=torch.float32
        ),
    )
    payload = _make_payload(
        DotsTtsState(
            text="target",
            prompt_text="reference",
            ref_audio_path=str(reference),
            max_audio_patches=4,
        )
    )

    with pytest.raises(ValueError, match="prompt audio patch count"):
        _make_encoder(tmp_path).encode_payload(payload)


def test_encode_payload_requires_reference_audio(tmp_path: Path) -> None:
    payload = _make_payload(DotsTtsState(text="target", prompt_text="reference"))

    with pytest.raises(ValueError, match="requires reference audio"):
        _make_encoder(tmp_path).encode_payload(payload)


def test_seeded_requests_reuse_the_cached_posterior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _write_wav(tmp_path / "ref.wav", b"\x01\x00" * 24000)
    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.reference_encode.load_reference_waveform",
        lambda path, *, target_sample_rate: torch.zeros(
            (1, SAMPLES_PER_PATCH * 3), dtype=torch.float32
        ),
    )
    encoder = _make_encoder(tmp_path)

    def encode(seed: int) -> torch.Tensor:
        payload = _make_payload(
            DotsTtsState(
                text="target",
                prompt_text="reference",
                ref_audio_path=str(reference),
                max_audio_patches=16,
                seed=seed,
            )
        )
        return DotsTtsState.from_dict(
            encoder.encode_payload(payload).data
        ).prompt_latent

    assert torch.equal(encode(7), encode(7))


def test_prompt_text_gets_the_training_continuation_boundary() -> None:
    assert normalize_prompt_text("  reference  ") == "reference\n"
    assert normalize_prompt_text("   ") == ""
    assert normalize_prompt_text(None) == ""
