# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
ln = pytest.importorskip("pyloudnorm")

from sglang_omni.models.chatterbox.stages import (
    REFERENCE_MIN_DURATION_S,
    REFERENCE_TARGET_LUFS,
    S3_SR,
    ChatterboxT3ReferenceEncodeHook,
    _ChatterboxReferenceInput,
    _norm_loudness,
    _punc_norm,
    _reference_payload_is_supported,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world", "Hello world."),
        ("", "You need to add some text for me to talk."),
        ("a:b", "A,b."),
        ("hello!", "Hello!"),
        ("HELLO", "HELLO."),
    ],
)
def test_punc_norm(text: str, expected: str) -> None:
    assert _punc_norm(text) == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ({"audio_path": "/tmp/x.wav"}, True),
        ({"bytes": b"RIFF"}, True),
        ({"base64": "cmlmZg=="}, True),
        ({"data": "cmlmZg=="}, True),
        ({"text": "no audio"}, False),
        ({}, False),
    ],
)
def test_reference_payload_is_supported(ref: dict, expected: bool) -> None:
    assert _reference_payload_is_supported(ref) is expected


def test_norm_loudness_preserves_float32_dtype() -> None:
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(S3_SR).astype(np.float32)
    out = _norm_loudness(wav, S3_SR, REFERENCE_TARGET_LUFS)
    assert out.dtype == np.float32


def test_norm_loudness_scales_to_target_lufs() -> None:
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(S3_SR).astype(np.float32)
    out = _norm_loudness(wav, S3_SR, REFERENCE_TARGET_LUFS)
    meter = ln.Meter(S3_SR)
    assert abs(meter.integrated_loudness(out) - REFERENCE_TARGET_LUFS) < 2.0


def test_encode_one_rejects_short_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    hook = ChatterboxT3ReferenceEncodeHook(
        ve=Mock(), s3_tokenizer=Mock(), checkpoint_id="test"
    )
    short = np.zeros(int(S3_SR * (REFERENCE_MIN_DURATION_S - 1.0)), dtype=np.float32)
    monkeypatch.setattr(hook, "_load_reference_wav", lambda item: short)

    with pytest.raises(ValueError, match="longer than 5 seconds"):
        hook.encode_one(_ChatterboxReferenceInput("path", "/tmp/x.wav"))


def test_encode_one_accepts_long_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    ve = Mock()
    ve.embeds_from_wavs = Mock(return_value=np.zeros((1, 256), dtype=np.float32))
    tokenizer = Mock()
    tokenizer.forward = Mock(return_value=(np.zeros((1, 100), dtype=np.int64), None))
    hook = ChatterboxT3ReferenceEncodeHook(
        ve=ve, s3_tokenizer=tokenizer, checkpoint_id="test"
    )
    long = np.zeros(int(S3_SR * (REFERENCE_MIN_DURATION_S + 1.0)), dtype=np.float32)
    monkeypatch.setattr(hook, "_load_reference_wav", lambda item: long)

    ref = hook.encode_one(_ChatterboxReferenceInput("path", "/tmp/x.wav"))

    assert ref.cond_prompt_speech_tokens == [0] * 100
    assert ref.speaker_embedding.shape == (1, 256)
