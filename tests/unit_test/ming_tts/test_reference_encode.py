# SPDX-License-Identifier: Apache-2.0
"""Cache-key contract for the Ming-Omni-TTS reference encoder."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from sglang_omni.models.ming_tts.reference_encode import _MingTTSReferenceEncodeHook
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService


class _StubEncoder:
    """Only the attributes the hook consults; no AudioVAE/CampPlus weights."""

    sample_rate = 44100
    patch_size = 4

    def __init__(self) -> None:
        self.encode_calls: list[str] = []

    def _audio_vae_floating_dtype(self):
        return torch.bfloat16

    def _encode_reference(self, ref_audio: str) -> dict:
        self.encode_calls.append(ref_audio)
        return {
            "prompt_latent_token_count": 1,
            "content": Path(ref_audio).read_bytes(),
        }


def _write_wav_like(path: Path, middle: bytes) -> None:
    """Same-size payloads that differ only in the middle bytes."""
    assert len(middle) == 4
    path.write_bytes(b"RIFF" + b"\x00" * 9000 + middle + b"\x00" * 9000 + b"data")


def _hook_and_service(tmp_path) -> tuple[_StubEncoder, ReferenceEncodeService]:
    encoder = _StubEncoder()
    hook = _MingTTSReferenceEncodeHook(encoder, model_identity=str(tmp_path))
    return encoder, ReferenceEncodeService(hook, max_items=16, max_bytes=1 << 20)


def test_same_size_references_do_not_share_a_cache_entry(tmp_path) -> None:
    """Two same-size files differing only in the middle must key separately;
    a sampled head/tail hash would collide here and serve the wrong speaker."""
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav_like(a, b"AAAA")
    _write_wav_like(b, b"BBBB")
    assert a.stat().st_size == b.stat().st_size

    encoder, service = _hook_and_service(tmp_path)
    artifact_a = service.get_or_encode(str(a))
    artifact_b = service.get_or_encode(str(b))

    assert artifact_a["content"] != artifact_b["content"]
    assert encoder.encode_calls == [str(a), str(b)]
    stats = service.stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 0


def test_same_reference_file_hits_the_cache(tmp_path) -> None:
    ref = tmp_path / "ref.wav"
    _write_wav_like(ref, b"AAAA")

    encoder, service = _hook_and_service(tmp_path)
    service.get_or_encode(str(ref))
    service.get_or_encode(str(ref))

    assert encoder.encode_calls == [str(ref)]
    assert service.stats()["hits"] == 1


def test_rewritten_reference_is_not_served_stale(tmp_path) -> None:
    ref = tmp_path / "ref.wav"
    _write_wav_like(ref, b"AAAA")

    encoder, service = _hook_and_service(tmp_path)
    first = service.get_or_encode(str(ref))

    _write_wav_like(ref, b"BBBB")
    os.utime(ref, (1_700_000_000, 1_700_000_000))
    second = service.get_or_encode(str(ref))

    assert first["content"] != second["content"]
    assert len(encoder.encode_calls) == 2
