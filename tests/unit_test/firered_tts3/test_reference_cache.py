# SPDX-License-Identifier: Apache-2.0
"""Reference encoding cache behavior."""

from pathlib import Path

import torch

from sglang_omni.models.firered_tts3.ref_encode import (
    FireRedReferenceHook,
    ReferenceAudioInput,
    ReferenceConditioning,
)
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService


class ReferenceEncoderStub:
    """Distinct artifacts make cache hits observable without loading an audio model."""

    sample_rate: int = 24000
    pad_multiple: int = 3840
    redae_scale: float = 1.0
    spk_extractor: torch.nn.Module | None = None

    def __init__(self) -> None:
        self.encode_calls: int = 0

    def encode(
        self, path: str, max_seconds: float | None = None
    ) -> dict[str, torch.Tensor]:
        self.encode_calls += 1
        return {"prompt_latents": torch.full((1, 2, 4), float(self.encode_calls))}


def make_reference_service(
    reference_encoder: ReferenceEncoderStub, *, cache_enabled: bool = True
) -> ReferenceEncodeService[
    ReferenceAudioInput,
    ReferenceConditioning,
    ReferenceConditioning,
    ReferenceAudioInput,
]:
    return ReferenceEncodeService(
        FireRedReferenceHook(
            reference_encoder, model_id="stub", cache_enabled=cache_enabled
        ),
        max_items=8,
        max_bytes=1 << 20,
    )


def test_reference_cache_hit_reuses_stored_artifact(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    reference_audio = ReferenceAudioInput(str(reference_audio_path), None)
    first = service.get_or_encode(reference_audio)
    second = service.get_or_encode(reference_audio)
    assert reference_encoder.encode_calls == 1
    assert service.stats()["hits"] == 1
    assert torch.equal(first["prompt_latents"], second["prompt_latents"])

    second["prompt_latents"].fill_(-1.0)
    third = service.get_or_encode(reference_audio)
    assert torch.equal(third["prompt_latents"], first["prompt_latents"])
    service.close()


def test_reference_cache_key_separates_edit_cap(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    uncapped_reference = ReferenceAudioInput(str(reference_audio_path), None)
    capped_reference = ReferenceAudioInput(str(reference_audio_path), 30.0)
    uncapped = service.get_or_encode(uncapped_reference)
    capped = service.get_or_encode(capped_reference)
    assert reference_encoder.encode_calls == 2
    assert service.stats()["entries"] == 2
    assert torch.equal(uncapped["prompt_latents"], torch.ones(1, 2, 4))
    assert torch.equal(capped["prompt_latents"], torch.full((1, 2, 4), 2.0))
    assert torch.equal(
        service.get_or_encode(uncapped_reference)["prompt_latents"],
        uncapped["prompt_latents"],
    )
    assert torch.equal(
        service.get_or_encode(capped_reference)["prompt_latents"],
        capped["prompt_latents"],
    )
    assert reference_encoder.encode_calls == 2
    service.close()


def test_reference_cache_disabled_encodes_every_call(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder, cache_enabled=False)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    reference_audio = ReferenceAudioInput(str(reference_audio_path), None)
    service.get_or_encode(reference_audio)
    service.get_or_encode(reference_audio)
    assert reference_encoder.encode_calls == 2
    stats = service.stats()
    assert stats["uncacheable"] == 2
    assert stats["hits"] == 0
    assert stats["entries"] == 0
    service.close()
