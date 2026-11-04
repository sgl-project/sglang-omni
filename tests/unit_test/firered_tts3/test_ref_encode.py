# SPDX-License-Identifier: Apache-2.0
"""Reference cache isolation, edit-cap keys, and cache bypass."""

from __future__ import annotations

from pathlib import Path

import torch

from sglang_omni.models.firered_tts3.ref_encode import FireRedReferenceHook
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService


class StubReferenceBundle:
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


def make_service(
    bundle: StubReferenceBundle, *, cache_enabled: bool = True
) -> ReferenceEncodeService:
    return ReferenceEncodeService(
        FireRedReferenceHook(bundle, model_id="stub", cache_enabled=cache_enabled),
        max_items=8,
        max_bytes=1 << 20,
    )


def test_reference_cache_hit_reuses_stored_artifact(tmp_path: Path) -> None:
    bundle = StubReferenceBundle()
    service = make_service(bundle)
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"ref-audio-bytes")

    first = service.get_or_encode((str(clip), None))
    second = service.get_or_encode((str(clip), None))
    assert bundle.encode_calls == 1
    assert service.stats()["hits"] == 1
    assert torch.equal(first["prompt_latents"], second["prompt_latents"])

    second["prompt_latents"].fill_(-1.0)
    third = service.get_or_encode((str(clip), None))
    assert torch.equal(third["prompt_latents"], first["prompt_latents"])
    service.close()


def test_reference_cache_key_separates_edit_cap(tmp_path: Path) -> None:
    bundle = StubReferenceBundle()
    service = make_service(bundle)
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"ref-audio-bytes")

    service.get_or_encode((str(clip), None))
    service.get_or_encode((str(clip), 30.0))
    assert bundle.encode_calls == 2
    assert service.stats()["entries"] == 2
    service.close()


def test_reference_cache_disabled_encodes_every_call(tmp_path: Path) -> None:
    bundle = StubReferenceBundle()
    service = make_service(bundle, cache_enabled=False)
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"ref-audio-bytes")

    service.get_or_encode((str(clip), None))
    service.get_or_encode((str(clip), None))
    assert bundle.encode_calls == 2
    stats = service.stats()
    assert stats["uncacheable"] == 2
    assert stats["hits"] == 0
    assert stats["entries"] == 0
    service.close()
