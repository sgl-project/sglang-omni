# SPDX-License-Identifier: Apache-2.0
"""Prompt formatting, reference audio boundaries, and reference caching."""

from __future__ import annotations

import base64
import binascii
import io
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Literal
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.models.firered_tts3.preprocessing import (
    compose_speaker_chatml,
    encode_speaker_text,
)
from sglang_omni.models.firered_tts3.ref_encode import FireRedReferenceHook
from sglang_omni.models.firered_tts3.task_prompts import (
    count_audio_patches,
    resolve_acoustic_instruction,
)
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer


@pytest.fixture
def audio_server_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    ) as server:
        server_thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        server_thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server_thread.join()


def test_speaker_chatml_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError, match="interleave_ratio"):
        compose_speaker_chatml(
            "<|f531_speaker|>", dual_stream=True, interleave_ratio=(0, 1)
        )


@pytest.mark.parametrize(
    "text,expected_tokens",
    [("你 好世界", ["你", "好", "世", "界"]), ("生僻", ["生僻"])],
)
def test_speaker_text_expands_han_tokens_and_preserves_unknown_characters(
    text: str, expected_tokens: list[str]
) -> None:
    tokenizer = make_speaker_tokenizer()
    token_ids = encode_speaker_text(tokenizer, text)
    assert token_ids == [
        tokenizer.convert_tokens_to_ids(token) for token in expected_tokens
    ]


@pytest.mark.parametrize(
    "sample_rate,sample_count,cap_seconds,expected_patches",
    [
        (24000, 3840, None, 1),
        (24000, 3841, None, 2),
        (16000, 2561, None, 2),
        (44100, 7056, None, 1),
        (44100, 7057, None, 2),
        (16000, 320001, 20.0, 125),
        (16000, 2561, 0.16005, 1),
    ],
)
def test_audio_prompt_slots_follow_resampled_patch_boundaries(
    tmp_path: Path,
    audio_server_url: str,
    monkeypatch: pytest.MonkeyPatch,
    sample_rate: int,
    sample_count: int,
    cap_seconds: float | None,
    expected_patches: int,
) -> None:
    """Clip before resampling so fractional caps cannot add a prompt slot."""
    encoded_audio = io.BytesIO()
    sf.write(
        encoded_audio,
        np.zeros(sample_count, dtype=np.float32),
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )
    audio_bytes = encoded_audio.getvalue()
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(audio_bytes)
    audio_uri = f"data:audio/wav;base64,{base64.b64encode(audio_bytes).decode('ascii')}"
    permissive_audio_uri = (
        f"data:audio/wav,{base64.b64encode(audio_bytes).decode('ascii')}\n"
    )
    audio_url = f"{audio_server_url}/reference.wav"
    monkeypatch.setenv("REQUEST_TIMEOUT", "1")
    with patch(
        "sglang_omni.models.firered_tts3.task_prompts.urlopen", wraps=urlopen
    ) as open_audio_url:
        for reference in (str(audio_path), audio_uri, permissive_audio_uri, audio_url):
            assert (
                count_audio_patches(reference, cap_seconds=cap_seconds)
                == expected_patches
            )
        open_audio_url.assert_called_once_with(audio_url, timeout=60)


@pytest.mark.parametrize(
    "source,expected_error",
    [("http", HTTPError), ("file", sf.LibsndfileError), ("data_uri", binascii.Error)],
)
def test_audio_prompt_count_preserves_source_errors(
    tmp_path: Path,
    audio_server_url: str,
    source: Literal["http", "file", "data_uri"],
    expected_error: type[Exception],
) -> None:
    """Source failures retain their HTTP, SoundFile, or base64 exception type."""
    if source == "http":
        audio_source = f"{audio_server_url}/missing.wav"
    elif source == "file":
        audio_path = tmp_path / "invalid.wav"
        audio_path.write_bytes(b"invalid audio")
        audio_source = str(audio_path)
    else:
        audio_source = "data:audio/wav;base64,YQ"
    with pytest.raises(expected_error) as failure:
        count_audio_patches(audio_source, cap_seconds=None)
    assert type(failure.value) is expected_error


def test_compose_acoustic_instruction_templates() -> None:
    assert (
        resolve_acoustic_instruction(
            None, {}, {"acoustic_attribute": "Speed", "acoustic_value": 0.8}
        )
        == "adjust the speed to 0.8x"
    )
    assert (
        resolve_acoustic_instruction(
            None, {}, {"acoustic_attribute": "Volume", "acoustic_value": 1.5}
        )
        == "adjust the volume to 1.5"
    )
    assert (
        resolve_acoustic_instruction(
            None, {}, {"acoustic_attribute": "Pitch", "acoustic_value": 2}
        )
        == "shift the pitch by 2 steps"
    )
    assert (
        resolve_acoustic_instruction(
            None, {}, {"acoustic_attribute": "Pitch", "acoustic_value": -1}
        )
        == "shift the pitch by -1 step"
    )


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
) -> ReferenceEncodeService:
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

    first = service.get_or_encode((str(reference_audio_path), None))
    second = service.get_or_encode((str(reference_audio_path), None))
    assert reference_encoder.encode_calls == 1
    assert service.stats()["hits"] == 1
    assert torch.equal(first["prompt_latents"], second["prompt_latents"])

    second["prompt_latents"].fill_(-1.0)
    third = service.get_or_encode((str(reference_audio_path), None))
    assert torch.equal(third["prompt_latents"], first["prompt_latents"])
    service.close()


def test_reference_cache_key_separates_edit_cap(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    uncapped = service.get_or_encode((str(reference_audio_path), None))
    capped = service.get_or_encode((str(reference_audio_path), 30.0))
    assert reference_encoder.encode_calls == 2
    assert service.stats()["entries"] == 2
    assert torch.equal(uncapped["prompt_latents"], torch.ones(1, 2, 4))
    assert torch.equal(capped["prompt_latents"], torch.full((1, 2, 4), 2.0))
    assert torch.equal(
        service.get_or_encode((str(reference_audio_path), None))["prompt_latents"],
        uncapped["prompt_latents"],
    )
    assert torch.equal(
        service.get_or_encode((str(reference_audio_path), 30.0))["prompt_latents"],
        capped["prompt_latents"],
    )
    assert reference_encoder.encode_calls == 2
    service.close()


def test_reference_cache_disabled_encodes_every_call(tmp_path: Path) -> None:
    reference_encoder = ReferenceEncoderStub()
    service = make_reference_service(reference_encoder, cache_enabled=False)
    reference_audio_path = tmp_path / "ref.wav"
    reference_audio_path.write_bytes(b"ref-audio-bytes")

    service.get_or_encode((str(reference_audio_path), None))
    service.get_or_encode((str(reference_audio_path), None))
    assert reference_encoder.encode_calls == 2
    stats = service.stats()
    assert stats["uncacheable"] == 2
    assert stats["hits"] == 0
    assert stats["entries"] == 0
    service.close()
