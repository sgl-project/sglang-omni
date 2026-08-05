# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave

import numpy as np
import pybase64
import pytest

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.preprocessing.transcription import (
    PreparedAudio,
    prepare_audio,
    resolve_audio_source,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int


def _payload(inputs) -> StagePayload:
    return StagePayload(request_id="req", request=OmniRequest(inputs=inputs), data={})


def _wav_bytes(
    num_samples: int = 1600, sample_rate: int = 16000, num_channels: int = 1
) -> bytes:
    rng = np.random.default_rng(1268)
    pcm = (rng.uniform(-0.5, 0.5, num_samples * num_channels) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_resolve_audio_source_prefers_bytes_like_over_path_like() -> None:
    inputs = {"audio_path": "/tmp/a.wav", "audio_bytes": b"wav"}
    assert resolve_audio_source(_payload(inputs)) == b"wav"


def test_resolve_audio_source_key_precedence_within_groups() -> None:
    assert resolve_audio_source(_payload({"bytes": b"b", "file": b"f"})) == b"b"
    assert resolve_audio_source(_payload({"url": "u", "path": "p"})) == "p"


def test_resolve_audio_source_passes_non_dict_inputs_through() -> None:
    assert resolve_audio_source(_payload(b"raw")) == b"raw"
    assert resolve_audio_source(_payload("/tmp/a.wav")) == "/tmp/a.wav"


def test_prepare_audio_derives_duration_and_fingerprint() -> None:
    prepared = prepare_audio(
        _payload({"audio_bytes": _wav_bytes(num_samples=1600)}), source_name="Test"
    )
    assert isinstance(prepared, PreparedAudio)
    assert prepared.sample_rate == 16000
    assert prepared.duration_s == pytest.approx(0.1)
    assert prepared.fingerprint == audio_fingerprint(prepared.waveform)
    assert prepared.fingerprint_int == audio_fingerprint_int(prepared.fingerprint)


def test_prepare_audio_uses_custom_source_resolver() -> None:
    prepared = prepare_audio(
        _payload({"nested": {"clip": _wav_bytes()}}),
        source_name="Test",
        source_resolver=lambda payload: payload.request.inputs["nested"]["clip"],
    )
    assert prepared.duration_s == pytest.approx(0.1)


def test_prepare_audio_enforces_duration_limit_with_custom_message(monkeypatch) -> None:
    fingerprinted = []
    monkeypatch.setattr(
        transcription,
        "audio_fingerprint",
        lambda audio: fingerprinted.append(audio) or "unused",
    )

    with pytest.raises(ValueError, match="model-specific limit text"):
        prepare_audio(
            _payload({"audio_bytes": _wav_bytes(num_samples=16000 * 2)}),
            source_name="Test",
            max_duration_s=1.0,
            max_duration_message="model-specific limit text",
        )
    # over-limit clips are rejected before fingerprinting
    assert fingerprinted == []


def test_prepare_audio_default_limit_message_names_the_model() -> None:
    with pytest.raises(ValueError, match="Test accepts audio up to 1.0 seconds"):
        prepare_audio(
            _payload({"audio_bytes": _wav_bytes(num_samples=16000 * 2)}),
            source_name="Test",
            max_duration_s=1.0,
        )


def test_prepare_audio_surfaces_source_name_in_load_errors() -> None:
    with pytest.raises(ValueError, match="Unsupported Test audio input"):
        prepare_audio(_payload({"audio_bytes": 123}), source_name="Test")


def _source_forms(data: bytes, tmp_path) -> dict[str, object]:
    """Every source form load_audio supported pre-refactor, minus HTTP URLs
    (exercised separately with a stubbed transport)."""
    path = tmp_path / "clip.wav"
    path.write_bytes(data)
    encoded = pybase64.b64encode(data).decode("ascii")
    return {
        "raw_bytes": data,
        "bytearray": bytearray(data),
        "memoryview": memoryview(data),
        "filesystem_path": str(path),
        "file_uri": path.as_uri(),
        "data_uri": f"data:audio/wav;base64,{encoded}",
    }


@pytest.mark.parametrize(
    "form",
    ["raw_bytes", "bytearray", "memoryview", "filesystem_path", "file_uri", "data_uri"],
)
def test_prepare_audio_accepts_every_legacy_source_form(form, tmp_path) -> None:
    data = _wav_bytes()
    reference = prepare_audio(_payload({"audio_bytes": data}), source_name="Test")

    prepared = prepare_audio(
        _payload({"audio_bytes": _source_forms(data, tmp_path)[form]}),
        source_name="Test",
    )

    assert np.array_equal(prepared.waveform, reference.waveform)
    assert prepared.duration_s == reference.duration_s
    assert prepared.fingerprint == reference.fingerprint


def test_prepare_audio_accepts_http_url(monkeypatch) -> None:
    data = _wav_bytes()

    class _Response:
        content = data

        @staticmethod
        def raise_for_status() -> None:
            pass

    monkeypatch.setattr(
        "sglang_omni.utils.audio.httpx.get", lambda url, **kwargs: _Response()
    )
    reference = prepare_audio(_payload({"audio_bytes": data}), source_name="Test")

    prepared = prepare_audio(
        _payload({"url": "https://example.com/clip.wav"}), source_name="Test"
    )

    assert prepared.fingerprint == reference.fingerprint


def test_prepare_audio_resamples_and_downmixes_like_before() -> None:
    # 8 kHz clip: resampled to the 16 kHz target, so duration is preserved
    prepared = prepare_audio(
        _payload({"audio_bytes": _wav_bytes(num_samples=800, sample_rate=8000)}),
        source_name="Test",
    )
    assert prepared.duration_s == pytest.approx(0.1)

    # stereo clip: downmixed to mono at the same length
    prepared = prepare_audio(
        _payload({"audio_bytes": _wav_bytes(num_channels=2)}), source_name="Test"
    )
    assert prepared.waveform.ndim == 1
    assert prepared.duration_s == pytest.approx(0.1)
