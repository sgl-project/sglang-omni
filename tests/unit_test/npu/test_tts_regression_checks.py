# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the audio assertions used by NPU serving regressions."""

import io
import struct
import wave

import pytest

from tests.test_model.test_npu_tts import _payload, _validate_pcm, _validate_wav


def _wav(pcm, sample_rate=24000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def test_valid_pcm_and_wav():
    pcm = struct.pack("<h", 1000) * 24000
    assert _validate_pcm(pcm) == 1.0
    assert _validate_wav(_wav(pcm)) == 1.0


@pytest.mark.parametrize("pcm", [b"", b"\x01", b"\x00\x00" * 24000])
def test_reject_empty_truncated_or_silent_pcm(pcm):
    with pytest.raises(AssertionError):
        _validate_pcm(pcm)


def test_reject_truncated_wav():
    data = _wav(struct.pack("<h", 1000) * 24000)
    with pytest.raises(AssertionError, match="Truncated"):
        _validate_wav(data[:-2])


def test_reject_wrong_sample_rate():
    with pytest.raises(AssertionError):
        _validate_wav(_wav(struct.pack("<h", 1000) * 24000, sample_rate=16000))


def test_base_reference_is_distinct_from_target_text(monkeypatch, tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(_wav(struct.pack("<h", 1000) * 24000))
    monkeypatch.setenv("OMNI_NPU_TTS_REFERENCE", str(reference))
    monkeypatch.setenv("OMNI_NPU_TTS_REFERENCE_TEXT", "The reference recording.")
    payload = _payload("Base")
    assert payload["references"][0]["text"] != payload["input"]
    assert "voice" not in payload


def test_custom_voice_and_voice_design_fields():
    assert _payload("CustomVoice", "Chinese")["voice"] == "Vivian"
    assert _payload("VoiceDesign")["instructions"]
    assert "references" not in _payload("VoiceDesign")
