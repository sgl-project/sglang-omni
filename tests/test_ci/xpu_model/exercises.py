# SPDX-License-Identifier: Apache-2.0
"""What one request per model family asks for, and what the answer must contain.

Not a test module: the unit lane imports EXERCISES from here to prove every
family in the table has a request shaped for it, and pytest must not collect it.
"""

from __future__ import annotations

import base64
import io
import json
import wave
from array import array
from pathlib import Path

import pytest
import requests

from tests.test_ci.xpu_model.conftest import REQUEST_TIMEOUT, SPEECH_REQUEST_TIMEOUT
from tests.test_ci.xpu_model.model_table import XpuModel

_DATA = Path(__file__).resolve().parents[2] / "data"

REF_AUDIO = _DATA / "query_to_cars.wav"
REF_TEXT = "How many cars are there in the picture?"
REF_KEYWORD = "car"

SPOKEN_TEXT = "The quick brown fox jumps over the lazy dog."
MIN_SPEECH_SECONDS = 1.0

# Every payload below is the one the matching recipe in
# docs/other_platforms/installation_xpu.md uses, and each was confirmed to return
# audible speech from its checkpoint on a B60. They are not interchangeable
# between families: a Qwen3-TTS reference pair sent to dots.tts is ignored, and
# AuK needs its generated length asked for explicitly.
CV_VOICE = "Ryan"
CV_LANGUAGE = "English"
VOICE_DESIGN_INSTRUCTIONS = "A warm, natural young adult voice."
AUK_INSTRUCTIONS = "A warm, relaxed voice."
AUK_GEN_SECONDS = 4
MUSIC_INSTRUCTIONS = (
    "An energetic arena rock anthem with distorted electric guitars, punchy "
    "live drums and a soaring male vocal at 130 BPM, wide stereo image"
)
# Bounded so the request cannot run for as long as the model feels like.
MUSIC_MAX_NEW_TOKENS = 750


def assert_audible_speech(raw: bytes, label: str) -> float:
    """Assert ``raw`` is a decodable WAV holding audible frames; return seconds.

    A broken vocoder still returns 200 with an undecodable container, a header
    with no frames, or frames that are all zeros.
    """
    assert raw, f"{label}: empty response body"
    try:
        with wave.open(io.BytesIO(raw)) as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            width = handle.getsampwidth()
            payload = handle.readframes(frames)
    except wave.Error as exc:
        pytest.fail(f"{label}: response is not a decodable WAV ({exc}); {raw[:64]!r}")

    assert frames > 0, f"{label}: WAV header declares no frames"
    assert rate > 0, f"{label}: WAV declares no sample rate"
    seconds = frames / rate
    assert seconds >= MIN_SPEECH_SECONDS, (
        f"{label}: {seconds:.2f}s of audio is too short for {SPOKEN_TEXT!r}; "
        "the utterance was truncated"
    )

    typecode = {1: "b", 2: "h", 4: "i"}.get(width)
    if typecode is None:  # pragma: no cover - no XPU vocoder emits these
        return seconds
    samples = array(typecode)
    samples.frombytes(payload[: len(payload) - len(payload) % samples.itemsize])
    peak = max((abs(sample) for sample in samples), default=0)
    full_scale = float(1 << (8 * width - 1))
    assert peak / full_scale > 0.01, (
        f"{label}: {seconds:.2f}s of audio peaks at {peak / full_scale:.4f} of "
        "full scale, which is silence"
    )
    return seconds


def message_content(body: dict, label: str) -> str:
    assert body.get("choices"), f"{label}: no choices in response: {body}"
    content = body["choices"][0].get("message", {}).get("content")
    assert content, f"{label}: no assistant content: {json.dumps(body)[:400]}"
    return content


def exercise_transcription(base_url: str, checkpoint: str, model: XpuModel) -> str:
    with REF_AUDIO.open("rb") as handle:
        response = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (REF_AUDIO.name, handle.read(), "audio/wav")},
            data={"model": checkpoint},
            timeout=REQUEST_TIMEOUT,
        )
    assert response.status_code == 200, f"http {response.status_code}: {response.text}"
    heard = response.json().get("text", "")
    assert heard.strip(), f"empty transcript for {REF_TEXT!r}"
    assert REF_KEYWORD in heard.lower(), (
        f"transcript {heard!r} does not contain {REF_KEYWORD!r}, so the decode "
        f"did not recover {REF_TEXT!r}"
    )
    return f"heard {heard.strip()!r}"


def speak(base_url: str, payload: dict, label: str) -> str:
    response = requests.post(
        f"{base_url}/v1/audio/speech",
        json=payload,
        timeout=SPEECH_REQUEST_TIMEOUT,
    )
    assert response.status_code == 200, f"http {response.status_code}: {response.text}"
    seconds = assert_audible_speech(response.content, label)
    return f"{seconds:.2f}s of audible speech"


def exercise_speech_cloned(base_url: str, checkpoint: str, model: XpuModel) -> str:
    return speak(
        base_url,
        {
            "model": checkpoint,
            "input": SPOKEN_TEXT,
            "voice": "default",
            "ref_audio": str(REF_AUDIO),
            "ref_text": REF_TEXT,
            "response_format": "wav",
        },
        model.key,
    )


def exercise_speech_builtin_voice(
    base_url: str, checkpoint: str, model: XpuModel
) -> str:
    return speak(
        base_url,
        {
            "model": checkpoint,
            "input": SPOKEN_TEXT,
            "voice": CV_VOICE,
            "task_type": "CustomVoice",
            "language": CV_LANGUAGE,
            "response_format": "wav",
        },
        model.key,
    )


def exercise_speech_default_voice(
    base_url: str, checkpoint: str, model: XpuModel
) -> str:
    """Families that ship their own voice and take no reference at all."""
    return speak(
        base_url,
        {
            "model": checkpoint,
            "input": SPOKEN_TEXT,
            "voice": "default",
            "response_format": "wav",
        },
        model.key,
    )


def exercise_speech_references(base_url: str, checkpoint: str, model: XpuModel) -> str:
    """dots.tts takes its reference as a list, not as ref_audio/ref_text."""
    return speak(
        base_url,
        {
            "model": checkpoint,
            "input": SPOKEN_TEXT,
            "voice": "default",
            "references": [{"audio_path": str(REF_AUDIO), "text": REF_TEXT}],
            "response_format": "wav",
        },
        model.key,
    )


def exercise_speech_voice_design(
    base_url: str, checkpoint: str, model: XpuModel
) -> str:
    """VoiceDesign builds the voice from a description instead of a sample."""
    return speak(
        base_url,
        {
            "model": checkpoint,
            "input": SPOKEN_TEXT,
            "voice": "default",
            "task_type": "VoiceDesign",
            "instructions": VOICE_DESIGN_INSTRUCTIONS,
            "response_format": "wav",
        },
        model.key,
    )


def exercise_speech_instructions(
    base_url: str, checkpoint: str, model: XpuModel
) -> str:
    """AuK needs a style instruction and is asked for a length explicitly."""
    return speak(
        base_url,
        {
            "model": checkpoint,
            "input": SPOKEN_TEXT,
            "voice": "default",
            "instructions": AUK_INSTRUCTIONS,
            "stage_params": {"auk_engine": {"gen_seconds": AUK_GEN_SECONDS}},
            "response_format": "wav",
        },
        model.key,
    )


def exercise_music(base_url: str, checkpoint: str, model: XpuModel) -> str:
    """Music takes a style prompt rather than a voice, and lyrics as the input."""
    return speak(
        base_url,
        {
            "model": checkpoint,
            "input": "[Chorus]\nWe are the fire that never dies",
            "instructions": MUSIC_INSTRUCTIONS,
            "max_new_tokens": MUSIC_MAX_NEW_TOKENS,
            "response_format": "wav",
        },
        model.key,
    )


def exercise_chat(base_url: str, checkpoint: str, model: XpuModel) -> str:
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": checkpoint,
            "messages": [
                {"role": "user", "content": "Answer with the number only: 17 plus 25?"}
            ],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout=REQUEST_TIMEOUT,
    )
    assert response.status_code == 200, f"http {response.status_code}: {response.text}"
    content = message_content(response.json(), model.key)
    assert "42" in content, f"expected 42 in the answer, got {content.strip()!r}"
    return f"answered {content.strip()[:40]!r}"


def exercise_chat_speech(base_url: str, checkpoint: str, model: XpuModel) -> str:
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": checkpoint,
            "messages": [{"role": "user", "content": f"Say exactly: {SPOKEN_TEXT}"}],
            "modalities": ["text", "audio"],
            "audio": {"format": "wav"},
            "max_tokens": 64,
        },
        timeout=SPEECH_REQUEST_TIMEOUT,
    )
    assert response.status_code == 200, f"http {response.status_code}: {response.text}"
    body = response.json()
    audio = body["choices"][0].get("message", {}).get("audio") or {}
    data = audio.get("data")
    assert data, f"no audio in response: {json.dumps(body)[:400]}"
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    seconds = assert_audible_speech(base64.b64decode(data), model.key)
    return f"talker and vocoder produced {seconds:.2f}s of audible speech"


EXERCISES = {
    "transcription": exercise_transcription,
    "speech_cloned": exercise_speech_cloned,
    "speech_builtin_voice": exercise_speech_builtin_voice,
    "speech_default_voice": exercise_speech_default_voice,
    "speech_references": exercise_speech_references,
    "speech_voice_design": exercise_speech_voice_design,
    "speech_instructions": exercise_speech_instructions,
    "music": exercise_music,
    "chat": exercise_chat,
    "chat_speech": exercise_chat_speech,
}
