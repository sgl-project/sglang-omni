# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS's /v1/audio/speech fields, which stub_server's stub_ming_tts
reads as ming_tts_server does, under the CUDA server's rules."""

from __future__ import annotations

import base64
import io
import wave

import pytest
from native_server import HttpResponse, SupervisedServer, silent_wav

# ming_tts_server's sample rate, request body limit and patch limit.
MING_SAMPLE_RATE = 44100
MING_MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_PATCHES = 200
SPEAKER_SAMPLE_RATE = 16000
REFERENCE_SECONDS = 0.5


def wav_url(wav: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav).decode()


def reference_url(
    sample_rate: int = SPEAKER_SAMPLE_RATE,
    seconds: float = REFERENCE_SECONDS,
    channel_count: int = 1,
) -> str:
    return wav_url(silent_wav(round(sample_rate * seconds), sample_rate, channel_count))


def speak(server: SupervisedServer, changes: dict[str, object]) -> HttpResponse:
    return server.post_speech({"input": "你好。", **changes})


@pytest.mark.parametrize(
    ("changes", "patches"),
    [
        ({}, DEFAULT_MAX_PATCHES),
        ({"max_new_tokens": 10}, 10),
        ({"max_new_tokens": 300, "stage_params": {"tts_engine": {"max_decode_steps": 12}}}, 12),
        ({"stage_params": {"tts_engine": {"ming_max_decode_steps": 7}}}, 7),
    ],
)  # fmt: skip
def test_the_patch_limit_is_chosen_as_on_cuda(
    ming_speech_server: SupervisedServer, changes: dict[str, object], patches: int
) -> None:
    response = speak(ming_speech_server, changes)
    assert response.status == 200
    assert response.headers["X-Completion-Tokens"] == str(patches)
    assert response.headers["X-Prompt-Tokens"] == "0"


@pytest.mark.parametrize(
    "changes",
    [
        {"voice": " Default "},
        {"voice": ""},
        {"speaker": "default"},
        {"language": "Auto"},
        {"task_type": "TTS"},
        {"instructions": "  "},
        {"prompt": "dropped, as CreateSpeechRequest drops it"},
        {"references": []},
        {"ref_audio": " ", "ref_text": " "},
        {"x_vector_only_mode": True, "unknown_field": 1},
        {"stage_params": {"audio_decode": {}, "tts_engine": {"do_sample": False}}},
        {"temperature": -1, "stage_params": {"tts_engine": {"temperature": 0}}},
        {"stage_params": {"tts_engine": {"cfg": 0.5, "sigma": 0, "ming_temperature": 0.5}}},
    ],
)  # fmt: skip
def test_what_cuda_accepts_is_accepted(
    ming_speech_server: SupervisedServer, changes: dict[str, object]
) -> None:
    assert speak(ming_speech_server, changes).status == 200


LOGITS_FIELD = "Ming-Omni-TTS does not use logits sampling field "
SEED = (
    "Ming-Omni-TTS seed is currently unsupported because FlowLoss sampling is unseeded"
)
TEMPERATURE = "Ming-Omni-TTS temperature must be >= 0"
REFERENCE_CONTRACT = (
    "Ming-Omni-TTS currently supports only one reference audio with non-empty "
    "reference text"
)


@pytest.mark.parametrize(
    ("changes", "detail"),
    [
        ({"input": " 　\n"}, "Ming-Omni-TTS requires non-empty input text"),
        ({"input": None}, "input must be a string"),
        ({"voice": "alloy"}, 'Ming-Omni-TTS currently supports only the default voice; got "alloy"'),
        ({"voice": None}, "voice must be a string"),
        ({"speed": 1.5}, "Ming-Omni-TTS speed control is currently unsupported; got 1.5"),
        ({"language": "en"}, 'Ming-Omni-TTS language selection is currently unsupported; got "en"'),
        ({"task_type": "VoiceDesign"}, 'Ming-Omni-TTS currently supports only default text-to-speech; got task_type="VoiceDesign"'),
        ({"instructions": "Speak slowly."}, "Ming-Omni-TTS instructions are currently unsupported"),
        ({"seed": 1}, SEED),
        ({"stage_params": {"tts_engine": {"seed": 1}}}, SEED),
        ({"top_p": 0.9}, LOGITS_FIELD + "'top_p'"),
        ({"stage_params": {"tts_engine": {"do_sample": True}}}, LOGITS_FIELD + "'do_sample'"),
        ({"token_count": 10}, "Ming-Omni-TTS field 'token_count' is currently unsupported"),
        ({"initial_codec_chunk_frames": 2}, "Ming-Omni-TTS does not support initial_codec_chunk_frames"),
        ({"references": [{"text": "参考。"}]}, "references are not supported on this server; give ref_audio and ref_text"),
        ({"max_new_tokens": 0}, "Ming-Omni-TTS max_decode_steps must be > 0, got 0"),
        ({"max_new_tokens": 257}, "Ming-Omni-TTS max_decode_steps exceeds serving cap: 257 > 256"),
        ({"max_new_tokens": 1.5}, "max_new_tokens must be an integer"),
        ({"stage_params": {"tts_engine": {"max_decode_steps": 1.5}}}, "Ming-Omni-TTS max_decode_steps must be an integer"),
        ({"temperature": "warm"}, "temperature must be a number"),
        ({"temperature": -1}, TEMPERATURE),
        ({"temperature": 0, "stage_params": {"tts_engine": {"temperature": -1}}}, TEMPERATURE),
        ({"stage_params": {"tts_engine": {"cfg": 1}}}, "Ming-Omni-TTS currently supports only guided-branch cfg values; cfg must be >= 1e-5 and not equal to 1.0"),
        ({"stage_params": {"tts_engine": {"sigma": -1}}}, "Ming-Omni-TTS sigma must be >= 0"),
        ({"stage_params": {"tts_engine": None}}, "stage_params.tts_engine must be an object"),
        ({"stage_params": []}, "stage_params must be an object"),
        ({"ref_text": "参考。"}, "Ming-Omni-TTS reference text requires reference audio"),
        ({"ref_audio": reference_url()}, REFERENCE_CONTRACT),
        ({"ref_audio": "/tmp/reference.wav", "ref_text": "参考。"}, "ref_audio must be a data:audio/wav;base64 URL on this server"),
        ({"ref_audio": "data:audio/wav;base64,A", "ref_text": "参考。"}, "ref_audio must be valid base64"),
        ({"ref_audio": wav_url(b"not a wav"), "ref_text": "参考。"}, "ref_audio: reference audio must be a RIFF/WAVE file"),
        ({"ref_audio": reference_url(channel_count=2), "ref_text": "参考。"}, "ref_audio: Ming-Omni-TTS currently supports only mono reference audio"),
        ({"ref_audio": reference_url(seconds=0.09), "ref_text": "参考。"}, "ref_audio: reference audio must be between 0.1 and 30 seconds long"),
        ({"ref_audio": reference_url(seconds=30.5), "ref_text": "参考。"}, "ref_audio: reference audio must be between 0.1 and 30 seconds long"),
    ],
)  # fmt: skip
def test_what_cuda_rejects_gets_400(
    ming_speech_server: SupervisedServer, changes: dict[str, object], detail: str
) -> None:
    response = speak(ming_speech_server, changes)
    assert response.status == 400
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize("sample_rate", [16000, 24000, 48000])
def test_a_reference_is_resampled_from_its_own_rate(
    ming_speech_server: SupervisedServer, sample_rate: int
) -> None:
    response = speak(
        ming_speech_server,
        {"ref_audio": reference_url(sample_rate), "ref_text": "参考。"},
    )
    assert response.status == 200
    assert response.headers["X-Prompt-Tokens"] == str(
        round(MING_SAMPLE_RATE * REFERENCE_SECONDS)
    )
    with wave.open(io.BytesIO(response.body)) as reader:
        assert reader.getframerate() == MING_SAMPLE_RATE
        assert reader.getnframes() == round(SPEAKER_SAMPLE_RATE * REFERENCE_SECONDS)


def test_a_body_over_the_limit_gets_413(ming_speech_server: SupervisedServer) -> None:
    response = ming_speech_server.post_speech(
        {"input": "x" * MING_MAX_REQUEST_BODY_BYTES}
    )
    assert response.status == 413
    assert response.json() == {
        "detail": f"request body must be at most {MING_MAX_REQUEST_BODY_BYTES} bytes"
    }
