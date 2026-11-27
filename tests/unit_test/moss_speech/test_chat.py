# SPDX-License-Identifier: Apache-2.0
"""Chat inputs, dual-channel rows and all four output combinations."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile
import torch
from fastapi.testclient import TestClient

from sglang_omni.models.moss_speech.chat import chat_rows
from sglang_omni.models.moss_speech.fsm import AUDIO_PAD, MODALITY_PAD
from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechState,
    RequestValidationError,
    build_result,
    build_sglang_moss_request,
    preprocess_request,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_api import create_app
from tests.unit_test.fixtures.moss_speech import (
    SpeechClient,
    TextTokenizer,
    audio_reference,
    make_codec,
)


def test_public_speech_and_chat_return_the_requested_modality() -> None:
    with TestClient(
        create_app(SpeechClient(make_codec()), model_name="moss-speech")
    ) as http:
        response = http.post(
            "/v1/audio/speech",
            json={
                "input": "Hello",
                "max_new_tokens": 3,
                "seed": 7,
                "ref_audio": audio_reference(),
            },
        )
        assert response.status_code == 200, response.text
        assert response.content[:4] == b"RIFF" and len(response.content) > 44
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "four"


def test_speech_input_keeps_first_channel_and_native_rate(tmp_path: Path) -> None:
    """Stereo speech input must reach token encoding at its original rate and first channel."""
    samples = np.random.default_rng(42).uniform(-0.2, 0.2, (4410, 2)).astype(np.float32)
    path = tmp_path / "speech.wav"
    soundfile.write(path, samples, 44100, subtype="FLOAT")
    codec = make_codec()
    codec.encode_tokens = Mock(wraps=codec.encode_tokens)
    codec.encode_audio(str(path))
    waveform, sample_rate = codec.encode_tokens.call_args.args
    assert sample_rate == 44100
    torch.testing.assert_close(
        waveform, torch.from_numpy(samples.T[:1]), rtol=0, atol=0
    )


@pytest.mark.parametrize("speech_input", [False, True])
@pytest.mark.parametrize("output_modality", ["text", "audio"])
def test_chat_preserves_both_token_channels_and_output_modality(
    speech_input: bool, output_modality: str
) -> None:
    """Text and audio prompts must reach one AR engine without losing codec rows."""
    encoder = Mock()
    encoder.encode_audio.return_value = [3, 17]
    content = (
        [{"type": "audio_url", "audio_url": {"url": "speech.wav"}}]
        if speech_input
        else "What is two plus two?"
    )
    payload = StagePayload(
        "chat",
        OmniRequest(
            inputs=[{"role": "user", "content": content}],
            params={"temperature": 0, "max_new_tokens": 128},
            metadata=(
                {
                    "output_modalities": [output_modality],
                    "audio_config": {"ref_audio": "voice.wav", "format": "wav"},
                }
                if output_modality == "audio"
                else {"output_modalities": ["text"]}
            ),
        ),
        {},
    )
    prepared = preprocess_request(payload, TextTokenizer(), encoder)
    state = MossSpeechState.from_dict(prepared.data)
    native = build_sglang_moss_request(prepared)
    assert state.output_modality == output_modality
    assert native.prompt_rows.shape == (state.prompt_token_count, 2)
    if speech_input:
        encoder.encode_audio.assert_called_once_with("speech.wav")
        assert native.prompt_rows[
            native.prompt_rows[:, 0] == MODALITY_PAD
        ].tolist() == [[MODALITY_PAD, 3], [MODALITY_PAD, 17], [MODALITY_PAD, 16384]]
    else:
        encoder.encode_audio.assert_not_called()
        assert (
            native.prompt_rows[:, 1].tolist() == [AUDIO_PAD] * state.prompt_token_count
        )
    native.output_rows = [
        (102, AUDIO_PAD),
        (111, AUDIO_PAD),
        (117, AUDIO_PAD),
        (114, AUDIO_PAD),
    ]
    native.finish_reason = "length"
    result = build_result(native)
    assert not MossSpeechState.from_dict(result.data).audio_token_ids
    if output_modality == "text":
        codec = make_codec()
        codec.decoder = None
        assert codec.decode(result, tokenizer=TextTokenizer()).data["text"] == "four"
    else:
        assert MossSpeechState.from_dict(result.data).reference_audio == "voice.wav"


@pytest.mark.parametrize("output_modality", ["text", "audio"])
def test_missing_system_matches_checkpoint_segment_order(output_modality: str) -> None:
    """The checkpoint appends its default system segment after supplied turns."""
    rows = chat_rows(
        [{"role": "user", "content": "Hello"}], TextTokenizer(), None, output_modality
    )
    text = bytes(row[0] for row in rows).decode()
    expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>system\nYou are a helpful assistant. Respond with "
    expected += "spoken" if output_modality == "audio" else "text"
    expected += " outputs.<|im_end|>\n<|im_start|>assistant\n"
    expected += "<|object_ref_start|>" if output_modality == "audio" else ""
    assert text == expected


@pytest.mark.parametrize("speech_input", [False, True])
@pytest.mark.parametrize("output_modality", ["text", "audio"])
def test_chat_http_supports_all_four_modalities(
    speech_input: bool, output_modality: str
) -> None:
    """Chat audio inputs and text outputs previously failed at the TTS-only boundary."""
    reference = audio_reference()
    content = (
        [
            {
                "type": "input_audio",
                "input_audio": {"data": reference.split(",", 1)[1], "format": "wav"},
            }
        ]
        if speech_input
        else "What is two plus two?"
    )
    body = {
        "model": "moss-speech",
        "messages": [{"role": "user", "content": content}],
        "modalities": [output_modality],
        "stream": False,
        "max_tokens": 8,
    }
    if output_modality == "audio":
        body["audio"] = {"ref_audio": reference, "format": "wav"}
    else:
        pass
    with TestClient(
        create_app(SpeechClient(make_codec()), model_name="moss-speech")
    ) as http:
        response = http.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    if output_modality == "text":
        assert message["content"] == "four"
    else:
        assert message["audio"]["data"]


@pytest.mark.parametrize("failure", ["missing_reference", "conflicting_seed"])
def test_invalid_chat_is_rejected_before_audio_encoding(failure: str) -> None:
    encoder = Mock()
    encoder.encode_audio.return_value = [3, 17]
    audio_config = (
        {} if failure == "missing_reference" else {"ref_audio": "voice.wav", "seed": 2}
    )
    payload = StagePayload(
        "invalid-chat",
        OmniRequest(
            inputs=[
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": "unused.wav"}}
                    ],
                }
            ],
            params={"seed": 1},
            metadata={"output_modalities": ["audio"], "audio_config": audio_config},
        ),
        {},
    )
    message = (
        "Provide ref_audio"
        if failure == "missing_reference"
        else "conflicting speech seeds"
    )
    with pytest.raises(RequestValidationError, match=message):
        preprocess_request(payload, TextTokenizer(), encoder)
    encoder.encode_audio.assert_not_called()
