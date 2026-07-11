# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
import torch

import sglang_omni.models.moss_transcribe_diarize.request_builders as request_builders
from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_TRANSCRIBE_DIARIZE_PROMPT,
    make_moss_transcribe_diarize_scheduler_adapters,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, OmniRequest, StagePayload


def _wav_bytes(num_samples: int = 1600, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_samples)
    return buffer.getvalue()


class FakeTokenizer:
    vocab_size = 200000
    eos_token_id = 151645

    def __init__(self) -> None:
        self._ids = {
            "<|audio_start|>": 151669,
            "<|audio_pad|>": 151671,
            "<|audio_end|>": 151670,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        return "".join(str(token_id) for token_id in token_ids)


class FakeProcessor:
    audio_token_id = 151671

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.messages = None

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ):
        del tokenize, add_generation_prompt
        self.messages = messages
        return "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>prompt"

    def __call__(self, *, text: str, audio, return_tensors: str, max_length: int):
        del text, audio, return_tensors, max_length
        return {
            "input_ids": torch.tensor(
                [[10, 151669, 151671, 151671, 20, 151671, 151670, 11]],
                dtype=torch.long,
            ),
            "input_features": torch.zeros((1, 80, 3000), dtype=torch.float32),
            "audio_feature_lengths": torch.tensor([3], dtype=torch.long),
            "audio_chunk_mapping": torch.tensor([0], dtype=torch.long),
        }


def _payload(
    prompt: str | None = None,
    params: dict | None = None,
    metadata: dict | None = None,
) -> StagePayload:
    request_params = dict(params or {})
    if prompt is not None:
        request_params["prompt"] = prompt
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={"audio_bytes": _wav_bytes()},
            params=request_params,
            metadata=metadata or {"model": "moss-transcribe-diarize"},
        ),
        data={},
    )


def _payload_with_inputs(inputs, *, metadata: dict | None = None) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=inputs,
            params={},
            metadata=metadata or {"model": "moss-transcribe-diarize"},
        ),
        data={},
    )


def _request_builder(processor: FakeProcessor | None = None):
    processor = processor or FakeProcessor()
    request_builder, _ = make_moss_transcribe_diarize_scheduler_adapters(
        processor=processor,
        tokenizer=processor.tokenizer,
        max_new_tokens=32,
    )
    return request_builder


def test_request_builder_replaces_audio_tokens_with_item_pad_value() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    data = request_builder(_payload())

    input_ids = data.input_ids.tolist()
    audio_token_id = processor.audio_token_id
    assert audio_token_id not in input_ids
    audio_item = data.req.multimodal_inputs.mm_items[0]
    assert audio_item.offsets == [(2, 3), (5, 5)]
    assert input_ids[2] == audio_item.pad_value
    assert input_ids[3] == audio_item.pad_value
    assert input_ids[5] == audio_item.pad_value
    assert input_ids[4] == 20
    assert data.req.sampling_params.max_new_tokens == 32


def test_request_builder_uses_moss_sampling_defaults() -> None:
    request_builder = _request_builder()

    data = request_builder(_payload())
    sampling_params = data.req.sampling_params

    assert data.temperature == DEFAULT_TEMPERATURE
    assert data.top_p == DEFAULT_TOP_P
    assert data.top_k == DEFAULT_TOP_K
    if DEFAULT_TEMPERATURE == 0.0:
        # SGLang encodes greedy sampling as temperature=1.0 with top_k=1.
        assert sampling_params.temperature == 1.0
        assert sampling_params.top_k == 1
    else:
        assert sampling_params.temperature == DEFAULT_TEMPERATURE
        assert sampling_params.top_p == DEFAULT_TOP_P
        assert sampling_params.top_k == DEFAULT_TOP_K


def test_request_builder_preserves_sampling_overrides() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 25,
            },
            metadata={
                "model": "moss-transcribe-diarize",
                EXPLICIT_GENERATION_PARAMS_KEY: ["temperature", "top_p", "top_k"],
            },
        )
    )

    assert data.temperature == 0.0
    assert data.top_p == 0.9
    assert data.top_k == 25


def test_request_builder_ignores_implicit_client_sampling_defaults() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            },
        )
    )

    assert data.temperature == DEFAULT_TEMPERATURE
    assert data.top_p == DEFAULT_TOP_P
    assert data.top_k == DEFAULT_TOP_K


def test_request_builder_preserves_explicit_default_valued_overrides() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            },
            metadata={
                "model": "moss-transcribe-diarize",
                EXPLICIT_GENERATION_PARAMS_KEY: ["temperature", "top_p", "top_k"],
            },
        )
    )

    assert data.temperature == 1.0
    assert data.top_p == 1.0
    assert data.top_k == -1


def test_request_builder_ignores_openai_transcription_temperature_default() -> None:
    request_builder = _request_builder()

    data = request_builder(
        _payload(
            params={"temperature": 0.0},
            metadata={"model": "moss-transcribe-diarize"},
        )
    )

    assert data.temperature == DEFAULT_TEMPERATURE


def test_request_builder_uses_default_prompt_for_empty_transcription_prompt() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    request_builder(_payload(prompt=""))

    assert processor.messages is not None
    assert (
        processor.messages[0]["content"][1]["text"] == DEFAULT_TRANSCRIBE_DIARIZE_PROMPT
    )


def test_request_builder_rejects_direct_waveform_list() -> None:
    request_builder = _request_builder()

    with pytest.raises(ValueError, match="Unsupported MOSS-Transcribe-Diarize"):
        request_builder(_payload_with_inputs({"audio_data": [0.0, 0.1, -0.1, 0.0]}))


def test_request_builder_accepts_single_audio_from_audios_list() -> None:
    request_builder = _request_builder()

    data = request_builder(_payload_with_inputs({"audios": [_wav_bytes()]}))

    assert data.audio_duration_s == 0.1
    assert len(data.req.multimodal_inputs.mm_items) == 1


def test_request_builder_rejects_multiple_audios() -> None:
    request_builder = _request_builder()

    with pytest.raises(ValueError, match="exactly one audio"):
        request_builder(_payload_with_inputs({"audios": [_wav_bytes(), _wav_bytes()]}))


def test_request_builder_uses_default_prompt_for_bare_string_audio_source(
    monkeypatch,
) -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)
    monkeypatch.setattr(
        request_builders,
        "_load_audio",
        lambda source: np.zeros(1600, dtype=np.float32),
    )

    request_builder(_payload_with_inputs("/tmp/audio.wav"))

    assert processor.messages is not None
    assert (
        processor.messages[0]["content"][1]["text"] == DEFAULT_TRANSCRIBE_DIARIZE_PROMPT
    )


def test_request_builder_uses_string_prompt_when_audio_is_supplied_separately() -> None:
    processor = FakeProcessor()
    request_builder = _request_builder(processor)

    request_builder(
        _payload_with_inputs(
            "custom diarization prompt",
            metadata={
                "model": "moss-transcribe-diarize",
                "audios": [_wav_bytes()],
            },
        )
    )

    assert processor.messages is not None
    assert processor.messages[0]["content"][1]["text"] == "custom diarization prompt"
