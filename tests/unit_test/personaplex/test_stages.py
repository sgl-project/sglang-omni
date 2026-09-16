# SPDX-License-Identifier: Apache-2.0
"""Preprocessing resolves the caller channel, the role prompt and the voice per request."""

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex import stages
from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.prompts import (
    DEFAULT_TEXT_PROMPT,
    DEFAULT_VOICE,
    VoicePrompt,
    tokenize_text_prompt,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_errors import is_bad_request_error

CALLER_SAMPLES = 2000


class _Tokenizer:
    def encode(self, text):
        return [len(word) for word in text.split()]


@pytest.fixture
def preprocess(monkeypatch, tmp_path):
    loads = []

    def load_voice_prompt(path, *, load_audio):
        loads.append(path)
        return VoicePrompt(
            frames=3,
            embeddings=torch.zeros(2, 4),
            tail_codes=torch.zeros(2, 8, dtype=torch.long),
        )

    caller = np.stack(
        [np.full(CALLER_SAMPLES, 0.5), np.full(CALLER_SAMPLES, -1.0)]
    ).astype(np.float32)
    monkeypatch.setattr(stages, "load_text_tokenizer", lambda _: _Tokenizer())
    sources = []

    def load_audio(source, **_):
        sources.append(source)
        return caller

    monkeypatch.setattr(stages, "load_audio", load_audio)
    monkeypatch.setattr(stages, "resolve_voice_path", lambda _, voice: f"{voice}.pt")
    monkeypatch.setattr(stages, "load_voice_prompt", load_voice_prompt)
    scheduler = stages.create_preprocessing_executor(str(tmp_path))

    def run(inputs=None, **params):
        payload = StagePayload(
            "r",
            request=OmniRequest(
                inputs=inputs or {"audio_path": "caller.wav"}, params=params
            ),
            data={},
        )
        return PersonaPlexState.from_dict(scheduler._fn(payload).data)

    run.loads = loads
    run.sources = sources
    return run


def test_caller_is_channel_zero_padded_to_whole_frames(preprocess):
    state = preprocess()
    waveform = state.waveform
    assert waveform.shape[-1] % SAMPLES_PER_FRAME == 0
    assert waveform.shape[-1] == SAMPLES_PER_FRAME * 2
    assert torch.all(waveform[:CALLER_SAMPLES] == 0.5)
    assert torch.all(waveform[CALLER_SAMPLES:] == 0.0)


def test_role_prompt_default_alias_and_empty(preprocess):
    tokenizer = _Tokenizer()
    assert preprocess().text_prompt_ids == tokenize_text_prompt(
        tokenizer, DEFAULT_TEXT_PROMPT
    )
    assert preprocess(text_prompt="Be brief").text_prompt_ids == [8, 2, 5, 8]
    assert preprocess(instructions="Be brief").text_prompt_ids == [8, 2, 5, 8]
    assert preprocess(text_prompt="").text_prompt_ids == []


def test_voice_default_empty_and_cached(preprocess):
    state = preprocess()
    assert preprocess.loads == [f"{DEFAULT_VOICE}.pt"]
    assert state.voice_frames == 3
    assert state.voice_embeddings.shape == (2, 4)

    preprocess()
    assert preprocess.loads == [f"{DEFAULT_VOICE}.pt"]

    state = preprocess(voice="")
    assert state.voice_frames == 0
    assert state.voice_embeddings is None and state.voice_tail_codes is None


def test_preprocessing_stage_params_override_top_level(preprocess):
    state = preprocess(
        voice="NATF2",
        text_prompt="Be brief",
        stage_params={"preprocessing": {"voice": "NATM1", "text_prompt": "Be kind"}},
    )
    assert preprocess.loads == ["NATM1.pt"]
    assert state.text_prompt_ids == [8, 2, 4, 8]


def test_chat_completions_audios_supply_the_caller(preprocess):
    preprocess(inputs={"messages": [], "audios": ["data:audio/wav;base64,AAAA"]})
    assert preprocess.sources == ["data:audio/wav;base64,AAAA"]
    with pytest.raises(ValueError, match="one caller recording") as error:
        preprocess(inputs={"messages": [], "audios": ["a.wav", "b.wav"]})
    assert is_bad_request_error(error.value)


def test_engine_context_length_reaches_the_builder(monkeypatch):
    built = {}

    class Builder:
        def __init__(self, *, max_running_requests, context_length):
            built["context_length"] = context_length

        def build(self, model_path, **kwargs):
            built["overrides"] = kwargs["server_args_overrides"]

    monkeypatch.setattr(stages, "PersonaPlexEngineBuilder", Builder)
    stages.create_lm_executor("m", server_args_overrides={"context_length": 16384})
    assert built["context_length"] == 16384
    assert built["overrides"] == {"context_length": 16384}

    stages.create_lm_executor("m", context_length=4096)
    assert built["context_length"] == 4096 and built["overrides"] is None
