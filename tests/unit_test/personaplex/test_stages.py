# SPDX-License-Identifier: Apache-2.0
"""Preprocessing resolves the caller channel, the role prompt and the voice per request."""

from unittest.mock import Mock

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
from sglang_omni.preprocessing import resource_connector
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_errors import is_bad_request_error

CALLER_SAMPLES = 2000


class FakeTokenizer:
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
    monkeypatch.setattr(stages, "load_text_tokenizer", lambda _: FakeTokenizer())
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
        return PersonaPlexState.from_dict(scheduler.fn(payload).data)

    run.loads = loads
    run.sources = sources
    return run


def test_caller_is_channel_zero_padded_to_whole_frames(preprocess):
    state = preprocess()
    waveform = state.waveform
    assert state.num_samples == CALLER_SAMPLES
    assert waveform.shape[-1] % SAMPLES_PER_FRAME == 0
    assert waveform.shape[-1] == SAMPLES_PER_FRAME * 2
    assert torch.all(waveform[:CALLER_SAMPLES] == 0.5)
    assert torch.all(waveform[CALLER_SAMPLES:] == 0.0)


def test_role_prompt_default_alias_and_empty(preprocess):
    tokenizer = FakeTokenizer()
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


def test_caller_audio_follows_the_server_media_policy(
    preprocess, monkeypatch, tmp_path
):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    monkeypatch.setenv(resource_connector.ALLOWED_LOCAL_MEDIA_PATH_ENV, str(allowed))
    monkeypatch.setattr(resource_connector, "_global_connector", None)

    for inputs in (
        {"audio_path": str(outside)},
        {"messages": [], "audios": [str(outside)]},
    ):
        with pytest.raises(ValueError, match="not within allowed directory"):
            preprocess(inputs=inputs)
    assert preprocess.sources == []


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


def test_whole_reply_decode_is_cut_back_to_the_caller_length(monkeypatch):
    frames, samples_per_frame = 4, SAMPLES_PER_FRAME
    num_samples = 3 * samples_per_frame + 7

    class _Codec:
        device = "cpu"

        def decode(self, codes_BKF):
            return torch.arange(float(codes_BKF.shape[-1] * samples_per_frame)).view(
                1, 1, -1
            )

    monkeypatch.setattr(stages, "load_codec", lambda *a, **k: (_Codec(), "cpu"))
    scheduler = stages.create_code2wav_executor("m")
    state = PersonaPlexState(
        num_samples=num_samples, codes=torch.zeros(frames, 8, dtype=torch.long)
    )
    payload = StagePayload(
        "r", request=OmniRequest(inputs={}, params={}), data=state.to_dict()
    )
    rendered = np.frombuffer(
        scheduler.compute_fn(payload).data["audio_waveform"], dtype=np.float32
    )
    assert rendered.shape[-1] == num_samples
    assert rendered[-1] == num_samples - 1


def test_mimi_encode_replaces_caller_and_voice_waveforms_with_codes(monkeypatch):
    class _Codec:
        def encode(self, waveform_B1T):
            frames = waveform_B1T.shape[-1] // SAMPLES_PER_FRAME
            first = int(waveform_B1T[0, 0, 0])
            return torch.full((1, 8, frames), first, dtype=torch.long)

    monkeypatch.setattr(stages, "load_codec", lambda *a, **k: (_Codec(), "cpu"))
    scheduler = stages.create_mimi_encode_executor("m")

    def run(state):
        payload = StagePayload(
            "r", request=OmniRequest(inputs={}, params={}), data=state.to_dict()
        )
        return PersonaPlexState.from_dict(scheduler.fn(payload).data)

    state = run(
        PersonaPlexState(
            waveform=torch.full((2 * SAMPLES_PER_FRAME,), 1.0),
            voice_waveform=torch.full((3 * SAMPLES_PER_FRAME,), 2.0),
        )
    )
    assert state.user_codes.shape == (2, 8) and torch.all(state.user_codes == 1)
    assert state.voice_codes.shape == (3, 8) and torch.all(state.voice_codes == 2)
    assert state.waveform is None and state.voice_waveform is None

    no_voice = run(PersonaPlexState(waveform=torch.ones(SAMPLES_PER_FRAME)))
    assert no_voice.user_codes.shape == (1, 8) and no_voice.voice_codes is None


@pytest.mark.parametrize(
    "prompt,budget,expected_loads",
    [
        (
            VoicePrompt(
                frames=1,
                embeddings=torch.ones(1, 4),
                tail_codes=torch.ones(1, 2, dtype=torch.long),
            ),
            31,
            4,
        ),
        (
            VoicePrompt(
                frames=1,
                embeddings=torch.ones(1, 4),
                tail_codes=torch.ones(1, 2, dtype=torch.long),
            ),
            32,
            3,
        ),
        (VoicePrompt(frames=1, waveform=torch.ones(4)), 32, 2),
    ],
)
def test_voice_prompt_cache_byte_budget_preserves_request_results(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    prompt: VoicePrompt,
    budget: int,
    expected_loads: int,
) -> None:
    monkeypatch.setattr(stages, "VOICE_PROMPT_CACHE_MAX_BYTES", budget)
    preprocess = request.getfixturevalue("preprocess")
    loader = Mock(return_value=prompt)
    monkeypatch.setattr(stages, "load_voice_prompt", loader)
    results = [preprocess(voice=voice) for voice in ("a", "a", "b", "a")]
    assert loader.call_count == expected_loads
    for state in results:
        assert state.voice_frames == prompt.frames
        for actual, expected in (
            (state.voice_embeddings, prompt.embeddings),
            (state.voice_tail_codes, prompt.tail_codes),
            (state.voice_waveform, prompt.waveform),
        ):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected)
