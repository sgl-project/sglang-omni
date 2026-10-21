# SPDX-License-Identifier: Apache-2.0
"""Preprocessing resolves the caller channel, the role prompt and the voice per request."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang_omni.admission import REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.sources import patches_from_stages_mapping
from sglang_omni.models.personaplex import stages
from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.config import (
    PersonaPlexPipelineConfig,
    PersonaPlexSessionPipelineConfig,
)
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
        def __init__(
            self, *, max_running_requests, context_length, enable_async_decode
        ):
            built["context_length"] = context_length

        def build(self, model_path, **kwargs):
            built["overrides"] = kwargs["server_args_overrides"]

    monkeypatch.setattr(stages, "PersonaPlexEngineBuilder", Builder)
    stages.create_lm_executor(
        "m", enable_async_decode=False, server_args_overrides={"context_length": 16384}
    )
    assert built["context_length"] == 16384
    assert built["overrides"] == {"context_length": 16384}

    stages.create_lm_executor("m", enable_async_decode=False, context_length=4096)
    assert built["context_length"] == 4096 and built["overrides"] is None


def test_realtime_engine_decodes_synchronously(monkeypatch, tmp_path):
    built = {}

    class Builder:
        def __init__(self, *, tokenizer, context_length, enable_async_decode):
            built["enable_async_decode"] = enable_async_decode

        def build(self, model_path, **kwargs):
            pass

    monkeypatch.setattr(stages, "load_text_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(stages, "PersonaPlexSessionEngineBuilder", Builder)
    stages.create_session_lm_executor(str(tmp_path))
    assert built["enable_async_decode"] is False


def test_realtime_session_stages_hold_max_sessions_calls_and_run_codecs_serially(
    monkeypatch, tmp_path
):
    loaded_modes = []

    def load_codec(*args, mode, **kwargs):
        loaded_modes.append(mode)
        return Mock(), "cpu"

    monkeypatch.setattr(stages, "load_text_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(stages, "load_codec", load_codec)
    decode_slots = Mock()
    monkeypatch.setattr(stages, "MimiDecodeSlots", decode_slots)
    prompt = stages.create_session_preprocessing_executor(
        str(tmp_path), max_open_sessions=3
    )
    codecs = [
        stages.create_session_mimi_encode_executor(str(tmp_path), max_open_sessions=3),
        stages.create_session_code2wav_executor(str(tmp_path), max_open_sessions=3),
    ]
    for scheduler in [prompt, *codecs]:
        assert scheduler.max_open_sessions == 3
    assert all(scheduler.max_concurrency == 1 for scheduler in codecs)
    assert decode_slots.call_args.kwargs["num_slots"] == 3
    assert loaded_modes == ["encode", "decode"]


def test_a_call_follows_the_realtime_stages_and_streams_nothing():
    config = PersonaPlexSessionPipelineConfig(model_path="personaplex")
    assert config.session_route() == ["preprocessing", "mimi_encode", "lm", "code2wav"]
    assert config.session_graph() is None
    assert not any(stage.stream_to for stage in config.stages)


def test_realtime_factories_get_no_offline_factory_fields_after_an_override():
    config = PersonaPlexSessionPipelineConfig(model_path="personaplex")
    patches = patches_from_stages_mapping(
        {"lm": {"engine": {"mem_fraction_static": 0.5}}}, type(config), ["lm"]
    )
    resolved = ConfigResolver(config).resolve(patches).config
    lm_args = resolve_stage_factory_args(resolved.stage_named("lm"), resolved)
    code2wav_args = resolve_stage_factory_args(
        resolved.stage_named("code2wav"), resolved
    )
    assert "enable_async_decode" not in lm_args
    assert "chunk_ramp_frames" not in code2wav_args


def test_every_realtime_stage_holds_max_sessions_calls():
    assert PersonaPlexSessionPipelineConfig(model_path="personaplex").max_sessions == 1
    config = PersonaPlexSessionPipelineConfig(model_path="personaplex", max_sessions=3)
    for name in ["preprocessing", "mimi_encode", "code2wav"]:
        assert config.stage_factory_kwargs(name) == {"max_open_sessions": 3}
    assert config.stage_factory_kwargs("lm") == {
        "server_args_overrides": {
            "max_running_requests": 3 + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
        }
    }


def resolve_session_config(
    model_path: str = "personaplex", **stages_mapping
) -> PersonaPlexSessionPipelineConfig:
    config = PersonaPlexSessionPipelineConfig(model_path=model_path, max_sessions=3)
    patches = patches_from_stages_mapping(
        stages_mapping, type(config), list(stages_mapping)
    )
    return ConfigResolver(config).resolve(patches).config


def test_session_stage_settings_reach_their_factories(monkeypatch, tmp_path):
    monkeypatch.setattr(stages, "load_codec", lambda *a, **k: (Mock(), "cpu"))
    decode_slots = Mock()
    monkeypatch.setattr(stages, "MimiDecodeSlots", decode_slots)
    cache = Mock()
    monkeypatch.setattr(stages, "StageOutputCache", cache)
    monkeypatch.setattr(stages, "load_text_tokenizer", lambda _: FakeTokenizer())
    config = resolve_session_config(
        str(tmp_path),
        preprocessing={"factory": {"voice_prompt_cache_bytes": 1024}},
        mimi_encode={"factory": {"max_concurrency": 2}},
        code2wav={"factory": {"max_graph_frames": 0}},
        lm={"engine": {"max_running_requests": 8}},
    )
    stage = config.stage_named("mimi_encode")
    encode = stages.create_session_mimi_encode_executor(
        **resolve_stage_factory_args(stage, config)
    )
    assert encode.max_concurrency == 2
    stage = config.stage_named("preprocessing")
    stages.create_session_preprocessing_executor(
        **resolve_stage_factory_args(stage, config)
    )
    assert cache.call_args.kwargs["max_bytes"] == 1024
    stage = config.stage_named("code2wav")
    stages.create_session_code2wav_executor(**resolve_stage_factory_args(stage, config))
    assert decode_slots.call_args.kwargs["max_graph_frames"] == 0
    lm_args = resolve_stage_factory_args(config.stage_named("lm"), config)
    assert lm_args["server_args_overrides"]["max_running_requests"] == 8

    config = resolve_session_config(mimi_encode={"factory": {"unknown": 1}})
    with pytest.raises(ValueError, match="does not accept"):
        resolve_stage_factory_args(config.stage_named("mimi_encode"), config)


def test_session_lm_slots_below_max_sessions_plus_one_are_rejected():
    with pytest.raises(ValueError, match="max_running_requests=3"):
        resolve_session_config(lm={"engine": {"max_running_requests": 3}})


def test_whole_reply_decode_is_cut_back_to_the_caller_length(monkeypatch):
    frames, samples_per_frame = 4, SAMPLES_PER_FRAME
    num_samples = 3 * samples_per_frame + 7

    class _Codec:
        device = torch.device("cpu")

        def decode(self, codes_BKF):
            return torch.arange(float(codes_BKF.shape[-1] * samples_per_frame)).view(
                1, 1, -1
            )

    monkeypatch.setattr(stages, "load_codec", lambda *a, **k: (_Codec(), "cpu"))
    built_slots = {}

    def build_slots(codec, **slot_kwargs):
        # The streaming slots never run here: the whole reply decodes at once.
        built_slots.update(slot_kwargs)
        return SimpleNamespace(
            codec=codec,
            max_batch_size=slot_kwargs["max_batch_size"],
            max_chunk_frames=slot_kwargs["max_chunk_frames"],
        )

    monkeypatch.setattr(stages, "MimiDecodeSlots", build_slots)
    scheduler = stages.create_code2wav_executor(
        "m",
        num_decode_slots=4,
        max_batch_size=2,
        max_graph_frames=1,
        chunk_ramp_frames=[1, 4],
    )
    assert built_slots == {
        "num_slots": 4,
        "max_batch_size": 2,
        "max_graph_frames": 1,
        "max_chunk_frames": 4,
    }
    assert scheduler.max_batch_size == 2
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


def test_selective_codec_stages_preserve_caller_voice_and_reply(
    random_codec: MimiCodec, mimi_checkpoint: Path
) -> None:
    encoder = stages.create_mimi_encode_executor(
        str(mimi_checkpoint.parent), device="cpu"
    )
    decoder = stages.create_code2wav_executor(
        str(mimi_checkpoint.parent),
        device="cpu",
        num_decode_slots=1,
        max_batch_size=1,
        max_graph_frames=0,
        chunk_ramp_frames=[1, 4],
    )
    caller_waveform = torch.randn(2 * SAMPLES_PER_FRAME)
    voice_waveform = torch.randn(3 * SAMPLES_PER_FRAME)
    caller_codes = random_codec.encode(caller_waveform.view(1, 1, -1))
    voice_codes = random_codec.encode(voice_waveform.view(1, 1, -1))
    caller_samples = SAMPLES_PER_FRAME + 7
    payload = StagePayload(
        "selective-mimi",
        request=OmniRequest(inputs={}, params={}),
        data=PersonaPlexState(
            waveform=caller_waveform,
            voice_waveform=voice_waveform,
            num_samples=caller_samples,
        ).to_dict(),
    )
    encoded = encoder.fn(payload)
    encoded_state = PersonaPlexState.from_dict(encoded.data)
    torch.testing.assert_close(encoded_state.user_codes, caller_codes[0].T)
    torch.testing.assert_close(encoded_state.voice_codes, voice_codes[0].T)
    encoded_state.codes = encoded_state.user_codes
    encoded.data = encoded_state.to_dict()
    actual_waveform = np.frombuffer(
        decoder.compute_fn(encoded).data["audio_waveform"], dtype=np.float32
    )
    expected_waveform = random_codec.decode(caller_codes)[0, 0, :caller_samples]
    torch.testing.assert_close(
        torch.from_numpy(actual_waveform.copy()), expected_waveform, atol=0, rtol=0
    )


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


@pytest.mark.parametrize(
    "chunk_ramp_frames,max_graph_frames,expected_graph_frames",
    [
        ([1, 4], None, 4),
        ([1, 16], None, 16),
        ([1, 8, 2], None, 8),
        ([1, 16], 0, 0),
        ([1, 16], 2, 2),
        ([1, 4], 16, 16),
    ],
)
def test_code2wav_graph_limit_follows_resolved_configuration(
    monkeypatch: pytest.MonkeyPatch,
    chunk_ramp_frames: list[int],
    max_graph_frames: int | None,
    expected_graph_frames: int,
) -> None:
    config = PersonaPlexPipelineConfig(model_path="dummy")
    overrides: dict[str, int | list[int]] = {"chunk_ramp_frames": chunk_ramp_frames}
    if max_graph_frames is not None:
        overrides["max_graph_frames"] = max_graph_frames
    else:
        pass
    patches = patches_from_stages_mapping(
        {"code2wav": {"factory": overrides}}, type(config), ["code2wav"]
    )
    resolved = ConfigResolver(config).resolve(patches).config
    codec = Mock(device=torch.device("cpu"))
    decode_slots = Mock(
        return_value=SimpleNamespace(
            codec=codec,
            max_batch_size=8,
            max_chunk_frames=max(max(chunk_ramp_frames), expected_graph_frames),
        )
    )
    monkeypatch.setattr(stages, "load_codec", lambda *a, **k: (codec, "cpu"))
    monkeypatch.setattr(stages, "MimiDecodeSlots", decode_slots)
    factory_args = resolve_stage_factory_args(
        resolved.stage_named("code2wav"), resolved
    )
    stages.create_code2wav_executor(**factory_args)
    decode_slots.assert_called_once_with(
        codec,
        num_slots=1,
        max_batch_size=8,
        max_graph_frames=expected_graph_frames,
        max_chunk_frames=max(chunk_ramp_frames),
    )
