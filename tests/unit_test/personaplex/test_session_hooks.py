# SPDX-License-Identifier: Apache-2.0
"""The stages around the LM keep per-call codec state and hand the prompt on once."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.session_hooks import (
    Code2WavHooks,
    MimiEncodeHooks,
    PreprocessingHooks,
    pcm16_to_waveform,
    waveform_to_pcm16,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import (
    SessionAppend,
    SessionContext,
    SessionScheduler,
)

CALL = SessionIdentity("call")
PROMPT_IDS = [11, 12, 13]


def context(emitted: list[TimedChunk]) -> SessionContext:
    return SessionContext(
        session_identity=CALL, cancelled=threading.Event(), emit=emitted.append
    )


def unit(index: int, pcm: bytes = b"", *, eos: bool = False) -> TimedChunk:
    return TimedChunk("audio", 80.0 * index, 80.0, index, pcm, "pcm16", eos)


def payload(state: PersonaPlexState | None = None) -> StagePayload:
    return StagePayload(
        "unit",
        request=OmniRequest(inputs=None),
        data={} if state is None else state.to_dict(),
    )


def open_hooks(hooks):
    hooks.open(CALL, OmniRequest(inputs=None))
    return hooks


def test_only_the_first_unit_carries_the_prompt(random_codec):
    preprocessing = open_hooks(PreprocessingHooks())
    hooks = open_hooks(
        MimiEncodeHooks(
            random_codec,
            torch.device("cpu"),
            prepare_prompt=lambda request: PersonaPlexState(text_prompt_ids=PROMPT_IDS),
            voice_code_cache_bytes=0,
        )
    )
    waveform = torch.linspace(-0.5, 0.5, SAMPLES_PER_FRAME)
    states = []
    for index in range(2):
        unit_payload = payload()
        preprocessing.append(
            unit(index, waveform_to_pcm16(waveform)), unit_payload, context([])
        )
        torch.testing.assert_close(
            PersonaPlexState.from_dict(unit_payload.data).waveform,
            waveform,
            atol=1e-4,
            rtol=0,
        )
        hooks.append(unit(index), unit_payload, context([]))
        states.append(PersonaPlexState.from_dict(unit_payload.data))
    hooks.close(CALL)

    assert states[0].text_prompt_ids == PROMPT_IDS
    assert states[1].text_prompt_ids == []


def test_units_must_be_whole_frames_of_pcm16():
    with pytest.raises(ValueError, match="whole 80 ms frames"):
        pcm16_to_waveform(unit(0, b"\0\0" * 10))
    with pytest.raises(ValueError, match="pcm16"):
        pcm16_to_waveform(TimedChunk("audio", 0.0, 0.0, 0, b"", format="wav"))
    with pytest.raises(ValueError, match="whole 80 ms frames"):
        pcm16_to_waveform(unit(0))
    end = TimedChunk("audio", 0.0, 0.0, 0, b"", "pcm16", eos=True)
    assert pcm16_to_waveform(end).numel() == 0


def test_streaming_encode_matches_the_whole_recording(random_codec):
    frames = 3
    waveform = pcm16_to_waveform(
        unit(0, waveform_to_pcm16(torch.randn(frames * SAMPLES_PER_FRAME) * 0.3))
    )
    whole = random_codec.encode(waveform.view(1, 1, -1))[0].T
    voice = torch.randn(2 * SAMPLES_PER_FRAME) * 0.3

    hooks = open_hooks(
        MimiEncodeHooks(
            random_codec,
            torch.device("cpu"),
            prepare_prompt=lambda request: PersonaPlexState(voice_waveform=voice),
            voice_code_cache_bytes=0,
        )
    )
    streamed = []
    for index in range(frames):
        state = PersonaPlexState(
            waveform=waveform[
                index * SAMPLES_PER_FRAME : (index + 1) * SAMPLES_PER_FRAME
            ],
        )
        unit_payload = payload(state)
        hooks.append(unit(index), unit_payload, context([]))
        encoded = PersonaPlexState.from_dict(unit_payload.data)
        assert encoded.waveform is None and encoded.voice_waveform is None
        assert (encoded.voice_codes is not None) == (index == 0)
        streamed.append(encoded.user_codes)
    hooks.close(CALL)

    assert torch.equal(torch.cat(streamed), whole)


@pytest.mark.parametrize("cached_voices,expected_encodes", [(0, 4), (1, 3), (2, 2)])
def test_recorded_voice_cache_preserves_codes_and_session_isolation(
    random_codec, cached_voices: int, expected_encodes: int
) -> None:
    """Reopening a recorded voice used to encode it again on the first unit."""
    voices = [torch.randn(2 * SAMPLES_PER_FRAME) * 0.3 for _ in range(2)]
    voice_codes = [random_codec.encode(voice[None, None])[0].T for voice in voices]
    caller = torch.randn(SAMPLES_PER_FRAME) * 0.3
    caller_codes = random_codec.encode(caller[None, None])[0].T
    hooks = MimiEncodeHooks(
        random_codec,
        torch.device("cpu"),
        prepare_prompt=lambda request: PersonaPlexState(
            voice_waveform=voices[request.params["voice_index"]].clone()
        ),
        voice_code_cache_bytes=cached_voices * voice_codes[0].numel() * 8,
    )
    with patch.object(random_codec, "encode", wraps=random_codec.encode) as encode:
        for open_index, voice_index in enumerate([0, 0, 1, 0]):
            session_identity = SessionIdentity("voice-call", open_index)
            hooks.open(
                session_identity,
                OmniRequest(inputs=None, params={"voice_index": voice_index}),
            )
            encodes_after_open = encode.call_count
            unit_payload = payload(PersonaPlexState(waveform=caller.clone()))
            hooks.append(
                unit(0),
                unit_payload,
                SessionContext(
                    session_identity=session_identity,
                    cancelled=threading.Event(),
                    emit=[].append,
                ),
            )
            encoded = PersonaPlexState.from_dict(unit_payload.data)
            assert torch.equal(encoded.voice_codes, voice_codes[voice_index])
            assert torch.equal(encoded.user_codes, caller_codes)
            assert encoded.voice_waveform is None
            assert encode.call_count == encodes_after_open
            hooks.close(session_identity)
        assert encode.call_count == expected_encodes
    assert hooks.states == {} and hooks.prompts == {}


def decode_slots(codec, num_slots: int) -> MimiDecodeSlots:
    return MimiDecodeSlots(
        codec,
        num_slots=num_slots,
        max_batch_size=num_slots,
        max_graph_frames=0,
        max_chunk_frames=2,
    )


def test_concurrent_opens_share_voice_codes_and_keep_caller_state_separate(
    random_codec,
) -> None:
    """Concurrent opens of a cold voice must not encode it twice or share streaming state."""
    voice = torch.randn(2 * SAMPLES_PER_FRAME) * 0.3
    callers = [torch.randn(2 * SAMPLES_PER_FRAME) * 0.3 for _ in range(2)]
    expected = [random_codec.encode(caller[None, None])[0].T for caller in callers]
    prepared = threading.Barrier(2)

    def prepare_prompt(request: OmniRequest) -> PersonaPlexState:
        prepared.wait(timeout=10)
        return PersonaPlexState(voice_waveform=voice.clone())

    hooks = MimiEncodeHooks(
        random_codec,
        torch.device("cpu"),
        prepare_prompt=prepare_prompt,
        voice_code_cache_bytes=4096,
    )
    identities = [SessionIdentity(f"concurrent-{index}") for index in range(2)]
    with patch.object(random_codec, "encode", wraps=random_codec.encode) as encode:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(hooks.open, identity, OmniRequest(inputs=None))
                for identity in identities
            ]
            for future in futures:
                future.result(timeout=30)
        assert encode.call_count == 1
        for index, identity in enumerate(identities):
            encoded_frames = []
            for frame_index in range(2):
                start = frame_index * SAMPLES_PER_FRAME
                unit_payload = payload(
                    PersonaPlexState(
                        waveform=callers[index][start : start + SAMPLES_PER_FRAME]
                    )
                )
                hooks.append(
                    unit(frame_index),
                    unit_payload,
                    SessionContext(
                        session_identity=identity,
                        cancelled=threading.Event(),
                        emit=[].append,
                    ),
                )
                encoded_frames.append(
                    PersonaPlexState.from_dict(unit_payload.data).user_codes
                )
            assert torch.equal(torch.cat(encoded_frames), expected[index])
            hooks.close(identity)
    assert hooks.states == {} and hooks.prompts == {}


@pytest.mark.parametrize("failed_open", [False, True])
def test_open_without_audio_releases_prompt_and_encoder_state(
    random_codec, failed_open: bool
) -> None:
    """A failed voice encode and a call closed without audio must both release open state."""
    hooks = MimiEncodeHooks(
        random_codec,
        torch.device("cpu"),
        prepare_prompt=lambda request: PersonaPlexState(
            voice_waveform=torch.zeros(SAMPLES_PER_FRAME)
        ),
        voice_code_cache_bytes=0,
    )
    scheduler = SessionScheduler(hooks, max_open_sessions=1)
    if failed_open:
        with patch.object(
            random_codec, "encode", side_effect=RuntimeError("encode failed")
        ):
            with pytest.raises(RuntimeError, match="encode failed"):
                scheduler.open_session(CALL, OmniRequest(inputs=None))
    else:
        scheduler.open_session(CALL, OmniRequest(inputs=None))
        scheduler.close_session(CALL, scheduler.open_sessions[CALL])
    assert scheduler.open_sessions == {}
    assert hooks.states == {} and hooks.prompts == {}


def test_code2wav_decodes_sessions_together_and_emits_each_unit_once(random_codec):
    sessions = [SessionIdentity("a"), SessionIdentity("b")]
    frames_per_unit = {sessions[0]: [2, 1, 1], sessions[1]: [1, 1, 2]}
    codes = {session: torch.randint(0, 2048, (4, 8)) for session in sessions}
    hooks = Code2WavHooks(decode_slots(random_codec, 2))
    emitted: dict[SessionIdentity, list[TimedChunk]] = {}
    for session in sessions:
        hooks.open(session, OmniRequest(inputs=None))
        emitted[session] = []
    starts = dict.fromkeys(sessions, 0)
    for index in range(3):
        appends = []
        for session in sessions:
            count = frames_per_unit[session][index]
            start = starts[session]
            starts[session] += count
            state = PersonaPlexState(
                codes=codes[session][start : start + count], text=f"w{index} "
            )
            appends.append(
                SessionAppend(
                    chunk=unit(index, eos=index == 2),
                    payload=payload(state),
                    context=SessionContext(
                        session_identity=session,
                        cancelled=threading.Event(),
                        emit=emitted[session].append,
                    ),
                )
            )
        hooks.append_batch(appends)
    for session in sessions:
        hooks.close(session)

    for session in sessions:
        chunks = emitted[session]
        whole = random_codec.decode(codes[session].T[None])[0, 0]
        streamed = np.frombuffer(
            b"".join(chunk.payload["pcm"] for chunk in chunks), dtype="<i2"
        )
        expected = np.frombuffer(waveform_to_pcm16(whole), dtype="<i2")
        assert np.abs(streamed.astype(np.int32) - expected).max() <= 1
        assert [chunk.payload["text"] for chunk in chunks] == ["w0 ", "w1 ", "w2 "]
        assert [chunk.eos for chunk in chunks] == [False, False, True]
        widths = [80.0 * count for count in frames_per_unit[session]]
        assert [chunk.t_start_ms for chunk in chunks] == [
            0.0,
            widths[0],
            sum(widths[:2]),
        ]


def test_an_empty_end_of_input_emits_only_the_end(random_codec):
    hooks = open_hooks(Code2WavHooks(decode_slots(random_codec, 1)))
    emitted: list[TimedChunk] = []
    state = PersonaPlexState(codes=torch.zeros(0, 8, dtype=torch.long))
    hooks.append_batch(
        [
            SessionAppend(
                chunk=unit(0, eos=True),
                payload=payload(state),
                context=context(emitted),
            )
        ]
    )
    assert [(chunk.eos, chunk.payload) for chunk in emitted] == [
        (True, {"text": "", "pcm": b""})
    ]


def test_warm_up_runs_both_codecs_and_opens_no_call(random_codec):
    encode = MimiEncodeHooks(
        random_codec,
        torch.device("cpu"),
        prepare_prompt=lambda request: PersonaPlexState(),
        voice_code_cache_bytes=0,
    )
    decode = Code2WavHooks(decode_slots(random_codec, 1))
    encode.warm_up_serving_thread()
    decode.warm_up_serving_thread()
    assert encode.states == {} and decode.states == {}
    assert decode.decode_slots.active_slot_count == 0
