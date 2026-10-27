# SPDX-License-Identifier: Apache-2.0
"""The stages around the LM keep per-call codec state and hand the prompt on once."""

import threading
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.mimi_decode_graph import MimiDecodeSlots
from sglang_omni.models.personaplex.mimi_encode_graph import MimiEncodeSlots
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
from sglang_omni.scheduling.session import SessionAppend, SessionContext

CALL = SessionIdentity("call")
PROMPT_IDS = [11, 12, 13]


def context(
    emitted: list[TimedChunk], session_identity: SessionIdentity = CALL
) -> SessionContext:
    return SessionContext(
        session_identity=session_identity,
        cancelled=threading.Event(),
        emit=emitted.append,
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


def test_only_the_first_unit_carries_the_prompt():
    hooks = open_hooks(
        PreprocessingHooks(lambda request: PersonaPlexState(text_prompt_ids=PROMPT_IDS))
    )
    waveform = torch.linspace(-0.5, 0.5, SAMPLES_PER_FRAME)
    states = []
    for index in range(2):
        unit_payload = payload()
        hooks.append(
            unit(index, waveform_to_pcm16(waveform)), unit_payload, context([])
        )
        states.append(PersonaPlexState.from_dict(unit_payload.data))
    hooks.close(CALL)

    assert states[0].text_prompt_ids == PROMPT_IDS
    assert states[1].text_prompt_ids == []
    torch.testing.assert_close(states[1].waveform, waveform, atol=1e-4, rtol=0)


def test_units_must_be_whole_frames_of_pcm16():
    with pytest.raises(ValueError, match="whole 80 ms frames"):
        pcm16_to_waveform(unit(0, b"\0\0" * 10))
    with pytest.raises(ValueError, match="pcm16"):
        pcm16_to_waveform(TimedChunk("audio", 0.0, 0.0, 0, b"", format="wav"))
    with pytest.raises(ValueError, match="whole 80 ms frames"):
        pcm16_to_waveform(unit(0))
    end = TimedChunk("audio", 0.0, 0.0, 0, b"", "pcm16", eos=True)
    assert pcm16_to_waveform(end).numel() == 0


def test_streaming_encode_batches_calls_without_advancing_empty_end(
    random_codec: MimiCodec, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = [SessionIdentity("a"), SessionIdentity("b")]
    waveforms = [
        pcm16_to_waveform(
            unit(0, waveform_to_pcm16(torch.randn(frames * SAMPLES_PER_FRAME) * 0.3))
        )
        for frames in (4, 3)
    ]
    voice = torch.randn(2 * SAMPLES_PER_FRAME) * 0.3
    expected_voice = random_codec.encode(voice.view(1, 1, -1))[0].T
    expected_codes = [
        random_codec.encode(waveform.view(1, 1, -1))[0].T for waveform in waveforms
    ]
    encode_slots = MimiEncodeSlots(
        random_codec, num_slots=2, max_batch_size=2, enable_graphs=False
    )
    encode_step = Mock(wraps=encode_slots.encode_step)
    monkeypatch.setattr(encode_slots, "encode_step", encode_step)
    hooks = MimiEncodeHooks(encode_slots)
    for session in sessions:
        hooks.open(session, OmniRequest(inputs=None))
    starts = [0, 0]
    streamed: list[list[torch.Tensor]] = [[], []]
    for index, entries in enumerate(
        ([(0, 1)], [(1, 2), (0, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 0)])
    ):
        appends = []
        for session_index, frames in entries:
            start = starts[session_index]
            starts[session_index] += frames * SAMPLES_PER_FRAME
            is_empty_end = frames == 0
            chunk = (
                TimedChunk("audio", 240.0, 0.0, index, b"", "pcm16", eos=True)
                if is_empty_end
                else unit(index)
            )
            appends.append(
                SessionAppend(
                    chunk=chunk,
                    payload=payload(
                        PersonaPlexState(
                            waveform=waveforms[session_index][
                                start : starts[session_index]
                            ],
                            voice_waveform=voice if index == 0 else None,
                        )
                    ),
                    context=context([], sessions[session_index]),
                )
            )
        results = hooks.append_batch(appends)
        assert all(
            result is append.payload
            for result, append in zip(results, appends, strict=True)
        )
        for (session_index, frames), result in zip(entries, results, strict=True):
            encoded = PersonaPlexState.from_dict(result.data)
            assert encoded.waveform is None and encoded.voice_waveform is None
            if index == 0:
                torch.testing.assert_close(
                    encoded.voice_codes, expected_voice, atol=0, rtol=0
                )
            else:
                assert encoded.voice_codes is None
            assert encoded.user_codes.shape[0] == frames
            streamed[session_index].append(encoded.user_codes)
    assert [call.args[0].shape[0] for call in encode_step.call_args_list] == [
        1,
        1,
        1,
        2,
        1,
    ]
    for session_index, session in enumerate(sessions):
        torch.testing.assert_close(
            torch.cat(streamed[session_index]),
            expected_codes[session_index],
            atol=0,
            rtol=0,
        )
        hooks.close(session)
    hooks.open(sessions[0], OmniRequest(inputs=None))
    reopened = payload(PersonaPlexState(waveform=waveforms[0]))
    hooks.append_batch(
        [
            SessionAppend(
                chunk=unit(0), payload=reopened, context=context([], sessions[0])
            )
        ]
    )
    torch.testing.assert_close(
        PersonaPlexState.from_dict(reopened.data).user_codes,
        expected_codes[0],
        atol=0,
        rtol=0,
    )
    hooks.close(sessions[0])


def decode_slots(codec, num_slots: int) -> MimiDecodeSlots:
    return MimiDecodeSlots(
        codec,
        num_slots=num_slots,
        max_batch_size=num_slots,
        max_graph_frames=0,
        max_chunk_frames=2,
    )


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
        MimiEncodeSlots(
            random_codec, num_slots=1, max_batch_size=1, enable_graphs=False
        )
    )
    decode = Code2WavHooks(decode_slots(random_codec, 1))
    encode.warm_up_serving_thread()
    decode.warm_up_serving_thread()
    assert encode.states == {} and decode.states == {}
    assert decode.decode_slots.active_slot_count == 0
