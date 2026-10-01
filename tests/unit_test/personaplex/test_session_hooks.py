# SPDX-License-Identifier: Apache-2.0
"""The stages around the LM keep per-call codec state and hand the prompt on once."""

import threading

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.session_hooks import (
    Code2WavSessionHooks,
    MimiEncodeSessionHooks,
    PromptSessionHooks,
    pcm16_to_waveform,
    waveform_to_pcm16,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext

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


def test_only_the_first_unit_carries_the_prompt():
    hooks = open_hooks(
        PromptSessionHooks(lambda request: PersonaPlexState(text_prompt_ids=PROMPT_IDS))
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

    assert states[0].carries_prompt and states[0].text_prompt_ids == PROMPT_IDS
    assert not states[1].carries_prompt and states[1].text_prompt_ids == []
    torch.testing.assert_close(states[1].waveform, waveform, atol=1e-4, rtol=0)


def test_units_must_be_whole_frames_of_pcm16():
    with pytest.raises(ValueError, match="whole 80 ms frames"):
        pcm16_to_waveform(unit(0, b"\0\0" * 10))
    with pytest.raises(ValueError, match="pcm16"):
        pcm16_to_waveform(TimedChunk("audio", 0.0, 0.0, 0, b"", format="wav"))
    assert pcm16_to_waveform(unit(0)).numel() == 0


def test_streaming_encode_matches_the_whole_recording(random_codec):
    frames = 3
    waveform = pcm16_to_waveform(
        unit(0, waveform_to_pcm16(torch.randn(frames * SAMPLES_PER_FRAME) * 0.3))
    )
    whole = random_codec.encode(waveform.view(1, 1, -1))[0].T
    voice = torch.randn(2 * SAMPLES_PER_FRAME) * 0.3

    hooks = open_hooks(MimiEncodeSessionHooks(random_codec, torch.device("cpu")))
    streamed = []
    for index in range(frames):
        state = PersonaPlexState(
            waveform=waveform[
                index * SAMPLES_PER_FRAME : (index + 1) * SAMPLES_PER_FRAME
            ],
            voice_waveform=voice if index == 0 else None,
        )
        unit_payload = payload(state)
        hooks.append(unit(index), unit_payload, context([]))
        encoded = PersonaPlexState.from_dict(unit_payload.data)
        assert encoded.waveform is None and encoded.voice_waveform is None
        assert (encoded.voice_codes is not None) == (index == 0)
        streamed.append(encoded.user_codes)
    hooks.close(CALL)

    assert torch.equal(torch.cat(streamed), whole)


class PieceTokenizer:
    def id_to_piece(self, token: int) -> str:
        return {100: "▁hello", 101: "▁there", 102: "!"}[token]


def test_code2wav_streams_audio_and_the_words_spoken(random_codec):
    frames = 3
    codes = torch.randint(0, 2048, (frames, 8))
    whole = random_codec.decode(codes.T[None])[0, 0]
    hooks = open_hooks(
        Code2WavSessionHooks(random_codec, torch.device("cpu"), PieceTokenizer())
    )
    emitted: list[TimedChunk] = []
    # Note (wilsonzheng0327): 3 is PAD and 0 is EPAD, frame markers never spoken.
    text_ids = [[3, 100], [0, 101], [102]]
    for index in range(frames):
        state = PersonaPlexState(
            codes=codes[index : index + 1], text_ids=text_ids[index]
        )
        hooks.append(
            unit(index, eos=index == frames - 1), payload(state), context(emitted)
        )
    hooks.close(CALL)

    audio = [chunk for chunk in emitted if chunk.modality == "audio" and not chunk.eos]
    streamed = np.frombuffer(b"".join(chunk.payload for chunk in audio), dtype="<i2")
    expected = np.frombuffer(waveform_to_pcm16(whole), dtype="<i2")
    assert np.abs(streamed.astype(np.int32) - expected).max() <= 1
    assert [chunk.t_start_ms for chunk in audio] == [0.0, 80.0, 160.0]
    words = [chunk.payload["text"] for chunk in emitted if chunk.modality == "text"]
    assert "".join(words) == "hello there!"
    assert emitted[-1].eos and emitted[-1].t_start_ms == 240.0


def test_an_empty_end_of_input_emits_only_the_end(random_codec):
    hooks = open_hooks(
        Code2WavSessionHooks(random_codec, torch.device("cpu"), PieceTokenizer())
    )
    emitted: list[TimedChunk] = []
    state = PersonaPlexState(codes=torch.zeros(0, 8, dtype=torch.long))
    hooks.append(unit(0, eos=True), payload(state), context(emitted))
    assert [(chunk.modality, chunk.eos, chunk.payload) for chunk in emitted] == [
        ("audio", True, None)
    ]
