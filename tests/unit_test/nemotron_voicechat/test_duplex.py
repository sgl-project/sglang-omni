# SPDX-License-Identifier: Apache-2.0
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.code2wav_stream import StreamingCodec
from sglang_omni.models.nemotron_voicechat.duplex import (
    CodecHooks,
    PerceptionHooks,
    PerceptionState,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext


def payload(data=None):
    return StagePayload("unit", OmniRequest(None), data or {})


def chunk(raw=b"\0\0" * 1280, *, eos=False):
    return TimedChunk("audio", 0, len(raw) / 32, 0, raw, "pcm16", eos)


def test_perception_retains_history_and_eos_does_not_invent_a_frame():
    rows = []
    stream = SimpleNamespace(push=lambda x: rows.append(x.clone()) or torch.zeros(1, 4))
    state = PerceptionState(stream=stream)
    hooks = PerceptionHooks(None)
    identity = SessionIdentity("s")
    hooks.states[identity] = state
    context = SessionContext(
        session_identity=identity, cancelled=threading.Event(), emit=lambda chunk: None
    )
    hooks.append(chunk(b"\x00\x80" * 1280), payload(), context)
    hooks.append(chunk(), payload(), context)
    ended = hooks.append(chunk(b"", eos=True), payload(), context)
    assert len(rows) == 2 and rows[0].eq(-1).all()
    assert ended.data == {"acoustic": None, "eos": True}
    with pytest.raises(ValueError, match="already ended"):
        hooks.append(chunk(), payload(), context)
    hooks.close(SessionIdentity("s"))
    assert hooks.usage(SessionIdentity("s")).bytes == 0


@pytest.mark.parametrize("raw", [b"", b"1", b"12", b"00" * 1281])
def test_perception_rejects_bad_frame_before_mutation(raw):
    stream = SimpleNamespace(push=lambda _: pytest.fail("invalid frame reached model"))
    hooks = PerceptionHooks(None)
    identity = SessionIdentity("s")
    hooks.states[identity] = PerceptionState(stream=stream)
    context = SessionContext(
        session_identity=identity, cancelled=threading.Event(), emit=lambda chunk: None
    )
    with pytest.raises(ValueError):
        hooks.append(chunk(raw), payload(), context)


class Decoder:
    samples_per_frame = 1764

    def __call__(self, codes):
        return codes[:, 0].float().repeat_interleave(self.samples_per_frame) / 100


def test_codec_matches_offline_streaming_with_bounded_history_and_flush():
    decoder = Decoder()
    hooks = CodecHooks(decoder, "cpu")
    hooks.open(SessionIdentity("s"), OmniRequest(None))
    reference = StreamingCodec(decoder, "cpu")
    emitted = []
    context = SessionContext(
        session_identity=SessionIdentity("s"),
        cancelled=threading.Event(),
        emit=emitted.append,
    )
    expected, actual = [], []
    for i in range(50):
        codes = torch.tensor([[i]])
        out = hooks.append(chunk(), payload({"codes": codes}), context)
        actual.append(out.data["pcm"])
        ref = reference.push(codes)
        expected.append((ref.numpy() * 32767).astype("<i2").tobytes())
        assert len(hooks.states[SessionIdentity("s")].code_frames) <= 16
    out = hooks.append(chunk(b"", eos=True), payload({"eos": True}), context)
    actual.append(out.data["pcm"])
    expected.append((reference.flush().numpy() * 32767).astype("<i2").tobytes())
    assert b"".join(actual) == b"".join(expected)
    assert sum(map(len, actual)) == 50 * 1764 * 2
    assert len(emitted) == 51 and emitted[-1].eos
    hooks.close(SessionIdentity("s"))
    assert hooks.usage(SessionIdentity("s")).bytes == 0


def test_output_response_spans_units_and_restarts_after_reopen():
    from sglang_omni.models.nemotron_voicechat.realtime import VoiceChatOutput
    from sglang_omni.serve.realtime.output import ResponseFinished, ResponseStarted

    converter = VoiceChatOutput()
    out = OutputChunk(
        SessionIdentity("s"),
        0,
        0,
        "audio",
        0,
        80,
        {"pcm": b"00", "text": "hi", "eos": False},
    )
    first = list(converter(out))
    second = list(converter(replace(out, input_seq=1)))
    assert isinstance(first[0], ResponseStarted)
    assert not any(isinstance(e, ResponseStarted) for e in second)
    third = list(
        converter(
            replace(
                out, session_identity=SessionIdentity("s", open_index=2), input_seq=2
            )
        )
    )
    assert isinstance(third[0], ResponseStarted)
    assert third[0].response_id != first[0].response_id
    last = list(
        converter(
            replace(
                out,
                session_identity=SessionIdentity("s", open_index=2),
                payload={"pcm": b"", "text": "", "eos": True},
            )
        )
    )
    assert isinstance(last[-1], ResponseFinished) and last[-1].text == "hi"
