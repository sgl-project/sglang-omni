# SPDX-License-Identifier: Apache-2.0
"""Model hooks preserve audio content and session lifetime across units."""

import threading
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.code2wav_stream import StreamingCodec
from sglang_omni.models.nemotron_voicechat.conformer import StreamingPerception
from sglang_omni.models.nemotron_voicechat.duplex import CodecHooks, PerceptionHooks
from sglang_omni.models.nemotron_voicechat.realtime import VoiceChatOutput
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext
from sglang_omni.serve.realtime.output import ResponseFinished, ResponseStarted


def stage_payload(
    fields: dict[str, torch.Tensor | bytes | str | int | bool | None] | None = None,
) -> StagePayload:
    return StagePayload("unit", OmniRequest(None), fields or {})


def audio_chunk(pcm: bytes = bytes(2560), *, eos: bool = False) -> TimedChunk:
    return TimedChunk("audio", 0, len(pcm) / 32, 0, pcm, "pcm16", eos)


def test_perception_preserves_pcm_and_drains_without_an_extra_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_waveforms: list[torch.Tensor] = []

    def encode_frame(waveform: torch.Tensor) -> torch.Tensor:
        input_waveforms.append(waveform.clone())
        return waveform.mean().repeat(1, 4)

    stream = Mock(spec=StreamingPerception, push=encode_frame)
    monkeypatch.setattr(
        "sglang_omni.models.nemotron_voicechat.duplex.GraphPerception",
        Mock(return_value=stream),
    )
    hooks = PerceptionHooks(Mock())
    session_identity = SessionIdentity("perception")
    hooks.open(session_identity, OmniRequest(None))
    context = SessionContext(
        session_identity=session_identity,
        cancelled=threading.Event(),
        emit=Mock(),
    )
    first_output = hooks.append(
        audio_chunk(b"\x00\x80" * 1280), stage_payload(), context
    )
    second_output = hooks.append(audio_chunk(), stage_payload(), context)
    final_output = hooks.append(audio_chunk(b"", eos=True), stage_payload(), context)
    torch.testing.assert_close(first_output.data["acoustic"], -torch.ones(1, 4))
    torch.testing.assert_close(second_output.data["acoustic"], torch.zeros(1, 4))
    assert len(input_waveforms) == 2
    assert final_output.data == {"acoustic": None, "eos": True}
    with pytest.raises(ValueError, match="already ended"):
        hooks.append(audio_chunk(), stage_payload(), context)
    hooks.close(session_identity)
    assert hooks.usage(session_identity).bytes == 0


@pytest.mark.parametrize("pcm", [b"", b"1", b"12", bytes(2562)])
def test_perception_rejects_incomplete_units_before_model_execution(
    monkeypatch: pytest.MonkeyPatch,
    pcm: bytes,
) -> None:
    stream = Mock(spec=StreamingPerception)
    monkeypatch.setattr(
        "sglang_omni.models.nemotron_voicechat.duplex.GraphPerception",
        Mock(return_value=stream),
    )
    hooks = PerceptionHooks(Mock())
    session_identity = SessionIdentity("invalid-input")
    hooks.open(session_identity, OmniRequest(None))
    context = SessionContext(
        session_identity=session_identity, cancelled=threading.Event(), emit=Mock()
    )
    with pytest.raises(ValueError):
        hooks.append(audio_chunk(pcm), stage_payload(), context)
    stream.push.assert_not_called()


class ConstantFrameDecoder:
    samples_per_frame = 1764

    def __call__(self, code_frames: torch.Tensor) -> torch.Tensor:
        return code_frames[:, 0].float().repeat_interleave(self.samples_per_frame) / 100


def test_codec_matches_offline_samples_including_final_tail() -> None:
    decoder = ConstantFrameDecoder()
    hooks = CodecHooks(decoder, "cpu")
    session_identity = SessionIdentity("codec")
    hooks.open(session_identity, OmniRequest(None))
    reference = StreamingCodec(decoder, "cpu")
    output_chunks: list[TimedChunk] = []
    context = SessionContext(
        session_identity=session_identity,
        cancelled=threading.Event(),
        emit=output_chunks.append,
    )
    expected_packets: list[bytes] = []
    actual_packets: list[bytes] = []
    for frame_index in range(50):
        codes = torch.tensor([[frame_index]])
        output = hooks.append(audio_chunk(), stage_payload({"codes": codes}), context)
        actual_packets.append(output.data["pcm"])
        reference_audio = reference.push(codes)
        expected_packets.append(
            (reference_audio.numpy() * 32767).astype("<i2").tobytes()
        )
    output = hooks.append(
        audio_chunk(b"", eos=True), stage_payload({"eos": True}), context
    )
    actual_packets.append(output.data["pcm"])
    expected_packets.append((reference.flush().numpy() * 32767).astype("<i2").tobytes())
    assert b"".join(actual_packets) == b"".join(expected_packets)
    assert sum(map(len, actual_packets)) == 50 * 1764 * 2
    assert len(output_chunks) == 51
    assert output_chunks[-1].eos
    hooks.close(session_identity)
    assert hooks.usage(session_identity).bytes == 0


def test_output_response_spans_units_and_restarts_after_reopen() -> None:
    converter = VoiceChatOutput()
    output = OutputChunk(
        SessionIdentity("conversation"),
        0,
        0,
        "audio",
        0,
        80,
        {"pcm": bytes(2), "text": "hi", "eos": False},
    )
    first_events = list(converter(output))
    continuation_events = list(converter(replace(output, input_seq=1)))
    assert isinstance(first_events[0], ResponseStarted)
    assert not any(isinstance(event, ResponseStarted) for event in continuation_events)
    reopened_output = replace(
        output, session_identity=SessionIdentity("conversation", open_index=2)
    )
    reopened_events = list(converter(reopened_output))
    assert isinstance(reopened_events[0], ResponseStarted)
    assert reopened_events[0].response_id != first_events[0].response_id
    final_events = list(
        converter(
            replace(reopened_output, payload={"pcm": b"", "text": "", "eos": True})
        )
    )
    assert isinstance(final_events[-1], ResponseFinished)
    assert final_events[-1].text == "hi"
