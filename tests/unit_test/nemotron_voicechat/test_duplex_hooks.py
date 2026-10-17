# SPDX-License-Identifier: Apache-2.0
"""Model hooks preserve audio content and session lifetime across units."""

import threading
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.code2wav_stream import StreamingCodec
from sglang_omni.models.nemotron_voicechat.conformer import StreamingPerception
from sglang_omni.models.nemotron_voicechat.duplex_hooks import (
    CodecHooks,
    PerceptionHooks,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext


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
        "sglang_omni.models.nemotron_voicechat.duplex_hooks.GraphPerception",
        Mock(return_value=stream),
    )
    hooks = PerceptionHooks(Mock(), max_open_sessions=1)
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


def test_perception_sessions_own_independent_stream_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_stream = Mock(spec=StreamingPerception)
    first_stream.push.return_value = torch.full((1, 4), 1.0)
    second_stream = Mock(spec=StreamingPerception)
    second_stream.push.return_value = torch.full((1, 4), 2.0)
    monkeypatch.setattr(
        "sglang_omni.models.nemotron_voicechat.duplex_hooks.GraphPerception",
        Mock(side_effect=[first_stream, second_stream]),
    )
    hooks = PerceptionHooks(Mock(), max_open_sessions=2)
    first_session = SessionIdentity("first")
    second_session = SessionIdentity("second")
    hooks.open(first_session, OmniRequest(None))
    hooks.open(second_session, OmniRequest(None))

    def session_context(session_identity: SessionIdentity) -> SessionContext:
        return SessionContext(
            session_identity=session_identity,
            cancelled=threading.Event(),
            emit=Mock(),
        )

    first_output = hooks.append(
        audio_chunk(b"\x01\x00" * 1280),
        stage_payload(),
        session_context(first_session),
    )
    second_output = hooks.append(
        audio_chunk(b"\x02\x00" * 1280),
        stage_payload(),
        session_context(second_session),
    )
    torch.testing.assert_close(first_output.data["acoustic"], torch.full((1, 4), 2.0))
    torch.testing.assert_close(second_output.data["acoustic"], torch.full((1, 4), 1.0))
    first_stream.push.assert_called_once()
    second_stream.push.assert_called_once()

    hooks.close(first_session)
    hooks.close(second_session)
    assert len(hooks.available_streams) == 2


@pytest.mark.parametrize(
    "chunk",
    [
        audio_chunk(b""),
        audio_chunk(b"1"),
        audio_chunk(b"12"),
        audio_chunk(bytes(2562)),
        replace(audio_chunk(), modality="text"),
        replace(audio_chunk(), format="float32"),
    ],
)
def test_perception_rejects_invalid_audio_before_model_execution(
    monkeypatch: pytest.MonkeyPatch,
    chunk: TimedChunk,
) -> None:
    stream = Mock(spec=StreamingPerception)
    monkeypatch.setattr(
        "sglang_omni.models.nemotron_voicechat.duplex_hooks.GraphPerception",
        Mock(return_value=stream),
    )
    hooks = PerceptionHooks(Mock(), max_open_sessions=1)
    session_identity = SessionIdentity("invalid-input")
    hooks.open(session_identity, OmniRequest(None))
    context = SessionContext(
        session_identity=session_identity, cancelled=threading.Event(), emit=Mock()
    )
    with pytest.raises(ValueError):
        hooks.append(chunk, stage_payload(), context)
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


def test_codec_memory_stays_bounded_and_reopen_clears_audio_history() -> None:
    hooks = CodecHooks(ConstantFrameDecoder(), "cpu")
    session_identity = SessionIdentity("bounded-codec")
    hooks.open(session_identity, OmniRequest(None))
    context = SessionContext(
        session_identity=session_identity, cancelled=threading.Event(), emit=Mock()
    )
    for frame_index in range(100):
        hooks.append(
            audio_chunk(),
            stage_payload({"codes": torch.tensor([[frame_index]])}),
            context,
        )
        assert (
            hooks.usage(session_identity).bytes <= 16 * torch.tensor(0).element_size()
        )
    hooks.append(audio_chunk(b"", eos=True), stage_payload({"eos": True}), context)
    with pytest.raises(ValueError, match="already ended"):
        hooks.append(
            audio_chunk(),
            stage_payload({"codes": torch.zeros(1, 1, dtype=torch.long)}),
            context,
        )
    hooks.close(session_identity)
    hooks.open(session_identity, OmniRequest(None))
    output = hooks.append(
        audio_chunk(),
        stage_payload({"codes": torch.zeros(1, 1, dtype=torch.long)}),
        context,
    )
    assert output.data["pcm"] == bytes((1764 - 256) * 2)


def test_codec_empty_session_drains_without_audio() -> None:
    hooks = CodecHooks(ConstantFrameDecoder(), "cpu")
    session_identity = SessionIdentity("empty-codec")
    hooks.open(session_identity, OmniRequest(None))
    emitted_chunks: list[TimedChunk] = []
    context = SessionContext(
        session_identity=session_identity,
        cancelled=threading.Event(),
        emit=emitted_chunks.append,
    )
    output = hooks.append(
        audio_chunk(b"", eos=True), stage_payload({"eos": True}), context
    )
    assert output.data["pcm"] == b""
    assert emitted_chunks[-1].eos
    assert emitted_chunks[-1].duration_ms == 0


def test_codec_sessions_keep_history_and_completion_independent() -> None:
    hooks = CodecHooks(ConstantFrameDecoder(), "cpu")
    first_session = SessionIdentity("first-codec")
    second_session = SessionIdentity("second-codec")
    hooks.open(first_session, OmniRequest(None))
    hooks.open(second_session, OmniRequest(None))

    def session_context(session_identity: SessionIdentity) -> SessionContext:
        return SessionContext(
            session_identity=session_identity,
            cancelled=threading.Event(),
            emit=Mock(),
        )

    first_output = hooks.append(
        audio_chunk(),
        stage_payload({"codes": torch.tensor([[1]])}),
        session_context(first_session),
    )
    second_output = hooks.append(
        audio_chunk(),
        stage_payload({"codes": torch.tensor([[2]])}),
        session_context(second_session),
    )
    assert first_output.data["pcm"] != second_output.data["pcm"]
    hooks.append(
        audio_chunk(b"", eos=True),
        stage_payload({"eos": True}),
        session_context(first_session),
    )
    continued = hooks.append(
        audio_chunk(),
        stage_payload({"codes": torch.tensor([[3]])}),
        session_context(second_session),
    )
    assert continued.data["pcm"]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires GPU")
@torch.inference_mode()
def test_codec_replay_matches_eager_for_changing_codes() -> None:
    def decode_codes(codes: torch.Tensor) -> torch.Tensor:
        return codes.float().sin().sum(-1).repeat_interleave(4)

    hooks = CodecHooks(decode_codes, "cuda")
    for frame_count in [1, 15, 16, 16, 16]:
        codes = torch.randint(0, 100, (frame_count, 8), device="cuda")
        torch.testing.assert_close(
            hooks.decode(codes), decode_codes(codes), rtol=0, atol=0
        )
