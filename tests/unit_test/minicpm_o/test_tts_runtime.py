# SPDX-License-Identifier: Apache-2.0
"""Vocoder sessions: shared per-voice stream caches, turn resets and failed opens."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.minicpm_o.components import tts_runtime as tts_runtime_module
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import StreamChunk
from sglang_omni.models.minicpm_o.components.tts_runtime import (
    CODEC_CHUNK_SIZE,
    MiniCPMOVocoderRuntime,
    SynthesisRequest,
)
from sglang_omni.models.minicpm_o.native_stages import SpeechHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionAppend, SessionContext
from sglang_omni.scheduling.speaker_cache import estimate_cache_bytes

PRE_LOOKAHEAD = 3


class FakeToken2Wav:
    """Counts stream prefills; streaming writes into the caches it is given."""

    def __init__(self) -> None:
        self.flow = SimpleNamespace(pre_lookahead_len=PRE_LOOKAHEAD)
        self.device = torch.device("cpu")
        self.opened = 0

    def open_stream(self, prompt: tuple[torch.Tensor, ...]) -> tuple[dict, dict]:
        self.opened += 1
        return {"estimator_attention_cache": torch.zeros(64)}, {
            "speech": torch.zeros(16)
        }

    def stream_batch(
        self, chunks: list[StreamChunk]
    ) -> list[tuple[bytes, tuple[dict, dict]]]:
        for chunk in chunks:
            chunk.caches[0]["estimator_attention_cache"].add_(1)
        return [(b"\0\0" * 100, chunk.caches) for chunk in chunks]


class FakeCode2Wav:
    """Stands in for the reference cache of MiniCPMOCode2Wav."""

    def __init__(self) -> None:
        self.token2wav = FakeToken2Wav()
        self.decode_stream = torch.cpu.current_stream()
        self.prepared: list[bytes] = []

    def resolve_reference_key(self, reference: bytes) -> tuple[str, bytes]:
        return f"bytes:{reference.hex()}", reference

    def prepare_references(self, references: list[bytes]) -> list[tuple]:
        if b"invalid" in references:
            raise ValueError("invalid audio")
        else:
            pass
        self.prepared.extend(references)
        return [(torch.zeros(4), torch.zeros(4), torch.ones(8), torch.zeros(1, 6, 80))]


def test_voice_cache_lifecycle() -> None:
    code2wav = FakeCode2Wav()
    runtime = MiniCPMOVocoderRuntime(
        code2wav, max_state_bytes_per_session=1 << 30, max_open_sessions=2
    )

    first = runtime.open_session("a", reference_audio=b"voice-1")
    second = runtime.open_session("b", reference_audio=b"voice-1")
    other = runtime.open_session("c", reference_audio=b"voice-2")

    assert code2wav.prepared == [b"voice-1", b"voice-2"]
    assert code2wav.token2wav.opened == 2
    assert first.speaker is second.speaker and other.speaker is not first.speaker
    assert (
        first.caches[0]["estimator_attention_cache"].data_ptr()
        != second.caches[0]["estimator_attention_cache"].data_ptr()
    )
    shared = estimate_cache_bytes(first.speaker.base_caches)
    own = estimate_cache_bytes((first.caches, first.pending_codec_token_ids))
    assert runtime.held("a").bytes == own + shared
    assert runtime.held("b").bytes == own + shared

    runtime.close_session("a")
    assert runtime.held("b").bytes == own + shared
    runtime.close_session("b")
    runtime.close_session("c")
    with pytest.raises(ValueError, match="invalid audio"):
        runtime.open_session("d", reference_audio=b"invalid")
    assert not runtime.speakers and not runtime.sessions


@pytest.mark.parametrize(("has_existing_session",), [(False,), (True,)])
def test_failed_cache_clone_leaves_no_speaker_owner(
    monkeypatch: pytest.MonkeyPatch, has_existing_session: bool
) -> None:
    runtime = MiniCPMOVocoderRuntime(
        FakeCode2Wav(), max_state_bytes_per_session=1 << 30, max_open_sessions=2
    )
    if has_existing_session:
        runtime.open_session("existing", reference_audio=b"voice")
    else:
        pass

    with monkeypatch.context() as patch:
        patch.setattr(
            tts_runtime_module,
            "clone_caches",
            Mock(
                spec=tts_runtime_module.clone_caches,
                side_effect=torch.OutOfMemoryError("cache clone failed"),
            ),
        )
        with pytest.raises(torch.OutOfMemoryError, match="cache clone failed"):
            runtime.open_session("failed", reference_audio=b"voice")
    if has_existing_session:
        runtime.close_session("existing")
    else:
        pass

    assert not runtime.speakers and not runtime.sessions
    runtime.open_session("recovered", reference_audio=b"voice")
    runtime.close_session("recovered")
    assert not runtime.speakers and not runtime.sessions


def test_turn_reset_restores_untouched_prompt_caches() -> None:
    runtime = MiniCPMOVocoderRuntime(
        FakeCode2Wav(), max_state_bytes_per_session=1 << 30, max_open_sessions=2
    )
    state = runtime.open_session("a", reference_audio=b"voice")
    runtime.open_session("b", reference_audio=b"voice")

    list(
        runtime.synthesize_batch(
            [
                SynthesisRequest(
                    session_id="a",
                    codec_token_ids=[1] * (CODEC_CHUNK_SIZE + PRE_LOOKAHEAD),
                    is_turn_start=True,
                    end_of_turn=False,
                )
            ]
        )
    )
    assert state.caches[0]["estimator_attention_cache"].sum() > 0
    assert state.speaker.base_caches[0]["estimator_attention_cache"].sum() == 0

    list(
        runtime.synthesize_batch(
            [
                SynthesisRequest(
                    session_id="a",
                    codec_token_ids=[],
                    is_turn_start=False,
                    end_of_turn=True,
                )
            ]
        )
    )
    assert state.caches[0]["estimator_attention_cache"].sum() == 0


@pytest.mark.parametrize("stream_decode_chunks", [False, True])
def test_speech_unit_pcm_and_final_markers(
    stream_decode_chunks: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MiniCPMOVocoderRuntime(
        FakeCode2Wav(), max_state_bytes_per_session=1 << 30, max_open_sessions=2
    )
    hooks = SpeechHooks(
        runtime,
        b"voice",
        gather_window_ms=0,
        stream_decode_chunks=stream_decode_chunks,
    )
    session_identity = SessionIdentity("session")
    request = OmniRequest(inputs=None)
    hooks.open(session_identity, request)

    stream_batch = Mock(wraps=runtime.token2wav.stream_batch)
    monkeypatch.setattr(runtime.token2wav, "stream_batch", stream_batch)
    outputs: list[tuple[TimedChunk, int]] = []

    def emit(chunk: TimedChunk) -> None:
        outputs.append((chunk, stream_batch.call_count))

    payload = StagePayload(
        request_id="unit",
        request=request,
        data={
            "ends_previous_turn": False,
            "end_of_turn": True,
            "codec_tokens": [1] * (2 * CODEC_CHUNK_SIZE),
            "speech_turn_start": False,
            "text": "hello",
            "is_speech_cut": True,
            "is_listen": False,
        },
    )
    append = SessionAppend(
        chunk=TimedChunk("audio", 0, 1000, 1, None, eos=True),
        payload=payload,
        context=SessionContext(
            session_identity=session_identity, cancelled=Event(), emit=emit
        ),
    )

    assert hooks.append_batch([append]) == [payload]
    assert payload.data is None
    assert [round_count for _, round_count in outputs] == (
        [1, 2, 3] if stream_decode_chunks else [3]
    )
    chunks = [chunk for chunk, _ in outputs]
    assert b"".join(chunk.payload["pcm"] for chunk in chunks) == bytes(600)
    assert "".join(chunk.payload["text"] for chunk in chunks) == "hello"
    assert [chunk.seq for chunk in chunks] == [1] * len(chunks)
    assert all(chunk.turn == 0 for chunk in chunks)
    assert sum(chunk.duration_ms for chunk in chunks) == pytest.approx(12.5)
    assert chunks[0].t_start_ms == 0
    for previous, current in zip(chunks, chunks[1:]):
        assert current.t_start_ms == pytest.approx(
            previous.t_start_ms + previous.duration_ms
        )
        assert not previous.eos
        assert not previous.payload["end_of_turn"]
        assert not previous.payload["is_speech_cut"]
    assert chunks[-1].eos
    assert chunks[-1].payload["end_of_turn"]
    assert chunks[-1].payload["is_speech_cut"]
    assert not runtime.sessions["session"].has_pending_turn
