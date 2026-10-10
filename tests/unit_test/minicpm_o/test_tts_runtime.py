# SPDX-License-Identifier: Apache-2.0
"""Vocoder sessions: shared per-voice stream caches, turn resets and failed opens."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.minicpm_o.components.token2wav.vocoder import StreamChunk
from sglang_omni.models.minicpm_o.components.tts_runtime import (
    CODEC_CHUNK_SIZE,
    MiniCPMOVocoderRuntime,
    SynthesisRequest,
)
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
