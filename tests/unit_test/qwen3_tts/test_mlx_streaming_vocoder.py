# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLX Qwen3-TTS streaming vocoder stage."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sglang_omni.models.qwen3_tts.mlx.config import (  # noqa: E402
    TokenizerConfig,
    TokenizerDecoderConfig,
)
from sglang_omni.models.qwen3_tts.mlx.speech_tokenizer import (  # noqa: E402
    Qwen3TTSSpeechTokenizer,
)
from sglang_omni.models.qwen3_tts.mlx.streaming_vocoder import (  # noqa: E402
    Qwen3TTSMlxStreamingVocoder,
)

GROUPS = 4


def _tokenizer() -> Qwen3TTSSpeechTokenizer:
    mx.random.seed(0)
    config = TokenizerConfig(
        encoder_config=None,
        decoder_config=TokenizerDecoderConfig(
            latent_dim=16,
            codebook_dim=8,
            codebook_size=32,
            decoder_dim=16,
            hidden_size=8,
            intermediate_size=16,
            head_dim=4,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=2,
            num_quantizers=GROUPS,
            num_semantic_quantizers=1,
            sliding_window=6,
            upsample_rates=[2, 2],
            upsampling_ratios=[2],
        ),
        decode_upsample_rate=8,
    )
    tokenizer = Qwen3TTSSpeechTokenizer(config)
    mx.eval(tokenizer.parameters())
    return tokenizer


def _vocoder(**kwargs) -> Qwen3TTSMlxStreamingVocoder:
    return Qwen3TTSMlxStreamingVocoder(_tokenizer(), stream_stride=2, **kwargs)


def _codes(frames: int, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(1, 32, size=(frames, GROUPS)).astype(np.int64))


def _feed(vocoder, request_id, state, codes):
    """Validate + ingest one chunk, mirroring the base class."""
    checked = vocoder.validate_chunk(request_id, state, codes)
    vocoder.ingest(request_id, state, checked)


# --------------------------------------------------------------------------
# Session isolation — the property concurrent requests depend on
# --------------------------------------------------------------------------


def test_interleaved_requests_do_not_corrupt_each_other() -> None:
    """Two streams decoded turn-by-turn must equal each decoded alone."""
    vocoder = _vocoder()
    codes_a = [_codes(3, seed=1), _codes(3, seed=2)]
    codes_b = [_codes(3, seed=3), _codes(3, seed=4)]

    def solo(chunks):
        state = vocoder.create_stream_state("solo")
        out = []
        for chunk in chunks:
            _feed(vocoder, "solo", state, chunk)
            delta = vocoder.decode_delta("solo", state, is_final=False)
            out.append(delta)
        return out

    expected_a = solo(codes_a)
    expected_b = solo(codes_b)

    state_a = vocoder.create_stream_state("a")
    state_b = vocoder.create_stream_state("b")
    got_a, got_b = [], []
    for index in range(2):
        _feed(vocoder, "a", state_a, codes_a[index])
        got_a.append(vocoder.decode_delta("a", state_a, is_final=False))
        _feed(vocoder, "b", state_b, codes_b[index])
        got_b.append(vocoder.decode_delta("b", state_b, is_final=False))

    for expected, got in ((expected_a, got_a), (expected_b, got_b)):
        for want, have in zip(expected, got):
            assert torch.allclose(want, have, atol=1e-5)


def test_a_session_is_independent_of_decoder_module_state() -> None:
    """Leaving a session must restore whatever was installed before it."""
    vocoder = _vocoder()
    decoder = vocoder._decoder
    decoder.reset_streaming_state()

    state = vocoder.create_stream_state("a")
    _feed(vocoder, "a", state, _codes(3))
    vocoder.decode_delta("a", state, is_final=False)

    # The request's buffers live in its session, not on the module.
    assert decoder._transformer_cache is None
    assert state.session["cache"] is not None
    assert any(value is not None for value in state.session["buffers"].values())


# --------------------------------------------------------------------------
# Streaming behaviour
# --------------------------------------------------------------------------


def test_streamed_deltas_concatenate_to_the_one_shot_decode() -> None:
    vocoder = _vocoder()
    frames = _codes(8, seed=7)

    state = vocoder.create_stream_state("a")
    deltas = []
    for start in range(0, 8, 2):
        _feed(vocoder, "a", state, frames[start : start + 2])
        delta = vocoder.decode_delta("a", state, is_final=False)
        if delta is not None:
            deltas.append(delta)
    streamed = torch.cat(deltas)

    whole = vocoder._decode_whole(frames.numpy().astype(np.int32), ref_frames=0)
    assert streamed.shape == whole.shape
    # Stateful streaming is the same computation as one pass, up to float noise.
    assert torch.allclose(streamed, whole, atol=2e-3)


def test_output_length_is_frames_times_upsample() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    _feed(vocoder, "a", state, _codes(5))
    delta = vocoder.decode_delta("a", state, is_final=False)
    assert delta.shape[0] == 5 * vocoder._decoder.total_upsample


def test_should_decode_waits_for_the_stride() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")

    _feed(vocoder, "a", state, _codes(1))
    assert not vocoder.should_decode(state, is_final=False)
    _feed(vocoder, "a", state, _codes(1, seed=9))
    assert vocoder.should_decode(state, is_final=False)
    # The final flush never waits.
    assert vocoder.should_decode(vocoder.create_stream_state("b"), is_final=True)


def test_decode_delta_is_none_without_buffered_frames() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    assert vocoder.decode_delta("a", state, is_final=True) is None


# --------------------------------------------------------------------------
# Reference priming (voice cloning)
# --------------------------------------------------------------------------


def test_reference_frames_prime_the_decoder_without_being_emitted() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    vocoder.latch_stream_contract(
        "a", state, {"ref_code_len": 3}, origin="stream metadata"
    )

    _feed(vocoder, "a", state, _codes(5))
    delta = vocoder.decode_delta("a", state, is_final=False)

    upsample = vocoder._decoder.total_upsample
    # 5 frames in, 3 of them reference: only 2 frames of audio come out.
    assert delta.shape[0] == 2 * upsample
    assert state.ref_frames_consumed == 3
    assert state.emitted_frames == 2


def test_reference_priming_spans_chunks() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    vocoder.latch_stream_contract(
        "a", state, {"ref_code_len": 4}, origin="stream metadata"
    )
    upsample = vocoder._decoder.total_upsample

    _feed(vocoder, "a", state, _codes(2))
    assert vocoder.decode_delta("a", state, is_final=False) is None
    assert state.ref_frames_consumed == 2

    _feed(vocoder, "a", state, _codes(3, seed=5))
    delta = vocoder.decode_delta("a", state, is_final=False)
    assert state.ref_frames_consumed == 4
    assert delta.shape[0] == 1 * upsample


def test_reference_frames_do_not_count_toward_the_stride() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    vocoder.latch_stream_contract(
        "a", state, {"ref_code_len": 4}, origin="stream metadata"
    )
    _feed(vocoder, "a", state, _codes(4))
    # All four frames are priming, so nothing is emittable yet.
    assert not vocoder.should_decode(state, is_final=False)


def test_ref_code_len_cannot_change_after_ingestion() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    vocoder.latch_stream_contract(
        "a", state, {"ref_code_len": 2}, origin="stream metadata"
    )
    _feed(vocoder, "a", state, _codes(3))
    with pytest.raises(ValueError, match="changed after"):
        vocoder.latch_stream_contract(
            "a", state, {"ref_code_len": 5}, origin="stream metadata"
        )


def test_negative_ref_code_len_is_rejected() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    with pytest.raises(ValueError, match="negative"):
        vocoder.latch_stream_contract(
            "a", state, {"ref_code_len": -1}, origin="stream metadata"
        )


# --------------------------------------------------------------------------
# Chunk validation
# --------------------------------------------------------------------------


def test_chunk_must_be_two_dimensional() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    with pytest.raises(ValueError, match=r"\[frames, quantizers\]"):
        vocoder.validate_chunk("a", state, torch.zeros(3, dtype=torch.long))


def test_quantizer_count_must_stay_constant() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    vocoder.validate_chunk("a", state, _codes(2))
    with pytest.raises(ValueError, match="quantizers"):
        vocoder.validate_chunk("a", state, torch.ones(2, GROUPS + 1, dtype=torch.long))


def test_out_of_range_codec_ids_are_rejected() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    bad = torch.full((2, GROUPS), 9999, dtype=torch.long)
    with pytest.raises(ValueError, match="outside"):
        vocoder.validate_chunk("a", state, bad)
    negative = torch.full((2, GROUPS), -1, dtype=torch.long)
    with pytest.raises(ValueError, match="outside"):
        vocoder.validate_chunk("a", state, negative)


def test_releasing_resources_drops_the_session() -> None:
    vocoder = _vocoder()
    state = vocoder.create_stream_state("a")
    _feed(vocoder, "a", state, _codes(3))
    vocoder.decode_delta("a", state, is_final=False)

    vocoder.release_stream_resources("a", state)
    assert state.session == {}
    assert state.pending == []
