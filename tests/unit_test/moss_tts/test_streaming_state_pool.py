# SPDX-License-Identifier: Apache-2.0
"""Native MOSS-Audio-Tokenizer decoder state, slot reuse and mixed streaming/offline regression tests."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioTokenizerVocoder


def _tiny_repository_vocoder(
    *, positional_embedding: str = "rope", context_duration: float = 4.0
) -> MossAudioTokenizerVocoder:
    config = {
        "sampling_rate": 8,
        "downsample_rate": 1,
        "number_channels": 1,
        "enable_channel_interleave": False,
        "quantizer_kwargs": {
            "input_dim": 4,
            "rvq_dim": 4,
            "output_dim": 4,
            "num_quantizers": 1,
            "codebook_size": 8,
            "codebook_dim": 4,
            "quantizer_type": "rlfq",
        },
        "decoder_kwargs": [
            {
                "module_type": "Transformer",
                "input_dimension": 4,
                "output_dimension": 4,
                "d_model": 4,
                "num_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 8,
                "causal": True,
                "positional_embedding": positional_embedding,
                "max_period": 10_000,
                "positional_scale": 1.0,
                "norm": "layer_norm",
                "gating": "none",
                "context_duration": context_duration,
            }
        ],
    }
    torch.manual_seed(7)
    model = MossAudioTokenizerVocoder(
        config,
        parameter_device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
        attention_backend="sdpa",
    )
    model.eval()
    return model


@pytest.mark.parametrize(
    "slots,lengths,valid,error",
    [
        ([0, 0], [3, 3], [True, True], "unique"),
        ([0, 2], [3, 3], [True, True], "real decoder state slots"),
        ([0, 1], [3, 2], [True, True], "full execution length"),
        ([0, 2], [3, 1], [True, False], "codes_lengths=0"),
    ],
)
def test_native_decoder_rejects_padded_or_aliased_rows(slots, lengths, valid, error):
    model = _tiny_repository_vocoder()
    model.initialize_decoder_state_pool(2, scratch_capacity=1)
    try:
        with pytest.raises(ValueError, match=error):
            model.decode_streaming_tensors(
                torch.zeros(1, 2, 3, dtype=torch.long),
                torch.tensor(lengths),
                torch.tensor(slots),
                torch.tensor(valid),
            )
    finally:
        model.close_decoder_state_pool()


def test_repository_codec_native_pool_keeps_compact_slots_isolated() -> None:
    from sglang_omni.models.moss_tts_local.streaming_vocoder import _CodecStreamSession

    session = _CodecStreamSession(_tiny_repository_vocoder(), stream_slots=4, n_vq=1)
    first, second, third = [session.acquire() for _ in range(3)]
    references = {
        slot: _CodecStreamSession(_tiny_repository_vocoder(), stream_slots=1, n_vq=1)
        for slot in (first, second, third)
    }
    for reference in references.values():
        assert reference.acquire() == 0

    def check_step(slots, length):
        codes = {slot: torch.randint(0, 8, (1, length)) for slot in slots}
        actual = session.step(codes)
        for slot in slots:
            expected = references[slot].step({0: codes[slot]})[0]
            torch.testing.assert_close(actual[slot], expected)

    try:
        check_step([third, first, second], 2)
        check_step([first, third], 5)
        check_step([second], 1)
        # note (Zhang Yiyang): Reusing a completed request's slot must leave
        # the other requests' histories intact, including idle requests.
        session.release(first)
        assert session.acquire() == first
        references[first].close()
        references[first] = _CodecStreamSession(
            _tiny_repository_vocoder(), stream_slots=1, n_vq=1
        )
        assert references[first].acquire() == 0
        check_step([third, first], 3)
        check_step([second, third, first], 2)
    finally:
        session.close()
        for reference in references.values():
            reference.close()


@pytest.mark.parametrize("offline_batch_size", [1, 3, 4])
def test_native_scheduler_keeps_offline_decode_out_of_streaming_state(
    offline_batch_size: int,
) -> None:
    from sglang_omni.models.moss_tts_local.streaming_vocoder import (
        MossTTSLocalStreamingVocoderScheduler,
    )

    model = _tiny_repository_vocoder(positional_embedding="rope")
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        model,
        n_vq=1,
        sample_rate=8,
        attention_backend="sdpa",
        stream_slots=3,
        max_batch_size=4,
        vocoder_cuda_graph=False,
    )
    rows = [
        torch.arange(2 + i).remainder(8).view(-1, 1) for i in range(offline_batch_size)
    ]
    expected_offline = scheduler._decode_codes_rows(rows)
    session = scheduler._ensure_session()
    first = {2: torch.tensor([[1, 2, 3]])}
    second = {2: torch.tensor([[4, 5]])}
    try:
        session.step(first)
        expected_stream = session.step(second)[2]
        session._reset_slots([2])
        session.step(first)
        # Non-streaming traffic can arrive at any batch width while a stream
        # owns its slot. Its output and the stream's continuation must not change.
        actual_offline = scheduler._decode_codes_rows(rows)
        actual_stream = session.step(second)[2]
        for actual, expected in zip(actual_offline, expected_offline, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_stream, expected_stream, rtol=0, atol=0)
        # The wrappers isolate state without duplicating model weights.
        assert list(scheduler._nonstream_decoder.parameters()) == list(
            model.decoder.parameters()
        )
    finally:
        session.close()


def test_indexed_attention_matches_sequential_across_chunk_sizes() -> None:
    model = _tiny_repository_vocoder(positional_embedding="rope", context_duration=0.5)
    reference = _tiny_repository_vocoder(
        positional_embedding="rope", context_duration=0.5
    )
    model.initialize_decoder_state_pool(3)
    try:
        with reference.streaming(1), torch.no_grad():
            for length in [2, 5, 1, 2, 1, 7, 1, 1]:
                codes = torch.randint(0, 8, (1, 1, length))
                # note (Zhang Yiyang): Keep the reference on sequential
                # attention, independent of indexed state-pool execution.
                hidden = reference.quantizer.decode_codes(codes)
                expected_audio, expected_lengths = reference.decoder(
                    hidden,
                    torch.tensor([length]),
                )
                actual = model.decode_streaming_tensors(
                    codes,
                    torch.tensor([length]),
                    torch.tensor([2]),
                    torch.tensor([True]),
                )
                torch.testing.assert_close(actual[0], expected_audio)
                torch.testing.assert_close(actual[1], expected_lengths)
    finally:
        model.close_decoder_state_pool()
