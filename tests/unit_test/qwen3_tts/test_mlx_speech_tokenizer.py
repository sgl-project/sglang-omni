# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MLX Qwen3-TTS speech tokenizer, DSP, and sampling."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from sglang_omni.models.qwen3_tts.mlx.config import (  # noqa: E402
    SpeakerEncoderConfig,
    TokenizerConfig,
    TokenizerDecoderConfig,
    TokenizerEncoderConfig,
)
from sglang_omni.models.qwen3_tts.mlx.dsp import (  # noqa: E402
    mel_filters,
    mel_spectrogram,
)
from sglang_omni.models.qwen3_tts.mlx.sampling import (  # noqa: E402
    SamplingParams,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample_codec_token,
    special_codec_token_ids,
    suppress,
)
from sglang_omni.models.qwen3_tts.mlx.speaker_encoder import (  # noqa: E402
    Qwen3TTSSpeakerEncoder,
)
from sglang_omni.models.qwen3_tts.mlx.speech_tokenizer import (  # noqa: E402
    Qwen3TTSSpeechTokenizer,
    sliding_causal_mask,
)
from sglang_omni.models.qwen3_tts.mlx.weights import align_conv_weights  # noqa: E402


def _tiny_tokenizer_config(*, with_encoder: bool = True) -> TokenizerConfig:
    decoder = TokenizerDecoderConfig(
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
        num_quantizers=4,
        num_semantic_quantizers=1,
        sliding_window=6,
        upsample_rates=[2, 2],
        upsampling_ratios=[2],
    )
    encoder = (
        TokenizerEncoderConfig(
            hidden_size=8,
            intermediate_size=16,
            head_dim=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=1,
            num_filters=4,
            num_quantizers=4,
            num_semantic_quantizers=1,
            codebook_size=32,
            vector_quantization_hidden_dimension=8,
            upsampling_ratios=[2, 2],
            sliding_window=5,
            frame_rate=1500.0,
            sampling_rate=24000,
        )
        if with_encoder
        else None
    )
    return TokenizerConfig(
        encoder_config=encoder,
        decoder_config=decoder,
        encoder_valid_num_quantizers=4,
        decode_upsample_rate=8,
    )


# --------------------------------------------------------------------------
# Sliding-window mask
# --------------------------------------------------------------------------


def test_sliding_causal_mask_keeps_only_the_trailing_window() -> None:
    mask = sliding_causal_mask(4, 4, window=2, dtype=mx.float32)
    allowed = np.array(mask == 0.0)
    # Query i may see keys i and i-1 only.
    expected = np.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    assert np.array_equal(allowed, expected)


def test_sliding_causal_mask_offsets_queries_to_the_end() -> None:
    """With a cache, queries are the last rows of the key range."""
    mask = sliding_causal_mask(1, 5, window=3, dtype=mx.float32)
    allowed = np.array(mask == 0.0)
    # The single query is at absolute position 4 and sees keys 2, 3, 4.
    assert np.array_equal(allowed, np.array([[False, False, True, True, True]]))


def test_sliding_causal_mask_is_skipped_when_the_window_cannot_bind() -> None:
    assert sliding_causal_mask(1, 3, window=8, dtype=mx.float32) is None
    assert sliding_causal_mask(1, 3, window=None, dtype=mx.float32) is None
    # A window shorter than the history must still produce a mask.
    assert sliding_causal_mask(1, 9, window=8, dtype=mx.float32) is not None


def test_windowed_attention_ignores_history_beyond_the_window() -> None:
    """Perturbing positions outside the window must not move the last output.

    With a single layer the window is exactly ``sliding_window`` positions, so
    the final position cannot see anything before it -- while an unwindowed
    causal model would.
    """
    config = _tiny_tokenizer_config(with_encoder=False)
    mx.random.seed(0)
    tokenizer = Qwen3TTSSpeechTokenizer(config)
    transformer = tokenizer.decoder.pre_transformer
    mx.eval(tokenizer.parameters())
    assert len(transformer.layers) == 1

    window = config.decoder_config.sliding_window
    outside = 4
    latent = mx.random.normal((1, window + outside, config.decoder_config.latent_dim))
    perturbed = mx.concatenate(
        [
            mx.random.normal((1, outside, config.decoder_config.latent_dim)),
            latent[:, outside:, :],
        ],
        axis=1,
    )

    base = transformer(latent)
    changed = transformer(perturbed)
    mx.eval(base, changed)

    # Last position: unaffected by anything older than the window.
    assert float(mx.abs(base[:, -1, :] - changed[:, -1, :]).max()) < 1e-5
    # First position: it *is* one of the perturbed ones, so it must move.
    assert float(mx.abs(base[:, 0, :] - changed[:, 0, :]).max()) > 1e-4


# --------------------------------------------------------------------------
# Weight handling
# --------------------------------------------------------------------------


def test_sanitize_folds_codebook_statistics_into_a_table() -> None:
    tokenizer = Qwen3TTSSpeechTokenizer(_tiny_tokenizer_config(with_encoder=False))
    weights = {
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum": mx.array(
            [[2.0, 4.0], [9.0, 3.0]]
        ),
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage": mx.array(
            [2.0, 3.0]
        ),
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.initialized": mx.array(
            [1.0]
        ),
        "decoder.pre_conv.conv.bias": mx.zeros((4,)),
    }
    out = tokenizer.sanitize(weights)

    key = "decoder.quantizer.rvq_first.vq.layers.0.codebook.embed.weight"
    assert set(out) == {key, "decoder.pre_conv.conv.bias"}
    assert np.allclose(np.array(out[key]), [[1.0, 2.0], [3.0, 1.0]])


def test_sanitize_accepts_the_encoder_spelling_of_the_codebook() -> None:
    tokenizer = Qwen3TTSSpeechTokenizer(_tiny_tokenizer_config())
    weights = {
        "encoder.quantizer.semantic_residual_vector_quantizer.layers.0."
        "codebook.embed_sum": mx.array([[4.0, 2.0]]),
        "encoder.quantizer.semantic_residual_vector_quantizer.layers.0."
        "codebook.cluster_usage": mx.array([4.0]),
    }
    out = tokenizer.sanitize(weights)
    assert list(out) == [
        "encoder.quantizer.semantic_residual_vector_quantizer.layers.0."
        "codebook.embed.weight"
    ]
    assert np.allclose(np.array(next(iter(out.values()))), [[1.0, 0.5]])


def test_sanitize_drops_the_encoder_when_the_model_has_none() -> None:
    tokenizer = Qwen3TTSSpeechTokenizer(_tiny_tokenizer_config(with_encoder=False))
    assert not tokenizer.has_encoder
    out = tokenizer.sanitize(
        {
            "encoder.downsample.conv.weight": mx.zeros((4, 4, 2)),
            "decoder.pre_conv.conv.bias": mx.zeros((4,)),
        }
    )
    assert list(out) == ["decoder.pre_conv.conv.bias"]


def test_align_conv_weights_uses_module_type_not_shape() -> None:
    """A square transposed conv is ambiguous by shape; type resolves it."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.ConvTranspose1d(4, 4, 2, stride=2)
            self.down = nn.Conv1d(4, 6, 3)

    net = Net()
    mx.eval(net.parameters())
    # PyTorch layouts: ConvTranspose1d [in, out, k]; Conv1d [out, in, k].
    torch_up = mx.random.normal((4, 4, 2))
    torch_down = mx.random.normal((6, 4, 3))
    aligned = align_conv_weights(
        {"up.weight": torch_up, "down.weight": torch_down}, net
    )

    assert aligned["up.weight"].shape == net.up.weight.shape
    assert aligned["down.weight"].shape == net.down.weight.shape
    assert np.allclose(
        np.array(aligned["up.weight"]), np.array(mx.transpose(torch_up, (1, 2, 0)))
    )
    assert np.allclose(
        np.array(aligned["down.weight"]), np.array(mx.transpose(torch_down, (0, 2, 1)))
    )
    # Already-MLX weights pass through untouched.
    again = align_conv_weights(aligned, net)
    assert np.allclose(np.array(again["up.weight"]), np.array(aligned["up.weight"]))


# --------------------------------------------------------------------------
# Decoder / encoder shapes and streaming
# --------------------------------------------------------------------------


def test_decode_upsamples_by_the_product_of_every_rate() -> None:
    config = _tiny_tokenizer_config(with_encoder=False)
    tokenizer = Qwen3TTSSpeechTokenizer(config)
    mx.eval(tokenizer.parameters())

    frames = 7
    groups = config.decoder_config.num_quantizers
    codes = mx.random.randint(1, 32, (1, frames, groups))
    waveform, lengths = tokenizer.decode(codes)
    mx.eval(waveform, lengths)

    upsample = tokenizer.decoder.total_upsample
    assert upsample == 2 * 2 * 2
    assert waveform.shape == (1, frames * upsample)
    assert int(lengths[0]) == frames * config.decode_upsample_rate


def test_decoder_rejects_the_wrong_number_of_codebooks() -> None:
    config = _tiny_tokenizer_config(with_encoder=False)
    tokenizer = Qwen3TTSSpeechTokenizer(config)
    with pytest.raises(ValueError, match="codebooks"):
        tokenizer.decoder(mx.zeros((1, 2, 5), dtype=mx.int32))


def test_streaming_decode_yields_the_same_total_length() -> None:
    config = _tiny_tokenizer_config(with_encoder=False)
    tokenizer = Qwen3TTSSpeechTokenizer(config)
    mx.eval(tokenizer.parameters())
    codes = mx.random.randint(1, 32, (1, 9, config.decoder_config.num_quantizers))

    full, _ = tokenizer.decode(codes)
    chunks = list(tokenizer.streaming_decode(codes, chunk_tokens=3))
    streamed = mx.concatenate(chunks, axis=-1)
    mx.eval(full, streamed)

    assert len(chunks) == 3
    assert streamed.shape[-1] == full.shape[-1]


def test_encode_produces_valid_quantizer_count_and_frame_rate() -> None:
    config = _tiny_tokenizer_config()
    tokenizer = Qwen3TTSSpeechTokenizer(config)
    mx.eval(tokenizer.parameters())

    # upsampling_ratios [2, 2] -> stride 4; encoder frame rate 24000/4 = 6000,
    # downsampled by 6000/1500 = 4, so 64 samples -> 4 frames.
    codes = tokenizer.encode(mx.random.normal((1, 1, 64)))
    mx.eval(codes)
    assert codes.shape == (1, config.encoder_valid_num_quantizers, 4)
    assert int(codes.min()) >= 0
    assert int(codes.max()) < config.encoder_config.codebook_size


def test_encode_raises_without_an_encoder() -> None:
    tokenizer = Qwen3TTSSpeechTokenizer(_tiny_tokenizer_config(with_encoder=False))
    with pytest.raises(ValueError, match="no encoder"):
        tokenizer.encode(mx.zeros((1, 1, 64)))


# --------------------------------------------------------------------------
# Speaker encoder and DSP
# --------------------------------------------------------------------------


def test_speaker_encoder_pools_time_into_one_embedding() -> None:
    config = SpeakerEncoderConfig(
        mel_dim=16,
        enc_dim=12,
        # Final width is the sum of the three SE-Res2Net widths, as released.
        enc_channels=[16, 16, 16, 16, 48],
        enc_kernel_sizes=[3, 3, 3, 3, 1],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_attention_channels=8,
        enc_res2net_scale=4,
        enc_se_channels=8,
    )
    encoder = Qwen3TTSSpeakerEncoder(config)
    mx.eval(encoder.parameters())

    for frames in (20, 57):
        embedding = encoder(mx.random.normal((1, frames, 16)))
        mx.eval(embedding)
        assert embedding.shape == (1, 12)


def test_speaker_encoder_rejects_an_inconsistent_channel_plan() -> None:
    with pytest.raises(ValueError, match="sum of the"):
        Qwen3TTSSpeakerEncoder(
            SpeakerEncoderConfig(
                mel_dim=16,
                enc_channels=[16, 16, 24],
                enc_kernel_sizes=[3, 3, 1],
                enc_dilations=[1, 2, 1],
            )
        )


def test_mel_filterbank_matches_librosa() -> None:
    librosa_filters = pytest.importorskip("librosa.filters")
    reference = librosa_filters.mel(
        sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000
    )
    ours = np.array(mel_filters(24000, 1024, 128, 0.0, 12000.0))
    assert ours.shape == reference.shape
    assert np.abs(ours - reference).max() < 1e-6


def test_mel_spectrogram_frame_count_follows_the_hop() -> None:
    audio = mx.zeros((24000,))
    mels = mel_spectrogram(audio)
    mx.eval(mels)
    # (n_fft - hop) // 2 padding each side, then floor((len - n_fft)/hop) + 1.
    padded = 24000 + 2 * ((1024 - 256) // 2)
    assert mels.shape == (1, (padded - 1024) // 256 + 1, 128)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_top_k_keeps_exactly_k_entries() -> None:
    logits = mx.array([[3.0, 1.0, 2.0, 0.0]])
    kept = np.array(~np.isinf(np.array(apply_top_k(logits, 2))))
    assert kept.tolist() == [[True, False, True, False]]
    # A k at or beyond the vocabulary is a no-op.
    assert not np.isinf(np.array(apply_top_k(logits, 4))).any()


def test_top_p_keeps_the_token_that_crosses_the_threshold() -> None:
    # softmax([2, 1, 0]) ~ [0.665, 0.245, 0.090]
    logits = mx.array([[2.0, 1.0, 0.0]])
    kept = np.array(~np.isinf(np.array(apply_top_p(logits, 0.7))))
    assert kept.tolist() == [[True, True, False]]
    assert not np.isinf(np.array(apply_top_p(logits, 1.0))).any()


def test_repetition_penalty_divides_positive_and_scales_negative() -> None:
    logits = mx.array([[4.0, -4.0, 1.0]])
    out = np.array(apply_repetition_penalty(logits, [0, 1], 2.0, 64))
    assert np.allclose(out, [[2.0, -8.0, 1.0]])


def test_repetition_penalty_only_looks_at_the_recent_window() -> None:
    logits = mx.array([[4.0, 4.0]])
    out = np.array(apply_repetition_penalty(logits, [0, 1], 2.0, context_size=1))
    assert np.allclose(out, [[4.0, 2.0]])


def test_suppressed_tokens_are_never_sampled() -> None:
    logits = mx.array([[10.0, 0.0, 0.0]])
    out = np.array(suppress(logits, [0]))
    assert np.isinf(out[0, 0]) and out[0, 0] < 0
    token = sample_codec_token(
        logits, SamplingParams(temperature=0.0), suppress_tokens=[0]
    )
    mx.eval(token)
    assert int(token[0, 0]) != 0


def test_greedy_sampling_is_argmax_and_shape_is_batch_by_one() -> None:
    logits = mx.array([[1.0, 5.0, 2.0], [7.0, 0.0, 1.0]])
    token = sample_codec_token(logits, SamplingParams(temperature=0.0))
    mx.eval(token)
    assert token.shape == (2, 1)
    assert [int(token[i, 0]) for i in range(2)] == [1, 0]


def test_special_codec_token_ids_spans_the_reserved_block_minus_eos() -> None:
    ids = special_codec_token_ids(3072, keep=2150)
    assert min(ids) == 2048 and max(ids) == 3071
    assert 2150 not in ids
    assert len(ids) == 1023
