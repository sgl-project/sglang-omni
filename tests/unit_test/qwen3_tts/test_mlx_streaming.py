# SPDX-License-Identifier: Apache-2.0
"""Incremental decoder parity and streaming generation contracts."""

from collections.abc import Iterator
from math import prod

import numpy as np
import pytest
from transformers import PreTrainedTokenizerBase

mx = pytest.importorskip("mlx.core")
from mlx.utils import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    tree_flatten,
)
from sglang.srt.hardware_backend.mlx.kv_cache import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    ContiguousAttentionKVCache,
)

from sglang_omni.models.qwen3_tts.mlx.decoder import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxSpeechDecoder,
    Qwen3TTSMlxTokenizerConfig,
)
from sglang_omni.models.qwen3_tts.mlx.decoder_stream import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxDecoderStream,
)
from sglang_omni.models.qwen3_tts.mlx.decoder_transformer import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxDecoderConfig,
)
from sglang_omni.models.qwen3_tts.mlx.generate import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxGenerator,
)
from sglang_omni.models.qwen3_tts.mlx.model import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxArtifactConfig,
    Qwen3TTSMlxCodePredictor,
    Qwen3TTSMlxCodePredictorConfig,
    Qwen3TTSMlxTalker,
    Qwen3TTSMlxTalkerConfig,
)
from tests.unit_test.qwen3_tts.test_mlx_decoder import (  # noqa: E402 - Shared optional MLX fixture.
    decoder_config as decoder_config,
)


@pytest.mark.parametrize(
    ("batch_size", "frame_count", "chunk_frames", "upsample_rates"),
    [
        (1, 1, 4, [2]),
        (2, 17, 1, [2]),
        (2, 301, 7, [2]),
        (1, 601, 29, [2]),
        (1, 17, 4, [8, 5, 4, 3]),
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_streaming_decode_matches_offline(
    decoder_config: Qwen3TTSMlxTokenizerConfig,
    batch_size: int,
    frame_count: int,
    chunk_frames: int,
    upsample_rates: list[int],
    dtype: mx.Dtype,
) -> None:
    settings = decoder_config.decoder_config.model_copy(
        update={"upsample_rates": upsample_rates}
    )
    config = decoder_config.model_copy(
        update={
            "decoder_config": settings,
            "decode_upsample_rate": prod(upsample_rates + settings.upsampling_ratios),
        }
    )
    mx.random.seed(23)
    decoder = Qwen3TTSMlxSpeechDecoder(config)
    decoder.load_weights(
        [
            (name, weight.astype(dtype))
            for name, weight in tree_flatten(decoder.parameters())
        ]
    )
    codes = (
        mx.arange(batch_size * frame_count * 3, dtype=mx.int32).reshape(
            batch_size, frame_count, 3
        )
        % 15
        + 1
    )
    codes[-1, -1, 0] = 0
    expected_waveform, expected_lengths = decoder.decode(codes)
    stream = Qwen3TTSMlxDecoderStream(decoder)
    waveforms: list[mx.array] = []
    lengths = mx.zeros((batch_size,), dtype=mx.int32)
    for start in range(0, frame_count, chunk_frames):
        waveform, chunk_lengths = stream.decode(codes[:, start : start + chunk_frames])
        waveforms.append(waveform)
        lengths = lengths + chunk_lengths
    waveform = mx.concatenate(waveforms, axis=-1)
    mx.eval(waveform, lengths, expected_waveform, expected_lengths)

    np.testing.assert_allclose(
        np.asarray(waveform), np.asarray(expected_waveform), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(lengths), np.asarray(expected_lengths))


def test_decoder_streams_are_independent(
    decoder_config: Qwen3TTSMlxTokenizerConfig,
) -> None:
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    codes = mx.arange(51, dtype=mx.int32).reshape(1, 17, 3) % 15 + 1
    expected, _ = decoder.decode(codes)
    first = Qwen3TTSMlxDecoderStream(decoder)
    second = Qwen3TTSMlxDecoderStream(decoder)
    prefix, _ = first.decode(codes[:, :4])
    repeated, _ = second.decode(codes)
    suffix, _ = first.decode(codes[:, 4:])
    mx.eval(expected, prefix, suffix, repeated)

    np.testing.assert_allclose(np.asarray(repeated), np.asarray(expected), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(mx.concatenate([prefix, suffix], axis=-1)),
        np.asarray(expected),
        atol=1e-6,
    )


@pytest.fixture
def code_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> Qwen3TTSMlxGenerator:
    mx.random.seed(31)
    predictor_config = Qwen3TTSMlxCodePredictorConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=64,
        vocab_size=8,
        num_code_groups=3,
    )
    talker_config = Qwen3TTSMlxTalkerConfig(
        **(predictor_config.model_dump() | {"vocab_size": 1032}),
        code_predictor_config=predictor_config,
        text_hidden_size=8,
        text_vocab_size=8,
        codec_eos_token_id=1031,
        codec_think_id=0,
        codec_nothink_id=1,
        codec_think_bos_id=2,
        codec_think_eos_id=3,
        codec_pad_id=4,
        codec_bos_id=5,
        codec_language_id={},
        spk_id={"ryan": 6},
        spk_is_dialect={},
    )
    artifact = Qwen3TTSMlxArtifactConfig(
        tts_model_type="custom_voice",
        tts_model_size="0b6",
        talker_config=talker_config,
        tts_bos_token_id=0,
        tts_eos_token_id=1,
        tts_pad_token_id=2,
    )
    generator = Qwen3TTSMlxGenerator.__new__(Qwen3TTSMlxGenerator)
    generator.talker = Qwen3TTSMlxTalker(artifact)
    generator.talker.codec_head.weight = mx.zeros_like(
        generator.talker.codec_head.weight
    )
    generator.predictor = Qwen3TTSMlxCodePredictor(talker_config)
    generator.tokenizer = PreTrainedTokenizerBase()

    def build_prompt_embeddings(
        tokenizer: PreTrainedTokenizerBase, *, text: str, voice: str, language: str
    ) -> tuple[mx.array, mx.array, mx.array]:
        return mx.ones((1, 3, 8)), mx.zeros((1, 0, 8)), mx.zeros((1, 1, 8))

    monkeypatch.setattr(
        generator.talker, "build_prompt_embeddings", build_prompt_embeddings
    )
    return generator


def test_generation_predictor_matches_fresh_cache_for_every_frame(
    monkeypatch: pytest.MonkeyPatch, code_generator: Qwen3TTSMlxGenerator
) -> None:
    generator = code_generator
    forward_embeddings = generator.predictor.forward_embeddings
    reference_cache: list[ContiguousAttentionKVCache] = []

    def compare_prediction(
        embeddings: mx.array,
        *,
        cache: list[ContiguousAttentionKVCache],
        code_group: int,
    ) -> mx.array:
        if code_group == 0:
            reference_cache[:] = [
                ContiguousAttentionKVCache(max_seq_len=3)
                for _ in generator.predictor.model.layers
            ]
        else:
            pass
        expected = forward_embeddings(
            embeddings, cache=reference_cache, code_group=code_group
        )
        actual = forward_embeddings(embeddings, cache=cache, code_group=code_group)
        mx.eval(expected, actual)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        return actual

    monkeypatch.setattr(generator.predictor, "forward_embeddings", compare_prediction)
    for _ in range(2):
        frames = list(
            generator.generate_codes(
                text="Hello",
                voice="Ryan",
                language="English",
                max_new_tokens=4,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
            )
        )
        assert len(frames) == 4
        assert all(frame.shape == (1, 3) for frame in frames)


@pytest.mark.parametrize(
    ("frame_count", "chunk_frames", "padding_indices"),
    [(1, 4, []), (7, 4, []), (17, 1, [0, 8, 16]), (301, 7, [0, 299, 300])],
)
def test_streaming_generation_emits_early_and_flushes_the_tail(
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
    chunk_frames: int,
    padding_indices: list[int],
) -> None:
    mx.random.seed(29)
    generator = Qwen3TTSMlxGenerator.__new__(Qwen3TTSMlxGenerator)
    generator.decoder = Qwen3TTSMlxSpeechDecoder(
        Qwen3TTSMlxTokenizerConfig(
            decoder_config=Qwen3TTSMlxDecoderConfig(
                attention_bias=True,
                latent_dim=4,
                codebook_dim=4,
                codebook_size=16,
                decoder_dim=8,
                hidden_size=4,
                intermediate_size=8,
                layer_scale_initial_scale=0.01,
                head_dim=4,
                num_attention_heads=1,
                num_hidden_layers=1,
                num_key_value_heads=1,
                num_quantizers=2,
                num_semantic_quantizers=1,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                upsample_rates=[3],
                upsampling_ratios=[2],
            ),
            output_sample_rate=24000,
            decode_upsample_rate=6,
        )
    )
    codes = (
        mx.arange(frame_count * 2, dtype=mx.int32).reshape(1, frame_count, 2) % 15 + 1
    )
    for index in padding_indices:
        codes[:, index, 0] = 0
    consumed_frames: list[int] = []

    def generate_codes(
        *,
        text: str,
        voice: str,
        language: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> Iterator[mx.array]:
        for index in range(frame_count):
            consumed_frames.append(index)
            yield codes[:, index]

    monkeypatch.setattr(generator, "generate_codes", generate_codes)
    parameters = dict(
        text="Hello",
        voice="Ryan",
        language="English",
        max_new_tokens=frame_count,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    stream = generator.generate_stream(**parameters, chunk_frames=chunk_frames)
    first_waveform, first_count = next(stream)
    assert len(consumed_frames) == min(chunk_frames, frame_count)
    assert first_count == min(chunk_frames, frame_count)
    chunks = [(first_waveform, first_count), *stream]
    waveform = mx.concatenate([chunk[0] for chunk in chunks])
    expected, expected_count = generator.generate(**parameters)
    mx.eval(waveform, expected)

    assert chunks[-1][1] == expected_count == frame_count
    assert waveform.shape == ((frame_count - len(padding_indices)) * 6,)
    np.testing.assert_allclose(
        np.asarray(waveform), np.asarray(expected), rtol=1e-4, atol=1e-6
    )

    with pytest.raises(ValueError, match="chunk_frames must be positive"):
        next(generator.generate_stream(**parameters, chunk_frames=0))
    consumed_frames.clear()
    cancelled_stream = generator.generate_stream(
        **parameters, chunk_frames=chunk_frames
    )
    next(cancelled_stream)
    cancelled_stream.close()
    assert len(consumed_frames) == min(chunk_frames, frame_count)
