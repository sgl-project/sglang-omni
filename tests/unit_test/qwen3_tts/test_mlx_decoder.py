# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for native MLX speech decoding and weight loading."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version
from math import prod
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    tree_flatten,
)

from sglang_omni.models.qwen3_tts.mlx.decoder import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxSpeechDecoder,
    Qwen3TTSMlxTokenizerConfig,
    load_qwen3_tts_mlx_decoder,
)
from sglang_omni.models.qwen3_tts.mlx.decoder_stream import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxDecoderStream,
)
from sglang_omni.models.qwen3_tts.mlx.decoder_transformer import (  # noqa: E402 - MLX is optional outside Apple Silicon.
    Qwen3TTSMlxDecoderConfig,
)

REFERENCE_MLX_AUDIO_VERSION = "0.4.6"


@pytest.fixture
def decoder_config() -> Qwen3TTSMlxTokenizerConfig:
    return Qwen3TTSMlxTokenizerConfig(
        decoder_config=Qwen3TTSMlxDecoderConfig(
            attention_bias=False,
            latent_dim=8,
            codebook_dim=8,
            codebook_size=16,
            decoder_dim=16,
            hidden_size=8,
            intermediate_size=16,
            layer_scale_initial_scale=0.01,
            head_dim=4,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_quantizers=3,
            num_semantic_quantizers=1,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            upsample_rates=[2],
            upsampling_ratios=[2],
        ),
        output_sample_rate=16000,
        decode_upsample_rate=4,
    )


def write_decoder_checkpoint(
    model_directory: Path,
    decoder: Qwen3TTSMlxSpeechDecoder,
    config: Qwen3TTSMlxTokenizerConfig,
) -> Path:
    tokenizer_directory = model_directory / "speech_tokenizer"
    tokenizer_directory.mkdir()
    (tokenizer_directory / "config.json").write_text(
        config.model_dump_json(), encoding="utf-8"
    )
    weights: dict[str, mx.array] = {}
    for name, weight in tree_flatten(decoder.parameters()):
        if ".codebooks." in name:
            quantizer_name, codebook_name = name.split(".codebooks.")
            codebook_index = codebook_name.removesuffix(".weight")
            name = f"{quantizer_name}.vq.layers.{codebook_index}.codebook.embed.weight"
        else:
            pass
        weights[f"decoder.{name}"] = weight
    mx.save_safetensors(str(tokenizer_directory / "model.safetensors"), weights)
    return tokenizer_directory


@pytest.mark.parametrize("frame_count", [1, 7, 301])
def test_decode_returns_finite_waveforms_and_valid_lengths(
    decoder_config: Qwen3TTSMlxTokenizerConfig, frame_count: int
) -> None:
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    codes = mx.ones((2, frame_count, 3), dtype=mx.int32)
    codes[1, -1, 0] = 0

    waveform, lengths = decoder.decode(codes)
    mx.eval(waveform, lengths)

    assert waveform.shape == (2, frame_count * 4)
    assert lengths.tolist() == [frame_count * 4, (frame_count - 1) * 4]
    assert mx.all(mx.isfinite(waveform)).item()
    assert mx.all(mx.abs(waveform) <= 1).item()


def test_decode_is_causal_and_independent_between_calls(
    decoder_config: Qwen3TTSMlxTokenizerConfig,
) -> None:
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    codes = mx.arange(21, dtype=mx.int32).reshape(1, 7, 3) % 15 + 1
    full_waveform, _ = decoder.decode(codes)
    prefix_waveform, _ = decoder.decode(codes[:, :3])
    repeated_waveform, _ = decoder.decode(codes)
    mx.eval(full_waveform, prefix_waveform, repeated_waveform)

    assert mx.allclose(
        prefix_waveform, full_waveform[:, :12], rtol=1e-5, atol=1e-5
    ).item()
    assert mx.array_equal(repeated_waveform, full_waveform).item()


def test_decode_uses_reloaded_convolution_weights(
    decoder_config: Qwen3TTSMlxTokenizerConfig,
) -> None:
    mx.random.seed(53)
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    codes = mx.arange(96, dtype=mx.int32).reshape(1, 32, 3) % 15 + 1
    original, _ = decoder.decode(codes)
    mx.eval(original)
    weights = [
        (name, weight * 0.5 if "block.1.conv.weight" in name else weight)
        for name, weight in tree_flatten(decoder.parameters())
    ]
    decoder.load_weights(weights)
    reloaded, _ = decoder.decode(codes)
    fresh_decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    fresh_decoder.load_weights(weights)
    expected, _ = fresh_decoder.decode(codes)
    mx.eval(reloaded, expected)

    assert not mx.allclose(reloaded, original).item()
    np.testing.assert_array_equal(np.asarray(reloaded), np.asarray(expected))


@pytest.mark.parametrize(
    ("weight_layout", "raw_codebooks"),
    [("native", False), ("pytorch", False), ("pytorch", True)],
)
def test_load_decoder_preserves_waveforms(
    tmp_path: Path,
    decoder_config: Qwen3TTSMlxTokenizerConfig,
    weight_layout: Literal["native", "pytorch"],
    raw_codebooks: bool,
) -> None:
    expected_decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    tokenizer_directory = write_decoder_checkpoint(
        tmp_path, expected_decoder, decoder_config
    )
    weights: dict[str, mx.array] = {}
    for name, weight in mx.load(str(tokenizer_directory / "model.safetensors")).items():
        if raw_codebooks and name.endswith(".codebook.embed.weight"):
            base_name = name.removesuffix(".codebook.embed.weight")
            weights[f"{base_name}._codebook.cluster_usage"] = mx.full(
                (weight.shape[0],), 2.0
            )
            weights[f"{base_name}._codebook.embedding_sum"] = weight * 2
        else:
            if weight_layout == "pytorch" and weight.ndim == 3:
                is_transpose_conv = (
                    "upsample" in name and ".0.conv.weight" in name
                ) or "block.1.conv.weight" in name
                weight = (
                    weight.transpose(2, 0, 1)
                    if is_transpose_conv
                    else weight.swapaxes(-1, -2)
                )
            else:
                pass
            weights[name] = weight
    weights["encoder.unused.weight"] = mx.zeros((1,))
    mx.save_safetensors(str(tokenizer_directory / "model.safetensors"), weights)
    codes = mx.ones((2, 5, 3), dtype=mx.int32)
    codes[1, 3:, 0] = 0

    loaded_decoder = load_qwen3_tts_mlx_decoder(tmp_path)
    expected_waveform, expected_lengths = expected_decoder.decode(codes)
    waveform, lengths = loaded_decoder.decode(codes)
    mx.eval(waveform, expected_waveform, lengths, expected_lengths)

    assert loaded_decoder.output_sample_rate == 16000
    assert mx.array_equal(waveform, expected_waveform).item()
    assert mx.array_equal(lengths, expected_lengths).item()


@pytest.mark.parametrize(
    ("batch_size", "frame_count", "upsample_rates", "upsampling_ratios"),
    [
        (1, 1, [2], [2]),
        (1, 7, [2], [2]),
        (2, 7, [2], [2]),
        (1, 300, [2], [2]),
        (2, 301, [2], [2]),
        (2, 601, [2], [2]),
        (1, 1, [8, 5, 4, 3], [3, 2]),
        (2, 301, [8, 5, 4, 3], [3, 2]),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [mx.float32, mx.float16, mx.bfloat16],
    ids=["float32", "float16", "bfloat16"],
)
@pytest.mark.parametrize(
    ("num_key_value_heads", "attention_bias"),
    [(1, False), (2, True)],
    ids=["grouped-query", "multi-head-with-bias"],
)
def test_loaded_decoder_matches_mlx_audio(
    tmp_path: Path,
    decoder_config: Qwen3TTSMlxTokenizerConfig,
    batch_size: int,
    frame_count: int,
    upsample_rates: list[int],
    upsampling_ratios: list[int],
    dtype: mx.Dtype,
    num_key_value_heads: int,
    attention_bias: bool,
) -> None:
    """Compare identical weights and codec frames against the original decoder."""
    pytest.importorskip(
        "mlx_audio", reason="Reference parity requires mlx-audio==0.4.6"
    )
    if version("mlx-audio") != REFERENCE_MLX_AUDIO_VERSION:
        pytest.skip(
            f"Reference parity requires mlx-audio=={REFERENCE_MLX_AUDIO_VERSION}"
        )
    else:
        pass
    from mlx_audio.tts.models.qwen3_tts.config import (
        Qwen3TTSTokenizerConfig,
        Qwen3TTSTokenizerDecoderConfig,
    )
    from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer

    decoder_settings = decoder_config.decoder_config.model_copy(
        update={
            "num_key_value_heads": num_key_value_heads,
            "attention_bias": attention_bias,
            "upsample_rates": upsample_rates,
            "upsampling_ratios": upsampling_ratios,
        }
    )
    config = decoder_config.model_copy(
        update={
            "decoder_config": decoder_settings,
            "decode_upsample_rate": prod(upsample_rates + upsampling_ratios),
        }
    )
    mx.random.seed(19)
    reference = Qwen3TTSSpeechTokenizer(
        Qwen3TTSTokenizerConfig(
            encoder_config=None,
            decoder_config=Qwen3TTSTokenizerDecoderConfig(
                **decoder_settings.model_dump()
            ),
            output_sample_rate=config.output_sample_rate,
            decode_upsample_rate=config.decode_upsample_rate,
        )
    )
    weights: list[tuple[str, mx.array]] = []
    for name, weight in tree_flatten(reference.parameters()):
        if name.endswith((".alpha", ".beta")):
            weight = mx.random.uniform(-0.5, 0.5, shape=weight.shape)
        else:
            pass
        weights.append((name, weight.astype(dtype)))
    reference.load_weights(weights, strict=True)
    reference.eval()
    tokenizer_directory = tmp_path / "speech_tokenizer"
    tokenizer_directory.mkdir()
    (tokenizer_directory / "config.json").write_text(
        config.model_dump_json(), encoding="utf-8"
    )
    mx.save_safetensors(str(tokenizer_directory / "model.safetensors"), dict(weights))
    decoder = load_qwen3_tts_mlx_decoder(tmp_path)
    codes = (
        mx.arange(
            batch_size * frame_count * decoder_settings.num_quantizers, dtype=mx.int32
        ).reshape(batch_size, frame_count, decoder_settings.num_quantizers)
        % (decoder_settings.codebook_size - 1)
        + 1
    )
    if batch_size > 1:
        codes[-1, frame_count // 2 :, 0] = 0
    else:
        pass

    expected_waveform, expected_lengths = reference.decode(codes)
    waveform, lengths = decoder.decode(codes)
    mx.eval(waveform, lengths, expected_waveform, expected_lengths)

    assert decoder.output_sample_rate == reference.output_sample_rate
    assert waveform.dtype == expected_waveform.dtype
    np.testing.assert_allclose(
        np.asarray(waveform.astype(mx.float32)),
        np.asarray(expected_waveform.astype(mx.float32)),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_array_equal(np.asarray(lengths), np.asarray(expected_lengths))
    assert mx.max(mx.abs(expected_waveform)).item() > 0


@pytest.mark.parametrize("shape", [(1, 0, 3), (1, 2, 2), (1, 2)])
@pytest.mark.parametrize("is_streaming", [False, True])
def test_decode_rejects_invalid_codec_frames(
    decoder_config: Qwen3TTSMlxTokenizerConfig,
    shape: tuple[int, ...],
    is_streaming: bool,
) -> None:
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    decode = (
        Qwen3TTSMlxDecoderStream(decoder).decode if is_streaming else decoder.decode
    )

    with pytest.raises(ValueError, match="codec frames|quantizers"):
        decode(mx.ones(shape, dtype=mx.int32))


def test_load_decoder_rejects_missing_weights(
    tmp_path: Path, decoder_config: Qwen3TTSMlxTokenizerConfig
) -> None:
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    tokenizer_directory = write_decoder_checkpoint(tmp_path, decoder, decoder_config)
    weights = mx.load(str(tokenizer_directory / "model.safetensors"))
    weights.pop("decoder.pre_conv.conv.weight")
    mx.save_safetensors(str(tokenizer_directory / "model.safetensors"), weights)

    with pytest.raises(ValueError, match="Missing|missing"):
        load_qwen3_tts_mlx_decoder(tmp_path)


def test_generator_and_decoder_work_without_mlx_audio(
    tmp_path: Path, decoder_config: Qwen3TTSMlxTokenizerConfig
) -> None:
    decoder = Qwen3TTSMlxSpeechDecoder(decoder_config)
    write_decoder_checkpoint(tmp_path, decoder, decoder_config)
    script = """
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

class RejectMlxAudio(MetaPathFinder):
    def find_spec(
        self, fullname: str, path: list[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname == "mlx_audio" or fullname.startswith("mlx_audio."):
            raise AssertionError(f"Unexpected dependency: {fullname}")
        else:
            return None

sys.meta_path.insert(0, RejectMlxAudio())
import mlx.core as mx
from sglang_omni.models.qwen3_tts.mlx.generate import Qwen3TTSMlxGenerator
from sglang_omni.models.qwen3_tts.mlx.decoder import load_qwen3_tts_mlx_decoder
decoder = load_qwen3_tts_mlx_decoder(Path(sys.argv[1]))
waveform, lengths = decoder.decode(mx.ones((1, 2, 3), dtype=mx.int32))
mx.eval(waveform, lengths)
assert waveform.shape == (1, 8)
assert lengths.tolist() == [8]
"""

    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
