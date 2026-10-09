# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.easymagpie_tts.codec import (
    CODEC_SUBDIR,
    EasyMagpieCodecConfig,
    FiniteScalarDequantizer,
    load_codec,
)


def test_dequantizer_maps_mixed_radix_digits_to_unit_levels() -> None:
    dequantizer = FiniteScalarDequantizer(1, [4, 4])
    # 9 = 1 + 2 * 4 -> digits (1, 2) -> ((1 - 2) / 2, (2 - 2) / 2)
    assert dequantizer(torch.tensor([[9]])).tolist() == [[-0.5, 0.0]]


def test_codec_config_rejects_mismatched_fsq_width(tiny_codec_config) -> None:
    with pytest.raises(ValueError, match="FSQ"):
        EasyMagpieCodecConfig(**{**tiny_codec_config, "input_dim": 5})


def test_decode_length_matches_stacked_frames(codec) -> None:
    assert codec.config.samples_per_stacked_frame == 2 * 2 * 3
    (audio,) = codec.decode_batch([torch.randint(0, 16, (5, 4))])
    assert audio.shape == (5 * 12,)
    assert float(audio.abs().max()) <= 1.0


def test_batched_decode_matches_single_decode(codec) -> None:
    long = torch.randint(0, 16, (7, 4))
    short = torch.randint(0, 16, (3, 4))
    batched = codec.decode_batch([long, short])
    torch.testing.assert_close(batched[0], codec.decode_batch([long])[0])
    torch.testing.assert_close(batched[1], codec.decode_batch([short])[0])


def test_chunked_stream_decode_matches_whole_utterance(codec) -> None:
    codes = torch.randint(0, 16, (15, 4))
    state = codec.empty_stream_state(1)
    pieces = []
    for start, end in ((0, 2), (2, 8), (8, 15)):
        audio, state = codec.stream(codes[None, start:end], state)
        assert audio.shape == (1, (end - start) * 12)
        pieces.append(audio[0])

    torch.testing.assert_close(torch.cat(pieces), codec.decode_batch([codes])[0])


def test_stream_state_rows_batch_independently(codec) -> None:
    first = torch.randint(0, 16, (5, 4))
    second = torch.randint(0, 16, (5, 4))
    _, warm = codec.stream(first[None, :3], codec.empty_stream_state(1))
    cold = codec.empty_stream_state(1)
    batched = [torch.cat(layer) for layer in zip(warm, cold)]

    audio, _ = codec.stream(torch.stack((first[3:], second[:2])), batched)

    torch.testing.assert_close(audio[0], codec.stream(first[None, 3:], warm)[0][0])
    torch.testing.assert_close(audio[1], codec.stream(second[None, :2], cold)[0][0])


def test_load_codec_is_strict_about_weight_names(
    tmp_path, codec, tiny_codec_config
) -> None:
    codec_dir = tmp_path / CODEC_SUBDIR
    codec_dir.mkdir()
    (codec_dir / "config.json").write_text(json.dumps(tiny_codec_config))
    weights = codec.state_dict()
    save_file(weights, str(codec_dir / "model.safetensors"))
    loaded = load_codec(str(tmp_path), "cpu")
    codes = torch.randint(0, 16, (4, 4))
    torch.testing.assert_close(
        loaded.decode_batch([codes]), codec.decode_batch([codes])
    )

    weights.pop("audio_decoder.post_conv.conv.bias")
    save_file(weights, str(codec_dir / "model.safetensors"))
    with pytest.raises(RuntimeError, match="post_conv"):
        load_codec(str(tmp_path), "cpu")
