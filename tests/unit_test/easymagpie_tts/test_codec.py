# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.easymagpie_tts.codec import (
    CODEC_SUBDIR,
    EasyMagpieCodec,
    EasyMagpieCodecConfig,
    FiniteScalarDequantizer,
    load_codec,
)

TINY_CODEC = {
    "input_dim": 4,
    "input_filters": 8,
    "hidden_filters": 8,
    "num_hidden_layers": 1,
    "pre_upsample_rates": [2],
    "pre_upsample_filters": [8],
    "resblock_upsample_rates": [3],
    "resblock_upsample_filters": [4],
    "kernel_size": 3,
    "resblock_kernel_size": 3,
    "num_codebooks": 2,
    "codebook_size": 16,
    "num_levels_per_group": [4, 4],
    "frame_stacking_factor": 2,
}


@pytest.fixture
def codec() -> EasyMagpieCodec:
    torch.manual_seed(0)
    return EasyMagpieCodec(EasyMagpieCodecConfig(**TINY_CODEC)).eval()


def test_dequantizer_maps_mixed_radix_digits_to_unit_levels() -> None:
    dequantizer = FiniteScalarDequantizer(1, [4, 4])
    # 9 = 1 + 2 * 4 -> digits (1, 2) -> ((1 - 2) / 2, (2 - 2) / 2)
    assert dequantizer(torch.tensor([[9]])).tolist() == [[-0.5, 0.0]]


def test_codec_config_rejects_mismatched_fsq_width() -> None:
    with pytest.raises(ValueError, match="FSQ"):
        EasyMagpieCodecConfig(**{**TINY_CODEC, "input_dim": 5})


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


def test_load_codec_is_strict_about_weight_names(tmp_path, codec) -> None:
    codec_dir = tmp_path / CODEC_SUBDIR
    codec_dir.mkdir()
    (codec_dir / "config.json").write_text(json.dumps(TINY_CODEC))
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
