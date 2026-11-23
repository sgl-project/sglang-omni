# SPDX-License-Identifier: Apache-2.0
"""Audio encoder contracts checked against fixed reference outputs."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from sglang_omni.modules.ming_audio.audio_encoder import (
    AudioEncoderConfig,
    Ming3AudioEncoder,
)


@pytest.mark.parametrize(
    ("mel_frames", "expected_tokens"),
    [(1, 1), (16, 1), (17, 2), (2999, 188), (3000, 188), (3001, 189), (6000, 376)],
)
def test_audio_token_count(mel_frames: int, expected_tokens: int) -> None:
    config = AudioEncoderConfig(
        n_mels=4,
        n_audio_ctx=3000,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3, 3, 3],
        downsampler_stride_list=[2, 2, 2],
    )
    assert config.output_length(mel_frames) == expected_tokens


def test_encoder_matches_reference() -> None:
    reference = load_file(
        str(Path(__file__).parent / "fixtures/audio_encoder.safetensors")
    )
    config = AudioEncoderConfig(
        n_mels=4,
        n_audio_ctx=32,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3, 3, 3],
        downsampler_stride_list=[2, 2, 2],
        whisper_chunk_size=32,
    )
    encoder = Ming3AudioEncoder(
        config,
        hidden_size=16,
        should_upcast_layer_norm=False,
        should_batch_chunks=True,
    ).eval()
    encoder.load_state_dict(
        {
            name.removeprefix("weights."): value
            for name, value in reference.items()
            if name.startswith("weights.")
        }
    )
    with torch.inference_mode():
        actual = encoder([reference["features.0"], reference["features.1"]])
    torch.testing.assert_close(actual, reference["expected"], rtol=1e-5, atol=1e-6)


def test_padded_batch_matches_separate_mels() -> None:
    torch.manual_seed(0)
    config = AudioEncoderConfig(
        n_mels=4,
        n_audio_ctx=32,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3],
        downsampler_stride_list=[2],
        whisper_chunk_size=32,
    )
    encoder = Ming3AudioEncoder(
        config,
        hidden_size=16,
        should_upcast_layer_norm=True,
        should_batch_chunks=False,
    ).eval()
    long_mel = torch.randn(20, config.n_mels)
    short_mel = torch.randn(9, config.n_mels)
    with torch.inference_mode():
        batched = encoder.embed_mels([long_mel, short_mel])
        separate = [
            encoder.embed_mels([long_mel])[0],
            encoder.embed_mels([short_mel])[0],
        ]
    assert [row.shape[0] for row in batched] == [
        config.output_length(20),
        config.output_length(9),
    ]
    torch.testing.assert_close(batched[0], separate[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batched[1], separate[1], rtol=1e-5, atol=1e-5)


def test_unbatched_chunks_encode_independently_of_neighbors() -> None:
    torch.manual_seed(0)
    config = AudioEncoderConfig(
        n_mels=4,
        n_audio_ctx=32,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        downsampler_kernel_size_list=[3],
        downsampler_stride_list=[2],
        whisper_chunk_size=32,
    )
    encoder = Ming3AudioEncoder(
        config,
        hidden_size=16,
        should_upcast_layer_norm=True,
        should_batch_chunks=False,
    ).eval()
    long_clip = torch.randn(40, config.n_mels)
    short_clip = torch.randn(9, config.n_mels)
    with torch.inference_mode():
        together = encoder([long_clip, short_clip])
        separate = torch.cat(
            [
                encoder([long_clip[:32]]),
                encoder([long_clip[32:]]),
                encoder([short_clip]),
            ]
        )
    assert together.shape[0] == config.output_length(40) + config.output_length(9)
    torch.testing.assert_close(together, separate, rtol=0, atol=0)
