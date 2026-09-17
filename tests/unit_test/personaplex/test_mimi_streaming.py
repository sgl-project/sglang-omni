# SPDX-License-Identifier: Apache-2.0
"""Chunked Mimi must land on the samples a whole-sequence pass produces."""

from dataclasses import replace

import pytest
import torch

from sglang_omni.models.personaplex.architecture import MIMI
from sglang_omni.models.personaplex.components.causal_conv import (
    CausalConv1d,
    CausalConvTranspose1d,
)
from sglang_omni.models.personaplex.components.mimi import (
    AttentionState,
    MimiAttention,
    MimiCodec,
    MimiTransformer,
    rename_mimi_key,
)


def test_causal_conv_chunks_match_whole():
    torch.manual_seed(1)
    for kernel, stride, dilation, mode in (
        (7, 1, 1, "constant"),
        (8, 4, 1, "constant"),
        (3, 1, 2, "constant"),
        (4, 2, 1, "replicate"),
    ):
        conv = CausalConv1d(
            3, 5, kernel, stride=stride, dilation=dilation, pad_mode=mode
        )
        x = torch.randn(2, 3, 48)
        whole = conv(x)
        state = conv.init_state()
        chunks = [conv.step(x[..., i : i + 8], state) for i in range(0, 48, 8)]
        torch.testing.assert_close(torch.cat(chunks, -1), whole, atol=1e-6, rtol=1e-5)


def test_causal_conv_transpose_chunks_match_whole():
    torch.manual_seed(2)
    for kernel, stride, groups in ((16, 8, 1), (4, 2, 6)):
        convtr = CausalConvTranspose1d(6, 6, kernel, stride=stride, groups=groups)
        x = torch.randn(1, 6, 10)
        whole = convtr(x)
        state = convtr.init_state()
        chunks = [convtr.step(x[..., i : i + 1], state) for i in range(10)]
        torch.testing.assert_close(torch.cat(chunks, -1), whole, atol=1e-6, rtol=1e-5)


def test_codec_encode_and_decode_step_match_whole(random_codec):
    codec = random_codec
    frames = 5
    x = torch.randn(1, 1, codec.samples_per_frame * frames)
    codes = codec.encode(x)
    assert codes.shape == (1, 8, frames)
    state = codec.init_encode_state()
    step = codec.samples_per_frame
    chunked = torch.cat(
        [
            codec.encode_step(x[..., i : i + step], state)
            for i in range(0, x.shape[-1], step)
        ],
        -1,
    )
    assert torch.equal(chunked, codes)

    whole = codec.decode(codes)
    assert whole.shape == (1, 1, codec.samples_per_frame * frames)
    state = codec.init_decode_state()
    chunked = torch.cat(
        [codec.decode_step(codes[..., f : f + 1], state) for f in range(frames)], -1
    )
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-5)


def test_checkpoint_names_map_onto_the_module_tree():
    codec = MimiCodec()
    expected = set(codec.state_dict())
    checkpoint_names = [
        "encoder.model.0.conv.conv.weight",
        "encoder.model.1.block.1.conv.conv.bias",
        "decoder.model.2.convtr.convtr.weight",
        "downsample.conv.conv.conv.weight",
        "upsample.convtr.convtr.convtr.weight",
        "encoder_transformer.transformer.layers.0.self_attn.in_proj_weight",
        "decoder_transformer.transformer.layers.7.layer_scale_2.scale",
        "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        "quantizer.rvq_rest.vq.layers.6._codebook.cluster_usage",
    ]
    for name in checkpoint_names:
        assert rename_mimi_key(name) in expected, name
    assert (
        rename_mimi_key("quantizer.rvq_rest.vq.layers.7._codebook.embedding_sum")
        is None
    )


SMALL = replace(MIMI, context=6, num_layers=2, dim=16, num_heads=2, ffn_dim=8)


def _small_transformer() -> MimiTransformer:
    torch.manual_seed(4)
    transformer = MimiTransformer(SMALL).eval()
    with torch.no_grad():
        for parameter in transformer.parameters():
            parameter.normal_(std=0.2)
    return transformer


def _influenced_steps(chunk: int) -> list[int]:
    """Which steps still depend on step 0, feeding chunk steps at a time."""
    torch.manual_seed(0)
    attention = MimiAttention(dim=8, num_heads=2, context=SMALL.context, max_period=1e4)
    with torch.no_grad():
        attention.in_proj_weight.normal_()
        attention.out_proj.weight.normal_()
    x = torch.randn(1, 4 * SMALL.context, 8)
    changed = x.clone()
    changed[:, 0] += 1.0

    def run(inp):
        state = AttentionState()
        with torch.no_grad():
            parts = [
                attention(inp[:, t : t + chunk], offset=t, state=state)
                for t in range(0, inp.shape[1], chunk)
            ]
        return torch.cat(parts, 1)

    differs = (run(x) - run(changed)).abs().amax(dim=(0, 2)) > 1e-6
    return differs.nonzero().flatten().tolist()


def test_the_ring_drops_its_oldest_step_as_the_reference_does():
    """The reference labels the slot it is about to overwrite as a future position,
    so once the ring is full its oldest entry leaves the window."""
    assert _influenced_steps(1) == list(range(SMALL.context - 1))
    assert _influenced_steps(SMALL.frame_ratio) == list(
        range(SMALL.context - SMALL.frame_ratio)
    )


@pytest.mark.parametrize(
    "length", [SMALL.context - 1, SMALL.context, 3 * SMALL.context + 1]
)
def test_whole_sequence_matches_the_streaming_replay(length):
    transformer = _small_transformer()
    x = torch.randn(1, SMALL.dim, length)
    state = transformer.init_state()
    with torch.no_grad():
        whole = transformer(x)
        chunked = torch.cat(
            [
                transformer.step(x[..., t : t + SMALL.frame_ratio], state)
                for t in range(0, x.shape[-1], SMALL.frame_ratio)
            ],
            -1,
        )
    torch.testing.assert_close(whole, chunked, atol=1e-6, rtol=1e-5)
