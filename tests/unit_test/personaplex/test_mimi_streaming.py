# SPDX-License-Identifier: Apache-2.0
"""Chunked Mimi must land on the samples a whole-sequence pass produces."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

import sglang_omni.models.personaplex.components.mimi_transformer as mimi_transformer_module
from sglang_omni.models.personaplex.architecture import MIMI
from sglang_omni.models.personaplex.components.causal_conv import (
    CausalConv1d,
    CausalConvTranspose1d,
)
from sglang_omni.models.personaplex.components.mimi import MimiCodec, rename_mimi_key
from sglang_omni.models.personaplex.components.mimi_transformer import (
    AttentionState,
    MimiAttention,
    MimiTransformer,
)

CUDA_ONLY = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.cuda is not None),
    reason="NVIDIA CUDA is unavailable",
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


def small_transformer() -> MimiTransformer:
    torch.manual_seed(4)
    transformer = MimiTransformer(SMALL).eval()
    with torch.no_grad():
        for parameter in transformer.parameters():
            parameter.normal_(std=0.2)
    return transformer


def influenced_steps(chunk: int) -> list[int]:
    """Which steps still depend on step 0, feeding chunk steps at a time."""
    torch.manual_seed(0)
    attention = MimiAttention(
        dim=8, num_heads=2, context=SMALL.context, max_period=1e4, write_chunk=chunk
    )
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
    assert influenced_steps(1) == list(range(SMALL.context - 1))
    assert influenced_steps(SMALL.frame_ratio) == list(
        range(SMALL.context - SMALL.frame_ratio)
    )


@pytest.mark.parametrize(
    "length", [SMALL.context - 1, SMALL.context, 3 * SMALL.context + 1]
)
def test_whole_sequence_matches_the_streaming_replay(length):
    transformer = small_transformer()
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


@pytest.fixture
def cuda_mimi_attention() -> MimiAttention:
    torch.manual_seed(42)
    attention = (
        MimiAttention(
            MIMI.dim,
            MIMI.num_heads,
            MIMI.context,
            MIMI.rope_max_period,
            write_chunk=MIMI.frame_ratio,
        )
        .cuda()
        .eval()
    )
    for parameter in attention.parameters():
        nn.init.normal_(parameter, std=0.05)
    return attention


@pytest.mark.accelerator
@CUDA_ONLY
@pytest.mark.parametrize("initial_end_offset", [0, 248, 249, 250, 498, 500, 2048])
@torch.no_grad()
def test_cuda_attention_matches_eager_ring(
    cuda_mimi_attention: MimiAttention, initial_end_offset: int
) -> None:
    shape = (1, MIMI.num_heads, MIMI.context, MIMI.dim // MIMI.num_heads)
    keys = torch.randn(shape, device="cuda")
    values = torch.randn_like(keys)
    candidate, reference = [
        AttentionState(
            keys.clone() if initial_end_offset else None,
            values.clone() if initial_end_offset else None,
            initial_end_offset,
        )
        for _ in range(2)
    ]
    hidden_states = torch.randn(6, 1, MIMI.frame_ratio, MIMI.dim, device="cuda")
    comparisons: list[tuple[torch.Tensor, torch.Tensor]] = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for step_index, step_states in enumerate(hidden_states):
            offset = initial_end_offset + step_index * MIMI.frame_ratio
            with patch.object(
                mimi_transformer_module,
                "apply_interleaved_rope",
                side_effect=AssertionError,
            ):
                actual = cuda_mimi_attention(
                    step_states, offset=offset, state=candidate
                )
            with patch.object(torch.version, "cuda", None):
                expected = cuda_mimi_attention(
                    step_states, offset=offset, state=reference
                )
            comparisons.extend(
                zip(
                    (actual, candidate.keys.clone(), candidate.values.clone()),
                    (expected, reference.keys.clone(), reference.values.clone()),
                )
            )
            assert (
                candidate.end_offset
                == reference.end_offset
                == offset + MIMI.frame_ratio
            )
    torch.cuda.current_stream().wait_stream(stream)
    for actual, expected in comparisons:
        torch.testing.assert_close(
            actual.view(torch.int32), expected.view(torch.int32), atol=0, rtol=0
        )


@pytest.mark.accelerator
@CUDA_ONLY
@pytest.mark.parametrize(
    "batch_size,frame_count,dtype,has_strided_cache",
    [
        (2, 2, torch.float32, False),
        (1, 1, torch.float32, False),
        (1, 2, torch.float64, False),
        (1, 2, torch.float32, True),
    ],
)
@torch.no_grad()
def test_cuda_attention_unsupported_inputs_use_eager(
    cuda_mimi_attention: MimiAttention,
    batch_size: int,
    frame_count: int,
    dtype: torch.dtype,
    has_strided_cache: bool,
) -> None:
    attention = cuda_mimi_attention.to(dtype=dtype)
    hidden_states = torch.randn(
        batch_size, frame_count, MIMI.dim, device="cuda", dtype=dtype
    )
    if has_strided_cache:
        caches = [
            torch.randn(
                batch_size,
                MIMI.num_heads,
                MIMI.context,
                2 * MIMI.dim // MIMI.num_heads,
                device="cuda",
                dtype=dtype,
            )[..., : MIMI.dim // MIMI.num_heads]
            for _ in range(4)
        ]
        caches[2].copy_(caches[0])
        caches[3].copy_(caches[1])
        candidate = AttentionState(caches[0], caches[1])
        reference = AttentionState(caches[2], caches[3])
    else:
        candidate, reference = AttentionState(), AttentionState()
    with patch.object(
        mimi_transformer_module, "fused_mimi_rope_cache", side_effect=AssertionError
    ):
        actual = attention(hidden_states, state=candidate)
    with patch.object(torch.version, "cuda", None):
        expected = attention(hidden_states, state=reference)
    for actual, expected in (
        (actual, expected),
        (candidate.keys, reference.keys),
        (candidate.values, reference.values),
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert candidate.end_offset == reference.end_offset == frame_count


@pytest.mark.accelerator
@CUDA_ONLY
@pytest.mark.parametrize("is_streaming", [False, True])
def test_cuda_attention_preserves_gradients(
    cuda_mimi_attention: MimiAttention, is_streaming: bool
) -> None:
    hidden_states = torch.randn(
        1, MIMI.frame_ratio, MIMI.dim, device="cuda", requires_grad=True
    )
    with patch.object(
        mimi_transformer_module, "fused_mimi_rope_cache", side_effect=AssertionError
    ):
        cuda_mimi_attention(
            hidden_states, state=AttentionState() if is_streaming else None
        ).square().mean().backward()
    for tensor in (hidden_states, *cuda_mimi_attention.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
