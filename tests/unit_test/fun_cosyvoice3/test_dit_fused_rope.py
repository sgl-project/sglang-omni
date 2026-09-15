# SPDX-License-Identifier: Apache-2.0
"""Partial-RoPE semantics can be checked without SGLang or a GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

# note (wirybeaver): Direct loading keeps CPU tests independent of SGLang.
_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni/models/fun_cosyvoice3/dit_fused_rope.py"
)
_SPEC = importlib.util.spec_from_file_location("cosyvoice3_dit_fused_rope", _PATH)
rope_impl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rope_impl)


def _torch_qk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate(x: torch.Tensor) -> torch.Tensor:
        prefix = x[..., : rope_impl._ROTARY_DIM].float()
        pairs = prefix.reshape(*prefix.shape[:-1], rope_impl._ROTARY_DIM // 2, 2)
        partner = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        rotated = prefix * cos + partner * sin
        return torch.cat((rotated, x[..., rope_impl._ROTARY_DIM :]), dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_rope_rotates_only_first_head(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    q = torch.randn(2, 73, 1024).to(dtype)
    k = torch.randn_like(q)
    rotary = RotaryEmbedding(64)
    freqs, scale = rotary.forward_from_seq_len(73)
    tables = rope_impl._RotaryTablesForward(rotary.forward_from_seq_len)(73)
    actual = _torch_qk_rope(q, k, tables.cos, tables.sin)
    for source, result in zip((q, k), actual):
        expected = apply_rotary_pos_emb(source, freqs, scale)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert torch.equal(result[..., 64:], source[..., 64:])
        assert not torch.equal(result[..., :64], source[..., :64])


@pytest.mark.parametrize("chunk_mask", [False, True])
def test_processor_preserves_sdpa_and_output_masks(chunk_mask: bool) -> None:
    torch.manual_seed(7)
    batch, length, width, heads = 2, 9, 1024, 16
    x = torch.randn(batch, length, width)
    attn = SimpleNamespace(
        heads=heads,
        inner_dim=width,
        to_q=torch.nn.Linear(width, width),
        to_k=torch.nn.Linear(width, width),
        to_v=torch.nn.Linear(width, width),
        to_out=[torch.nn.Linear(width, width), torch.nn.Identity()],
    )
    padding = torch.arange(length)[None, :] < torch.tensor([7, 9])[:, None]
    mask = padding
    if chunk_mask:
        mask = (
            padding[:, None, None, :]
            & torch.ones(length, length, dtype=torch.bool).tril()
        )
    rotary = RotaryEmbedding(64)
    freqs, scale = rotary.forward_from_seq_len(length)
    tables = rope_impl._RotaryTablesForward(rotary.forward_from_seq_len)(length)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fused(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((cos, sin))
        return _torch_qk_rope(q, k, cos, sin)

    processor = rope_impl._FusedRopeAttnProcessor(fused)
    with torch.inference_mode():
        q = apply_rotary_pos_emb(attn.to_q(x), freqs, scale)
        k = apply_rotary_pos_emb(attn.to_k(x), freqs, scale)
        v = attn.to_v(x)
        q, k, v = [
            t.reshape(batch, length, heads, 64).transpose(1, 2) for t in (q, k, v)
        ]
        expected = (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask if chunk_mask else mask[:, None, None, :]
            )
            .transpose(1, 2)
            .reshape(batch, length, width)
        )
        expected = attn.to_out[0](expected).masked_fill(~padding[..., None], 0)
        for _ in range(2):
            actual = processor(attn, x, mask=mask, rope=tables)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(cos is tables.cos and sin is tables.sin for cos, sin in calls)


def test_tables_follow_current_length_and_do_not_keep_a_shape_cache() -> None:
    rotary = RotaryEmbedding(64)
    forward = rope_impl._RotaryTablesForward(rotary.forward_from_seq_len)
    for length in (73, 129, 73):
        tables = forward(length)
        freqs, _ = rotary.forward_from_seq_len(length)
        torch.testing.assert_close(tables.cos, freqs.cos(), rtol=0, atol=0)
        torch.testing.assert_close(tables.sin, freqs.sin(), rtol=0, atol=0)


def test_cpu_install_fails_before_importing_cuda_dependencies() -> None:
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        rope_impl.install_dit_fused_rope(torch.nn.Linear(1, 1))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_kernel_matches_native_across_shapes(dtype: torch.dtype) -> None:
    from sglang_omni.models.fun_cosyvoice3.dit_fused_rope_kernel import fused_qk_rope

    torch.manual_seed(7)
    rotary = RotaryEmbedding(64).cuda()
    with torch.inference_mode():
        for batch, length in ((2, 73), (6, 129), (2, 769)):
            q = torch.randn(batch, length, 1024, device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            original_q, original_k = q.clone(), k.clone()
            freqs, scale = rotary.forward_from_seq_len(length)
            outputs = fused_qk_rope(q, k, freqs.cos(), freqs.sin())
            for source, actual in zip((q, k), outputs):
                expected = apply_rotary_pos_emb(source, freqs, scale)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.equal(actual[..., 64:], source[..., 64:])
                assert actual.dtype == dtype
            assert torch.equal(q, original_q) and torch.equal(k, original_k)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_compile_and_graph_replay_use_current_inputs() -> None:
    from sglang_omni.models.fun_cosyvoice3.dit_fused_rope_kernel import fused_qk_rope

    compiled = torch.compile(fused_qk_rope, dynamic=True, fullgraph=True)
    rotary = RotaryEmbedding(64).cuda()
    with torch.inference_mode():
        for length in (73, 129):
            q = torch.randn(2, length, 1024, device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            freqs, scale = rotary.forward_from_seq_len(length)
            cos, sin = freqs.cos(), freqs.sin()
            actual = compiled(q, k, cos, sin)
            for source, output in zip((q, k), actual):
                torch.testing.assert_close(
                    output, apply_rotary_pos_emb(source, freqs, scale), rtol=0, atol=0
                )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused_qk_rope(q, k, cos, sin)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_qk_rope(q, k, cos, sin)
        q.normal_()
        k.normal_()
        graph.replay()
        for source, output in zip((q, k), captured):
            torch.testing.assert_close(
                output, apply_rotary_pos_emb(source, freqs, scale), rtol=0, atol=0
            )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_real_dit_install_preserves_weights_and_streaming_outputs() -> None:
    from cosyvoice.flow.DiT.dit import DiT

    torch.manual_seed(7)
    estimator = DiT(dim=1024, depth=2, heads=16, dim_head=64, spk_dim=80).cuda().eval()
    keys = set(estimator.state_dict())
    inputs: list[tuple[tuple[torch.Tensor, ...], bool]] = []
    expected: list[torch.Tensor] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for streaming, length in ((False, 73), (True, 129)):
            x = torch.randn(2, 80, length, device="cuda")
            mask = torch.ones(2, 1, length, device="cuda")
            mask[0, :, -5:] = 0
            args = (
                x,
                mask,
                torch.randn_like(x),
                torch.zeros(2, device="cuda"),
                torch.randn(2, 80, device="cuda"),
                torch.randn_like(x),
            )
            inputs.append((args, streaming))
            expected.append(estimator(*args, streaming=streaming))
        rope_impl.install_dit_fused_rope(estimator)
        assert set(estimator.state_dict()) == keys
        for (args, streaming), reference in zip(inputs, expected):
            torch.testing.assert_close(
                estimator(*args, streaming=streaming), reference, rtol=0, atol=0
            )
