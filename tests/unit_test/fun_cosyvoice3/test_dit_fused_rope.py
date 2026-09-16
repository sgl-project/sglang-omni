# SPDX-License-Identifier: Apache-2.0
"""Native-oracle tests for partial RoPE and installed DiT execution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest.mock import Mock

import pytest
import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

if TYPE_CHECKING:
    from cosyvoice.flow.DiT.dit import DiT

# note (wirybeaver): Direct loading keeps CPU tests independent of SGLang.
_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni/models/fun_cosyvoice3/dit_fused_rope.py"
)
_SPEC = importlib.util.spec_from_file_location("cosyvoice3_dit_fused_rope", _PATH)
rope_impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rope_impl
_SPEC.loader.exec_module(rope_impl)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


def test_tables_follow_current_length() -> None:
    rotary = RotaryEmbedding(64)
    forward = rope_impl._RotaryTablesForward(rotary.forward_from_seq_len)
    for length in (73, 129, 73):
        tables = forward(length)
        freqs, _ = rotary.forward_from_seq_len(length)
        torch.testing.assert_close(tables.cos, freqs.cos(), rtol=0, atol=0)
        torch.testing.assert_close(tables.sin, freqs.sin(), rtol=0, atol=0)


def test_cpu_install_rejects_unsupported_device() -> None:
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        rope_impl.install_dit_fused_rope(torch.nn.Linear(1, 1))


@pytest.mark.accelerator
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_kernel_matches_native_across_shapes(dtype: torch.dtype) -> None:
    from sglang_omni.models.fun_cosyvoice3.dit_fused_rope_kernel import fused_qk_rope

    torch.manual_seed(7)
    rotary = RotaryEmbedding(64).cuda()
    with torch.inference_mode():
        for batch, length in ((2, 73), (6, 129), (2, 769)):
            q = torch.randn(batch, length, 1024, device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            originals = (q.clone(), k.clone())
            freqs, scale = rotary.forward_from_seq_len(length)
            outputs = fused_qk_rope(q, k, freqs.cos(), freqs.sin())
            for source, original, actual in zip((q, k), originals, outputs):
                torch.testing.assert_close(source, original, rtol=0, atol=0)
                expected = apply_rotary_pos_emb(original, freqs, scale)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert torch.equal(actual[..., 64:], original[..., 64:])
                assert not torch.equal(actual[..., :64], original[..., :64])


@pytest.fixture
def estimator() -> DiT:
    from cosyvoice.flow.DiT.dit import DiT

    torch.manual_seed(7)
    return DiT(dim=1024, depth=2, heads=16, dim_head=64, spk_dim=80).cuda().eval()


def _inputs(length: int) -> tuple[torch.Tensor, ...]:
    noisy_mel = torch.randn(2, 80, length, device="cuda")
    mask = torch.ones(2, 1, length, device="cuda")
    mask[0, :, -5:] = 0
    return (
        noisy_mel,
        mask,
        torch.randn_like(noisy_mel),
        torch.zeros(2, device="cuda"),
        torch.randn(2, 80, device="cuda"),
        torch.randn_like(noisy_mel),
    )


@pytest.mark.accelerator
@requires_cuda
@pytest.mark.parametrize("streaming", [False, True])
def test_installed_dit_preserves_output_weights_and_shares_tables(
    estimator: DiT, streaming: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sglang_omni.models.fun_cosyvoice3 import dit_fused_rope_kernel

    weights = {name: tensor.clone() for name, tensor in estimator.state_dict().items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        args = _inputs(73)
        expected = estimator(*args, streaming=streaming)
        rotary = Mock(wraps=estimator.rotary_embed.forward_from_seq_len)
        kernel = Mock(wraps=dit_fused_rope_kernel.fused_qk_rope)
        monkeypatch.setattr(estimator.rotary_embed, "forward_from_seq_len", rotary)
        monkeypatch.setattr(dit_fused_rope_kernel, "fused_qk_rope", kernel)
        rope_impl.install_dit_fused_rope(estimator)
        torch.testing.assert_close(estimator.state_dict(), weights, rtol=0, atol=0)
        torch.testing.assert_close(
            estimator(*args, streaming=streaming), expected, rtol=0, atol=0
        )
        rotary.assert_called_once_with(73)
        assert kernel.call_count == len(estimator.transformer_blocks)
        first, second = kernel.call_args_list
        assert first.args[2] is second.args[2]
        assert first.args[3] is second.args[3]


@pytest.mark.accelerator
@requires_cuda
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("execution", ["compile", "graph"])
def test_installed_dit_execution_uses_current_inputs(
    estimator: DiT,
    streaming: bool,
    execution: Literal["compile", "graph"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosyvoice.flow.DiT import dit

    rope_impl.install_dit_fused_rope(estimator)
    native_mask = dit.add_optional_chunk_mask
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # note (wirybeaver): Mask construction has host checks; capture the DiT tensor path.
        def prepared_mask(*args: object, **kwargs: object) -> torch.Tensor:
            return attention_mask

        run = (
            torch.compile(estimator, dynamic=True, fullgraph=True)
            if execution == "compile"
            else estimator
        )
        for length in (73, 129):
            args = _inputs(length)
            attention_mask = native_mask(
                args[0].transpose(1, 2),
                args[1].bool(),
                False,
                False,
                0,
                estimator.static_chunk_size if streaming else 0,
                -1,
            )
            monkeypatch.setattr(dit, "add_optional_chunk_mask", prepared_mask)
            if execution == "graph":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        estimator(*args, streaming=streaming)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = estimator(*args, streaming=streaming)
            for _ in range(2):
                args[0].normal_()
                args[3].fill_(0.5)
                expected = estimator(*args, streaming=streaming)
                if execution == "graph":
                    graph.replay()
                else:
                    output = run(*args, streaming=streaming)
                if execution == "graph":
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                else:
                    # note (wirybeaver): Inductor changes BF16 rounding outside RoPE.
                    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
