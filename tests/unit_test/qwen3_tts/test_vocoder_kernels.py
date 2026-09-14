# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import sglang_omni.models.qwen3_tts.vocoder_kernels as vocoder_kernels


class _StubSnakeBeta(torch.nn.Module):
    """Stand-in with the qwen-tts SnakeBeta attribute layout."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_features = channels
        self.alpha = torch.nn.Parameter(torch.randn(channels) * 0.1)
        self.beta = torch.nn.Parameter(torch.randn(channels) * 0.1)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )


_StubSnakeBeta.__name__ = "SnakeBeta"


def test_fuse_vocoder_decoder_keeps_originals_on_prewarm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _StubSnakeBeta(4)
    second = _StubSnakeBeta(4)
    decoder = torch.nn.Sequential(first, torch.nn.Sequential(second))

    monkeypatch.setattr(vocoder_kernels, "_HAS_TRITON", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_prewarm(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("prewarm failed")

    monkeypatch.setattr(vocoder_kernels, "_prewarm_replacements", fail_prewarm)

    assert vocoder_kernels.fuse_vocoder_decoder(decoder) == 0
    assert decoder[0] is first
    assert decoder[1][0] is second


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused SnakeBeta parity needs CUDA"
)
@pytest.mark.parametrize(
    ("batch", "channels", "frames"),
    [
        (1, 96, 33),
        (2, 192, 96),
        (1, 384, 192),
        (1, 768, 257),
    ],
)
def test_fused_snake_beta_cuda_parity_uses_kernel(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
    channels: int,
    frames: int,
) -> None:
    # note (db-ol): on the accelerator runner a missing Triton must fail
    # loudly, a skip here would hide the kernel from CI again.
    assert vocoder_kernels._HAS_TRITON, "Triton is required on accelerator CI"

    torch.manual_seed(0)
    device = torch.device("cuda")
    original = _StubSnakeBeta(channels).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(
        (batch, channels, frames),
        device=device,
        dtype=torch.bfloat16,
    )
    expected = original(x)
    launches: list[tuple[int, int, int]] = []
    original_launch = vocoder_kernels._launch

    def record_launch(
        hidden_states: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        launches.append(tuple(hidden_states.shape))
        return original_launch(hidden_states, alpha, beta)

    monkeypatch.setattr(vocoder_kernels, "_launch", record_launch)

    actual = vocoder_kernels.fused_snake_beta(x, original.alpha, original.beta)

    assert actual is not None
    assert launches == [(batch, channels, frames)]
    assert torch.equal(actual, expected)
