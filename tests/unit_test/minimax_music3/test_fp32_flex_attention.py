# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import pytest
import torch

import sglang_omni.models.minimax_music3.dit as dit_module
from sglang_omni.models.minimax_music3.config import MiniMaxMusic3PipelineConfig
from sglang_omni.models.minimax_music3.dit import Attention
from sglang_omni.models.minimax_music3.stages import create_dit_dav_executor


def test_fp32_flex_attention_defaults_false() -> None:
    config = MiniMaxMusic3PipelineConfig(model_path="/models/minimax")
    factory = next(stage for stage in config.stages if stage.name == "dit_dav").factory

    assert factory.fp32_flex_attention is False
    assert factory.dtype == "float32"
    assert factory.attention_backend == "torch_sdpa"
    assert (
        inspect.signature(create_dit_dav_executor)
        .parameters["fp32_flex_attention"]
        .default
        is False
    )


def test_false_path_uses_original_attention_backend() -> None:
    module = Attention(
        128,
        dim_heads=64,
        compute_dtype=torch.float32,
        attention_backend="torch_sdpa",
    )
    calls = []

    class Backend(torch.nn.Module):
        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            calls.append((q.shape, k.shape, v.shape))
            return v

    module.backend = Backend()
    x = torch.randn(2, 7, 128)

    assert module(x, torch.ones(7, 32), torch.zeros(7, 32)).shape == x.shape
    assert calls == [
        (
            torch.Size((2, 7, 2, 64)),
            torch.Size((2, 7, 2, 64)),
            torch.Size((2, 7, 2, 64)),
        )
    ]


def test_true_path_uses_head_major_flex_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_flex_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        calls.append((q.shape, k.shape, v.shape))
        return v

    monkeypatch.setattr(dit_module, "flex_attention", fake_flex_attention)
    module = Attention(
        128,
        dim_heads=64,
        compute_dtype=torch.float32,
        attention_backend="torch_sdpa",
        fp32_flex_attention=True,
    )
    x = torch.randn(2, 7, 128)

    assert module(x, torch.ones(7, 32), torch.zeros(7, 32)).shape == x.shape
    assert calls == [
        (
            torch.Size((2, 2, 7, 64)),
            torch.Size((2, 2, 7, 64)),
            torch.Size((2, 2, 7, 64)),
        )
    ]


@pytest.mark.parametrize(
    ("compute_dtype", "attention_backend"),
    [
        (torch.bfloat16, "torch_sdpa"),
        (torch.float32, "auto"),
    ],
)
def test_fp32_flex_attention_rejects_unvalidated_runtime_combinations(
    compute_dtype: torch.dtype, attention_backend: str
) -> None:
    with pytest.raises(ValueError, match="requires dtype=float32"):
        Attention(
            128,
            dim_heads=64,
            compute_dtype=compute_dtype,
            attention_backend=attention_backend,
            fp32_flex_attention=True,
        )
