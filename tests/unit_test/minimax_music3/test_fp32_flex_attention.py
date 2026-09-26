# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from torch.nn import functional as F

import sglang_omni.models.minimax_music3.dit as dit_module
from sglang_omni.models.minimax_music3.acoustic import resolve_fp32_flex_attention
from sglang_omni.models.minimax_music3.config import MiniMaxMusic3PipelineConfig
from sglang_omni.models.minimax_music3.dit import Attention, RotaryEmbedding


def test_fp32_flex_attention_default_and_opt_out() -> None:
    config = MiniMaxMusic3PipelineConfig(model_path="/models/minimax")
    acoustic_factory = next(
        stage for stage in config.stages if stage.name == "dit_dav"
    ).factory

    assert acoustic_factory.fp32_flex_attention is None
    for requested_fp32_flex_attention, expected_enabled in (
        (acoustic_factory.fp32_flex_attention, True),
        (True, True),
        (False, False),
    ):
        assert (
            resolve_fp32_flex_attention(
                requested_fp32_flex_attention,
                device_type="cuda",
                dtype=torch.float32,
                attention_backend="torch_sdpa",
            )
            is expected_enabled
        )


@pytest.mark.parametrize(
    ("device_type", "dtype", "attention_backend"),
    [
        ("musa", torch.float32, "torch_sdpa"),
        ("cuda", torch.bfloat16, "torch_sdpa"),
        ("cuda", torch.float32, "auto"),
    ],
)
def test_incompatible_runtime_defaults_off_and_rejects_opt_in(
    device_type: str, dtype: torch.dtype, attention_backend: str
) -> None:
    assert not resolve_fp32_flex_attention(
        None, device_type=device_type, dtype=dtype, attention_backend=attention_backend
    )
    with pytest.raises(ValueError, match="fp32_flex_attention requires"):
        resolve_fp32_flex_attention(
            True,
            device_type=device_type,
            dtype=dtype,
            attention_backend=attention_backend,
        )


def test_flex_dispatch_preserves_sdpa_output(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(17)
    attention = Attention(
        128,
        dim_heads=64,
        compute_dtype=torch.float32,
        attention_backend="torch_sdpa",
        fp32_flex_attention=False,
    ).eval()
    rotary_frequencies, _ = RotaryEmbedding(32).forward_from_seq_len(7)
    x = torch.randn(2, 7, 128)
    flex_attention_mock = Mock(wraps=F.scaled_dot_product_attention)
    monkeypatch.setattr(dit_module, "flex_attention", flex_attention_mock)

    with torch.inference_mode(), set_forward_context(0, None):
        expected = attention(x, rotary_frequencies.cos(), rotary_frequencies.sin())
        flex_attention_mock.assert_not_called()
        attention.fp32_flex_attention = True
        actual = attention(x, rotary_frequencies.cos(), rotary_frequencies.sin())

    flex_attention_mock.assert_called_once()
    torch.testing.assert_close(actual, expected)
