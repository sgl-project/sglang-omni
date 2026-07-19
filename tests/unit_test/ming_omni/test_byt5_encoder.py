# SPDX-License-Identifier: Apache-2.0
"""ByT5 render-text encoder tests for Ming image generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch.nn as nn  # noqa: E402

from sglang_omni.models.ming_omni.diffusion.byt5_encoder import (  # noqa: E402
    ByT5TextEncoder,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(
        self,
        texts,
        *,
        padding,
        max_length,
        truncation,
        add_special_tokens,
        return_tensors,
    ):
        self.calls.append(
            {
                "texts": texts,
                "padding": padding,
                "max_length": max_length,
                "truncation": truncation,
                "add_special_tokens": add_special_tokens,
                "return_tensors": return_tensors,
            }
        )
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long),
        )


class _FakeByT5Encoder(nn.Module):
    def forward(self, *, input_ids, attention_mask):
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeMapper(nn.Module):
    def forward(self, inputs_embeds, attention_mask):
        return inputs_embeds + 10.0


def test_format_render_text_matches_ming_prompt_contract() -> None:
    assert ByT5TextEncoder.format_render_text("SALE") == 'Text "SALE". '
    assert ByT5TextEncoder.format_render_text("") == ""


def test_encode_formats_text_and_zeroes_padding_rows() -> None:
    tokenizer = _FakeTokenizer()
    encoder = ByT5TextEncoder(_FakeByT5Encoder(), _FakeMapper())

    pos, neg = encoder.encode(
        ["SALE", ""],
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        max_length=4,
    )

    assert tokenizer.calls == [
        {
            "texts": ['Text "SALE". ', ""],
            "padding": "max_length",
            "max_length": 4,
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt",
        }
    ]
    assert len(pos) == 2
    assert len(neg) == 2
    assert pos[0].shape == (4, 3)
    assert pos[1].shape == (4, 3)
    torch.testing.assert_close(pos[0][0], torch.full((3,), 11.0))
    torch.testing.assert_close(pos[0][1], torch.full((3,), 12.0))
    torch.testing.assert_close(pos[0][2:], torch.zeros(2, 3))
    torch.testing.assert_close(pos[1][0], torch.full((3,), 13.0))
    torch.testing.assert_close(pos[1][1:], torch.zeros(3, 3))
    torch.testing.assert_close(neg[0], torch.zeros_like(pos[0]))
    torch.testing.assert_close(neg[1], torch.zeros_like(pos[1]))
