# SPDX-License-Identifier: Apache-2.0
"""Conditioner construction and dtype handling for AuK."""

from types import SimpleNamespace

import torch


def test_conditioner_honors_requested_dtype(monkeypatch):
    import sys

    from sglang_omni.models.auk.reference_encode import AuKConditionEncoder

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

    class FakeModel(torch.nn.Module):
        _keys_to_ignore_on_load_unexpected = None

        @classmethod
        def from_pretrained(cls, path, torch_dtype):
            model = cls()
            model.loaded_dtype = torch_dtype
            return model

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(num_hidden_layers=1)
            )
            self.visual = object()
            self.lm_head = object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Qwen2_5OmniProcessor=FakeProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration=FakeModel,
        ),
    )

    encoder = AuKConditionEncoder("stub", dtype=torch.bfloat16)

    assert encoder.dtype is torch.bfloat16
    assert encoder.model.loaded_dtype is torch.bfloat16
    assert next(encoder.model.parameters()).dtype is torch.bfloat16
