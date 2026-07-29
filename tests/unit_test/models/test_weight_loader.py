# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from torch import nn

from sglang_omni.models import weight_loader


def test_non_strict_module_load_can_require_every_local_key(monkeypatch) -> None:
    module = nn.Linear(2, 2)
    monkeypatch.setattr(
        weight_loader,
        "load_weights_by_prefix",
        lambda *args, **kwargs: {"weight": torch.ones(2, 2)},
    )

    with pytest.raises(RuntimeError, match="missing required module keys"):
        weight_loader.load_module(
            module,
            "model",
            prefix="tts.",
            strict=False,
            require_all_module_keys=True,
        )


def test_non_strict_module_load_allows_only_named_extra_prefixes(monkeypatch) -> None:
    module = nn.Linear(2, 2)
    state = {
        "weight": torch.ones(2, 2),
        "bias": torch.zeros(2),
        "projector_spk.weight": torch.ones(1),
    }
    monkeypatch.setattr(
        weight_loader,
        "load_weights_by_prefix",
        lambda *args, **kwargs: state,
    )

    loaded = weight_loader.load_module(
        module,
        "model",
        prefix="tts.",
        strict=False,
        require_all_module_keys=True,
        allowed_unexpected_prefixes=("projector_spk.",),
    )

    torch.testing.assert_close(loaded.weight, torch.ones(2, 2))
    torch.testing.assert_close(loaded.bias, torch.zeros(2))


def test_non_strict_module_load_rejects_unknown_extra_keys(monkeypatch) -> None:
    module = nn.Linear(2, 2)
    state = {
        "weight": torch.ones(2, 2),
        "bias": torch.zeros(2),
        "unknown.weight": torch.ones(1),
    }
    monkeypatch.setattr(
        weight_loader,
        "load_weights_by_prefix",
        lambda *args, **kwargs: state,
    )

    with pytest.raises(RuntimeError, match="unsupported unexpected module keys"):
        weight_loader.load_module(
            module,
            "model",
            prefix="tts.",
            strict=False,
            require_all_module_keys=True,
            allowed_unexpected_prefixes=("projector_spk.",),
        )
