# SPDX-License-Identifier: Apache-2.0
"""Device policy tests for the Ming audio encoder."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.ming_omni.components import audio_encoder


@pytest.mark.parametrize(
    ("device_type", "expected_enabled"),
    [
        ("cpu", False),
        ("cuda", True),
        ("npu", True),
    ],
)
def test_autocast_context_uses_tensor_device(
    monkeypatch,
    device_type: str,
    expected_enabled: bool,
) -> None:
    autocast = Mock()
    monkeypatch.setattr(torch, "autocast", autocast)
    tensor = SimpleNamespace(device=SimpleNamespace(type=device_type))

    audio_encoder._autocast_context(tensor)

    autocast.assert_called_once_with(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=expected_enabled,
    )
