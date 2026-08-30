# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.nemotron_voicechat.audio_runtime import (
    _copy_checkpoint_prefix,
    _validate_max_sessions,
)


def test_checkpoint_prefix_requires_exact_module_coverage(tmp_path) -> None:
    source = torch.nn.Linear(2, 3)
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {f"voice.{name}": tensor for name, tensor in source.state_dict().items()},
        checkpoint,
    )
    target = torch.nn.Linear(2, 3)

    assert _copy_checkpoint_prefix(target, checkpoint, "voice.") == 2
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], tensor)


def test_checkpoint_prefix_rejects_missing_or_unexpected_tensors(tmp_path) -> None:
    module = torch.nn.Linear(2, 3)
    missing = tmp_path / "missing.safetensors"
    save_file({"voice.weight": module.weight.detach()}, missing)
    with pytest.raises(ValueError, match="missing 1 module tensors"):
        _copy_checkpoint_prefix(module, missing, "voice.")

    unexpected = tmp_path / "unexpected.safetensors"
    tensors = {f"voice.{name}": tensor for name, tensor in module.state_dict().items()}
    tensors["voice.extra"] = torch.zeros(1)
    save_file(tensors, unexpected)
    with pytest.raises(ValueError, match="unexpected 1 checkpoint tensors"):
        _copy_checkpoint_prefix(module, unexpected, "voice.")


def test_max_sessions_must_be_positive() -> None:
    assert _validate_max_sessions(1) == 1
    with pytest.raises(ValueError, match="must be positive"):
        _validate_max_sessions(0)
