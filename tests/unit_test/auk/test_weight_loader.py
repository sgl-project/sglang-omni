# SPDX-License-Identifier: Apache-2.0
"""Checkpoint selection and restoration, without constructing the audio models."""

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.auk.weight_loader import (
    load_dit_weights,
    load_vae_weights,
    resolve_weight_file,
)


@pytest.mark.parametrize(
    "files, expected",
    [
        (["model.safetensors", "auk_base.safetensors"], "auk_base.safetensors"),
        (["custom.safetensors", "vae.safetensors"], "custom.safetensors"),
        (["vae.safetensors"], None),
    ],
)
def test_selects_dit_checkpoint_without_confusing_it_with_vae(
    tmp_path, files, expected
):
    for name in files:
        (tmp_path / name).touch()
    if expected is None:
        with pytest.raises(FileNotFoundError):
            resolve_weight_file(str(tmp_path))
    else:
        assert resolve_weight_file(str(tmp_path)).name == expected


@pytest.mark.parametrize(
    "loader, filename, prefix",
    [
        (load_dit_weights, "auk_base.safetensors", ""),
        (load_dit_weights, "auk_base.safetensors", "ema_model."),
        (load_vae_weights, "vae.safetensors", "module."),
    ],
)
def test_restores_checkpoint_parameters(tmp_path, loader, filename, prefix):
    model = torch.nn.Linear(2, 2)
    expected = {
        name: torch.full_like(value, 3) for name, value in model.state_dict().items()
    }
    save_file(
        {prefix + name: value for name, value in expected.items()}, tmp_path / filename
    )

    report = loader(model, str(tmp_path))

    assert report.loaded == len(expected)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


@pytest.mark.parametrize(
    "loader, filename",
    [(load_dit_weights, "auk_base.safetensors"), (load_vae_weights, "vae.safetensors")],
)
def test_incomplete_checkpoint_is_rejected(tmp_path, loader, filename):
    model = torch.nn.Linear(2, 2)
    save_file({"weight": model.weight, "stale": torch.zeros(1)}, tmp_path / filename)
    with pytest.raises(RuntimeError, match="bias") as error:
        loader(model, str(tmp_path))
    assert "stale" in str(error.value)


def test_missing_vae_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vae_weights(torch.nn.Linear(2, 2), str(tmp_path))
