# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PEFT / safetensors LoRA checkpoint loading on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.lora.checkpoint import (
    discover_solver_override,
    infer_target_modules_from_tensors,
    load_lora_checkpoint,
)
from sglang_omni.lora.solver_override import ODESolverOverride


def test_load_peft_directory_safetensors_and_embedded_solver_override(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "flash_adapter"
    adapter_dir.mkdir()
    lora_a = torch.randn(4, 16)
    lora_b = torch.randn(16, 4)
    save_file(
        {
            "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": lora_a,
            "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 4,
                "lora_alpha": 8.0,
                "target_modules": ["to_q", "to_v"],
                "ode_solver": {
                    "steps": 4,
                    "cfg_strength": 1.0,
                    "clear_sway_sampling_coef": True,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_lora_checkpoint(adapter_dir)
    assert loaded.source.adapter_id == "flash_adapter"
    assert loaded.source.target_modules == ("to_q", "to_v")
    assert loaded.source.alpha == 8.0
    assert loaded.config["r"] == 4
    assert loaded.solver_override == ODESolverOverride(
        num_steps=4,
        cfg_scale=1.0,
        clear_sway_sampling_coef=True,
    )
    named_map = dict(loaded.source.named_tensors)
    assert len(named_map) == 2
    assert torch.equal(
        named_map["base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight"],
        lora_a,
    )


def test_solver_override_precedence_sidecar_and_explicit(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "distill_v2"
    adapter_dir.mkdir()
    save_file(
        {
            "dit.to_q.lora_A.weight": torch.randn(2, 8),
            "dit.to_q.lora_B.weight": torch.randn(8, 2),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 4.0,
                "target_modules": ["to_q"],
                "ode_solver": {"num_steps": 12, "cfg_scale": 2.5},
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "ode_solver_config.json").write_text(
        json.dumps({"num_steps": 4, "cfg_scale": 1.0}),
        encoding="utf-8",
    )

    # Sidecar overrides embedded adapter_config.json.
    sidecar_loaded = load_lora_checkpoint(adapter_dir)
    assert sidecar_loaded.solver_override == ODESolverOverride(
        num_steps=4, cfg_scale=1.0
    )

    # Explicit override beats sidecar and embedded config.
    explicit_loaded = load_lora_checkpoint(
        adapter_dir,
        adapter_id="custom_id",
        solver_override={"num_steps": 2, "cfg_scale": 0.0},
    )
    assert explicit_loaded.source.adapter_id == "custom_id"
    assert explicit_loaded.solver_override == ODESolverOverride(
        num_steps=2, cfg_scale=0.0
    )

    # Helper returns None when no override exists anywhere.
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert discover_solver_override(empty_dir, adapter_config=None) is None


def test_load_sharded_safetensors_and_single_pt_file(tmp_path: Path) -> None:
    sharded_dir = tmp_path / "sharded_lora"
    sharded_dir.mkdir()
    save_file(
        {"blocks.0.to_q.lora_A.weight": torch.randn(2, 8)},
        str(sharded_dir / "adapter_model-00001-of-00002.safetensors"),
    )
    save_file(
        {"blocks.0.to_q.lora_B.weight": torch.randn(8, 2)},
        str(sharded_dir / "adapter_model-00002-of-00002.safetensors"),
    )
    loaded_sharded = load_lora_checkpoint(sharded_dir)
    assert loaded_sharded.source.target_modules == ("to_q",)
    assert len(loaded_sharded.source.named_tensors) == 2

    # Single .pt file with nested state_dict and inferred target_modules.
    pt_path = tmp_path / "single_lora.pt"
    torch.save(
        {
            "state_dict": {
                "layers.0.attn.to_k.lora_A.weight": torch.randn(2, 8),
                "layers.0.attn.to_k.lora_B.weight": torch.randn(8, 2),
                "non_lora_metadata": torch.tensor([1.0]),
            }
        },
        pt_path,
    )
    loaded_pt = load_lora_checkpoint(pt_path, alpha=6.0)
    assert loaded_pt.source.adapter_id == "single_lora"
    assert loaded_pt.source.alpha == 6.0
    assert loaded_pt.source.target_modules == ("to_k",)
    assert len(loaded_pt.source.named_tensors) == 2


def test_checkpoint_loader_error_cases(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lora_checkpoint(tmp_path / "missing")

    empty_dir = tmp_path / "no_weights"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no LoRA weight files"):
        load_lora_checkpoint(empty_dir)

    no_lora_file = tmp_path / "plain.safetensors"
    save_file({"weight": torch.randn(4, 4)}, str(no_lora_file))
    with pytest.raises(ValueError, match="no lora_A / lora_B factor tensors"):
        load_lora_checkpoint(no_lora_file)

    assert infer_target_modules_from_tensors(
        {
            "blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 4),
            "blocks.0.attn.to_v.lora_A.default.weight": torch.randn(2, 4),
        }
    ) == ("to_q", "to_v")
