# SPDX-License-Identifier: Apache-2.0
"""Perception and language-model checkpoint names, without loading tensors."""

from __future__ import annotations

import ast
from pathlib import Path

import torch
from torch.nn import functional as F

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
)
from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_language_model_weights,
)

PACKAGE = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni"
    / "models"
    / "nemotron_transcribe"
)


def test_package_does_not_import_nemo() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "nemo" or name.startswith("nemo."):
                    offenders.append(f"{path.name}:{name}")
                else:
                    pass
    assert offenders == []


def test_language_model_names_cover_the_decoder_remap() -> None:
    samples = {
        "llm.model.embed_tokens.weight": ("rename", "backbone.embed_tokens.weight"),
        "llm.lm_head.weight": ("rename", "lm_head.weight"),
        "llm.model.norm.weight": ("rename", "backbone.norm_f.weight"),
        "llm.model.layers.0.mixer._fp32_params.A_log": (
            "rename",
            "backbone.layers.0.mixer.A_log",
        ),
        "llm.model.layers.3.mixer.experts.down_projs": (
            "split_down",
            "backbone.layers.3.mixer",
        ),
        "llm.model.layers.3.mixer.experts.gate_and_up_projs": (
            "split_up",
            "backbone.layers.3.mixer",
        ),
        "llm.mtp.layers.0.mixer.q_proj.weight": ("skip", ""),
        "ctc_timestamp.decoder.decoder_layers.0.weight": ("skip", ""),
        "llm.lm_head._extra_state": ("skip", ""),
        "perception.proj.weight": ("perception", "proj.weight"),
    }
    for name, expected in samples.items():
        assert classify_checkpoint_name(name) == expected


def test_mtp_draft_names_keep_only_the_mtp_layer() -> None:
    """The draft loads from the target checkpoint, so every backbone tensor is dropped and llm.mtp. becomes mtp."""
    samples = {
        "llm.mtp.layers.0.eh_proj.weight": ("rename", "mtp.layers.0.eh_proj.weight"),
        "llm.mtp.layers.0.mixer.q_proj.weight": (
            "rename",
            "mtp.layers.0.mixer.q_proj.weight",
        ),
        "llm.mtp.layers.1.mixer.experts.gate_and_up_projs": (
            "split_up",
            "mtp.layers.1.mixer",
        ),
        "llm.mtp.layers.1.mixer.experts.down_projs": (
            "split_down",
            "mtp.layers.1.mixer",
        ),
        "llm.mtp.layers.1.final_layernorm.weight": (
            "rename",
            "mtp.layers.1.final_layernorm.weight",
        ),
        "llm.mtp.layers.0.eh_proj._extra_state": ("skip", ""),
        "llm.model.embeddings.weight": ("skip", ""),
        "llm.lm_head.weight": ("skip", ""),
        "llm.model.layers.3.mixer.experts.down_projs": ("skip", ""),
        "perception.proj.weight": ("skip", ""),
    }
    for name, expected in samples.items():
        assert classify_checkpoint_name(name, is_mtp_draft=True) == expected
    converted = dict(
        checkpoint_to_language_model_weights(
            [
                (
                    "llm.mtp.layers.1.mixer.experts.gate_and_up_projs",
                    torch.ones(2, 3, 4),
                ),
                (
                    "llm.model.layers.1.mixer.experts.gate_and_up_projs",
                    torch.ones(2, 3, 4),
                ),
            ],
            is_mtp_draft=True,
        )
    )
    assert sorted(converted) == [
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "mtp.layers.1.mixer.experts.1.up_proj.weight",
    ]
    assert converted["mtp.layers.1.mixer.experts.0.up_proj.weight"].shape == (4, 3)


def test_expert_weight_conversion_preserves_ungated_relu2_computation() -> None:
    up_weights = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 16
    down_weights = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 16
    converted = dict(
        checkpoint_to_language_model_weights(
            [
                ("llm.model.layers.3.mixer.experts.gate_and_up_projs", up_weights),
                ("llm.model.layers.3.mixer.experts.down_projs", down_weights),
            ]
        )
    )
    hidden_states = torch.tensor([[1.0, -2.0, 3.0]])
    for expert_index in range(2):
        prefix = f"backbone.layers.3.mixer.experts.{expert_index}"
        reference = (
            F.relu(hidden_states @ up_weights[expert_index]).square()
            @ down_weights[expert_index]
        )
        actual = F.linear(
            F.relu(
                F.linear(hidden_states, converted[f"{prefix}.up_proj.weight"])
            ).square(),
            converted[f"{prefix}.down_proj.weight"],
        )
        torch.testing.assert_close(actual, reference)
