# SPDX-License-Identifier: Apache-2.0
"""Perception and language-model checkpoint names, without loading tensors."""

from __future__ import annotations

import ast
import json
import os
import struct
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
    perception_checkpoint_keys,
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


def local_checkpoint_directory() -> Path | None:
    checkpoint_directory = os.environ.get("NEMOTRON_TRANSCRIBE_MODEL")
    return Path(checkpoint_directory) if checkpoint_directory else None


def safetensors_names(path: Path) -> set[str]:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return {name for name in header if name != "__metadata__"}


def test_local_checkpoint_matches_the_native_contract() -> None:
    model_dir = local_checkpoint_directory()
    if model_dir is None:
        pytest.skip("local Nemotron Transcribe checkpoint is not present")
    else:
        pass
    config = json.loads((model_dir / "config.json").read_text())
    encoder = config["pe_encoder_config"]
    asr = encoder["asr_encoder_cfg"]
    diar = encoder["diarization_model_cfg"]
    names = safetensors_names(model_dir / "model.safetensors")
    expected = perception_checkpoint_keys(
        int(asr["n_layers"]),
        int(diar["encoder"]["n_layers"]),
        asr_qk_norm=bool(asr["qk_norm"]),
        high_resolution=bool(diar["high_resolution"]),
        learnable_silence=bool(diar["sortformer_modules"]["use_learnable_sil_emb"]),
    )
    perception = {name for name in names if name.startswith("perception.")}
    assert perception == expected

    kinds = {classify_checkpoint_name(name)[0] for name in names}
    assert kinds <= {"skip", "perception", "rename", "split_down", "split_up"}
    renamed = [name for name in names if classify_checkpoint_name(name)[0] == "rename"]
    assert renamed
    assert all(
        "llm.model." not in classify_checkpoint_name(name)[1] for name in renamed
    )
    assert all(
        "_fp32_params" not in classify_checkpoint_name(name)[1] for name in renamed
    )


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
