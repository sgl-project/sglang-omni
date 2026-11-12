# SPDX-License-Identifier: Apache-2.0
"""Perception and language-model checkpoint names, without loading tensors."""

from __future__ import annotations

import ast
from pathlib import Path

import torch
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from torch.nn import functional as F

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
)
from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_language_model_weights,
)
from sglang_omni.models.nemotron_transcribe.sglang_model import (
    NemotronTranscribeForConditionalGeneration,
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


def test_mixed_precision_layers_resolve_under_the_transcribe_wrapper() -> None:
    """The NVFP4 release keys layers as backbone.*; SGLang builds language_model.model.*."""
    nvfp4 = {"quant_algo": "NVFP4", "group_size": 16}
    model_class = NemotronTranscribeForConditionalGeneration
    config = ModelOptMixedPrecisionConfig.from_config(
        {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "exclude_modules": ["backbone.embeddings", "lm_head", "ctc_timestamp*"],
                "quantized_layers": {
                    "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                    "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"},
                    "backbone.layers.1.mixer.experts": nvfp4,
                    "backbone.layers.1.mixer.shared_experts.up_proj": nvfp4,
                    "backbone.layers.1.mixer.shared_experts.down_proj": nvfp4,
                    "backbone.layers.5.mixer.q_proj": nvfp4,
                    "backbone.layers.5.mixer.k_proj": nvfp4,
                    "backbone.layers.5.mixer.v_proj": nvfp4,
                    "backbone.layers.5.mixer.o_proj": nvfp4,
                    "perception.encoder.asr_encoder.layers.0.attn.w_qkv": {
                        "quant_algo": "FP8"
                    },
                },
            },
            "packed_modules_mapping": model_class.packed_modules_mapping,
        }
    )
    config.apply_weight_name_mapper(model_class.hf_to_sglang_mapper)

    layers = "language_model.model.layers"
    expected = {
        f"{layers}.0.mixer.in_proj": "FP8",
        f"{layers}.0.mixer.out_proj": "FP8",
        f"{layers}.1.mixer.experts": "NVFP4",
        f"{layers}.1.mixer.shared_experts.up_proj": "NVFP4",
        f"{layers}.1.mixer.shared_experts.down_proj": "NVFP4",
        f"{layers}.5.mixer.qkv_proj": "NVFP4",
        f"{layers}.5.mixer.o_proj": "NVFP4",
        f"{layers}.1.mixer.gate": None,
        "language_model.lm_head": None,
    }
    for prefix, quant_algo in expected.items():
        assert config.resolve_quant_algo(prefix) == quant_algo, prefix


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
