# SPDX-License-Identifier: Apache-2.0
"""Perception and language-model checkpoint names, without loading tensors."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
    classify_mtp_checkpoint_name,
)
from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_language_model_weights,
    checkpoint_to_mtp_weights,
)
from sglang_omni.models.nemotron_transcribe.mtp_draft import (
    NemotronTranscribeMTPDraft,
    expected_mtp_checkpoint_names,
    validate_mtp_checkpoint_inventory,
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
    """The draft loads from the target checkpoint, so every backbone tensor is dropped and each MTP prefix becomes mtp."""
    samples = {
        "llm.mtp.layers.0.eh_proj.weight": ("rename", "mtp.layers.0.eh_proj.weight"),
        "language_model.mtp.layers.0.enorm.weight": (
            "rename",
            "mtp.layers.0.enorm.weight",
        ),
        "mtp.layers.0.hnorm.weight": ("rename", "mtp.layers.0.hnorm.weight"),
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
        assert classify_mtp_checkpoint_name(name) == expected
    for name in (
        "language_model.mtp.layers.0.enorm.weight",
        "mtp.layers.0.hnorm.weight",
    ):
        assert classify_checkpoint_name(name) == ("skip", "")
    converted = dict(
        checkpoint_to_mtp_weights(
            [
                (
                    "llm.mtp.layers.1.mixer.experts.gate_and_up_projs",
                    torch.ones(2, 3, 4),
                ),
                (
                    "llm.model.layers.1.mixer.experts.gate_and_up_projs",
                    torch.ones(2, 3, 4),
                ),
            ]
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


# Every llm.mtp. tensor of nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B c17667641642, shrunk to hidden 2, 2 experts.
MTP_CHECKPOINT_SHAPES = {
    "llm.mtp.layers.0.eh_proj._extra_state": (0,),
    "llm.mtp.layers.0.eh_proj.weight": (2, 4),
    "llm.mtp.layers.0.enorm.weight": (2,),
    "llm.mtp.layers.0.hnorm.weight": (2,),
    "llm.mtp.layers.0.mixer.k_proj.weight": (2, 2),
    "llm.mtp.layers.0.mixer.o_proj.weight": (2, 2),
    "llm.mtp.layers.0.mixer.q_proj.weight": (2, 2),
    "llm.mtp.layers.0.mixer.v_proj.weight": (2, 2),
    "llm.mtp.layers.0.norm.weight": (2,),
    "llm.mtp.layers.1.final_layernorm.weight": (2,),
    "llm.mtp.layers.1.mixer.experts.down_projs": (2, 3, 2),
    "llm.mtp.layers.1.mixer.experts.gate_and_up_projs": (2, 2, 3),
    "llm.mtp.layers.1.mixer.gate.e_score_correction_bias": (2,),
    "llm.mtp.layers.1.mixer.gate.weight": (2, 2),
    "llm.mtp.layers.1.mixer.shared_experts.down_proj.weight": (2, 3),
    "llm.mtp.layers.1.mixer.shared_experts.up_proj.weight": (3, 2),
    "llm.mtp.layers.1.norm.weight": (2,),
}
# The parameters SGLang's NemotronH MTP builds for the pattern *E at the same sizes.
MTP_PARAMETER_SHAPES = {
    "model.embed_tokens.weight": (4, 2),
    "model.layers.0.eh_proj.weight": (2, 4),
    "model.layers.0.enorm.weight": (2,),
    "model.layers.0.hnorm.weight": (2,),
    "model.layers.0.mixer.qkv_proj.weight": (6, 2),
    "model.layers.0.mixer.o_proj.weight": (2, 2),
    "model.layers.0.norm.weight": (2,),
    "model.layers.1.final_layernorm.weight": (2,),
    "model.layers.1.mixer.experts.w13_weight": (2, 3, 2),
    "model.layers.1.mixer.experts.w2_weight": (2, 2, 3),
    "model.layers.1.mixer.gate.e_score_correction_bias": (2,),
    "model.layers.1.mixer.gate.weight": (2, 2),
    "model.layers.1.mixer.shared_experts.down_proj.weight": (2, 3),
    "model.layers.1.mixer.shared_experts.up_proj.weight": (3, 2),
    "model.layers.1.norm.weight": (2,),
    "lm_head.weight": (4, 2),
}


def load_qkv_shard(
    parameter: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
) -> None:
    offset = "qkv".index(shard_id) * loaded_weight.shape[0]
    parameter.data[offset : offset + loaded_weight.shape[0]].copy_(loaded_weight)


def load_expert(
    parameter: nn.Parameter,
    loaded_weight: torch.Tensor,
    parameter_name: str,
    *,
    shard_id: str,
    expert_id: int,
) -> None:
    parameter.data[expert_id].copy_(loaded_weight)


def make_mtp_draft() -> NemotronTranscribeMTPDraft:
    draft = NemotronTranscribeMTPDraft.__new__(NemotronTranscribeMTPDraft)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(
        max_n_routed_experts=2, mtp_hybrid_override_pattern="*E"
    )
    draft.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    modules: dict[str, nn.Module] = {"": draft}
    for parameter_name, shape in MTP_PARAMETER_SHAPES.items():
        module_path, leaf_name = parameter_name.rsplit(".", 1)
        parent_path = ""
        for module_name in module_path.split("."):
            path = f"{parent_path}.{module_name}".removeprefix(".")
            if path not in modules:
                modules[path] = nn.Module()
                modules[parent_path].add_module(module_name, modules[path])
            else:
                pass
            parent_path = path
        parameter = nn.Parameter(torch.full(shape, torch.nan), requires_grad=False)
        if module_path.endswith(".qkv_proj"):
            parameter.weight_loader = load_qkv_shard
        elif module_path.endswith(".experts"):
            parameter.weight_loader = load_expert
        else:
            pass
        modules[module_path].register_parameter(leaf_name, parameter)
    draft.model.start_layer = 0
    draft.model.end_layer = 2
    return draft


def mtp_checkpoint_weights(
    shapes: dict[str, tuple[int, ...]],
) -> list[tuple[str, torch.Tensor]]:
    target_weights = [
        ("llm.model.embeddings.weight", torch.ones(4, 2)),
        ("llm.lm_head.weight", torch.ones(4, 2)),
        ("perception.proj.weight", torch.ones(2, 2)),
    ]
    return target_weights + [
        (
            name,
            torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape),
        )
        for name, shape in shapes.items()
    ]


def test_mtp_draft_fills_every_layer_parameter_from_the_checkpoint_inventory() -> None:
    draft = make_mtp_draft()
    draft.load_weights(mtp_checkpoint_weights(MTP_CHECKPOINT_SHAPES))
    unfilled = [
        name
        for name, parameter in draft.named_parameters()
        if name.startswith("model.layers.") and parameter.isnan().any()
    ]
    assert unfilled == []
    torch.testing.assert_close(
        draft.get_parameter("model.layers.0.mixer.qkv_proj.weight"),
        torch.cat([torch.arange(4.0).reshape(2, 2)] * 3),
    )
    up_weights = torch.arange(12.0).reshape(2, 2, 3)
    torch.testing.assert_close(
        draft.get_parameter("model.layers.1.mixer.experts.w13_weight"),
        torch.stack([up_weights[index].t() for index in range(2)]),
    )


@pytest.mark.parametrize(
    ("shapes", "message"),
    [
        (
            {
                name: shape
                for name, shape in MTP_CHECKPOINT_SHAPES.items()
                if name != "llm.mtp.layers.0.mixer.k_proj.weight"
            },
            r"missing 1 tensors \['mtp.layers.0.mixer.k_proj.weight'\]",
        ),
        (
            {
                **MTP_CHECKPOINT_SHAPES,
                "llm.mtp.layers.1.mixer.experts.gate_and_up_projs": (1, 2, 3),
            },
            r"missing 1 tensors \['mtp.layers.1.mixer.experts.1.up_proj.weight'\]",
        ),
        (
            {**MTP_CHECKPOINT_SHAPES, "llm.mtp.layers.2.mixer.q_proj.weight": (2, 2)},
            r"unexpected 1 tensors \['mtp.layers.2.mixer.q_proj.weight'\]",
        ),
        (
            {**MTP_CHECKPOINT_SHAPES, "mtp.layers.0.norm.weight": (2,)},
            r"duplicate 1 tensors \['mtp.layers.0.norm.weight'\]",
        ),
    ],
    ids=["missing", "fewer-experts", "unexpected", "duplicate"],
)
def test_mtp_draft_rejects_an_inventory_that_does_not_fill_it_exactly_once(
    shapes: dict[str, tuple[int, ...]], message: str
) -> None:
    """A missing, short, unknown or repeated MTP tensor used to leave a draft parameter unloaded or overwritten with only a warning."""
    with pytest.raises(ValueError, match=message):
        make_mtp_draft().load_weights(mtp_checkpoint_weights(shapes))


def test_mtp_inventory_expands_qkv_and_every_expert() -> None:
    expected = expected_mtp_checkpoint_names(
        [
            "model.layers.0.mixer.qkv_proj.weight",
            "model.layers.1.mixer.experts.w13_weight",
            "model.layers.1.mixer.experts.w2_weight",
            "model.layers.1.mixer.gate.e_score_correction_bias",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ],
        num_experts=2,
    )
    assert expected == {
        "mtp.layers.0.mixer.q_proj.weight",
        "mtp.layers.0.mixer.k_proj.weight",
        "mtp.layers.0.mixer.v_proj.weight",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "mtp.layers.1.mixer.experts.1.up_proj.weight",
        "mtp.layers.1.mixer.experts.0.down_proj.weight",
        "mtp.layers.1.mixer.experts.1.down_proj.weight",
        "mtp.layers.1.mixer.gate.e_score_correction_bias",
    }
    validate_mtp_checkpoint_inventory(
        expected, expected, set(), set(), layout="*E, 2 experts"
    )


@pytest.mark.parametrize(
    ("seen", "unexpected", "duplicates", "message"),
    [
        (
            {"mtp.layers.0.norm.weight"},
            set(),
            set(),
            r"missing 1 tensors \['mtp.layers.0.enorm.weight'\]",
        ),
        (
            {"mtp.layers.0.norm.weight", "mtp.layers.0.enorm.weight"},
            {"mtp.layers.9.unexpected.weight"},
            set(),
            r"unexpected 1 tensors \['mtp.layers.9.unexpected.weight'\]",
        ),
        (
            {"mtp.layers.0.norm.weight", "mtp.layers.0.enorm.weight"},
            set(),
            {"mtp.layers.0.norm.weight"},
            r"duplicate 1 tensors \['mtp.layers.0.norm.weight'\]",
        ),
    ],
    ids=["missing", "unexpected", "duplicate"],
)
def test_mtp_inventory_mismatch_names_the_tensors_and_the_fix(
    seen: set[str], unexpected: set[str], duplicates: set[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message) as rejected:
        validate_mtp_checkpoint_inventory(
            {"mtp.layers.0.norm.weight", "mtp.layers.0.enorm.weight"},
            seen,
            unexpected,
            duplicates,
            layout="*E, 2 experts",
        )
    assert "(*E, 2 experts)" in str(rejected.value)
    assert "turn off speculative decoding" in str(rejected.value)
