# SPDX-License-Identifier: Apache-2.0
"""Perception and language-model checkpoint names, without loading tensors."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP
from torch.nn import functional as F

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
    classify_mtp_checkpoint_name,
)
from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_language_model_weights,
    checkpoint_to_mtp_weights,
)
from sglang_omni.models.nemotron_transcribe.mtp_model import (
    NemotronTranscribeMTP,
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


def test_mtp_names_are_routed_separately_from_target_weights() -> None:
    assert classify_checkpoint_name("llm.mtp.layers.0.mixer.q_proj.weight") == (
        "skip",
        "",
    )
    assert classify_mtp_checkpoint_name("llm.mtp.layers.0.mixer.q_proj.weight") == (
        "rename",
        "mtp.layers.0.mixer.q_proj.weight",
    )
    assert classify_mtp_checkpoint_name("llm.mtp.layers.1._extra_state") == (
        "skip",
        "",
    )


def test_mtp_mapping_preserves_qkv_names_expert_transforms_and_bias_dtype() -> None:
    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
    weights = [
        (
            "llm.mtp.layers.0.mixer.q_proj.weight",
            torch.ones((2, 2), dtype=torch.bfloat16),
        ),
        (
            "llm.mtp.layers.0.mixer.k_proj.weight",
            torch.ones((2, 2), dtype=torch.bfloat16),
        ),
        (
            "llm.mtp.layers.0.mixer.v_proj.weight",
            torch.ones((2, 2), dtype=torch.bfloat16),
        ),
        (
            "llm.mtp.layers.1.mixer.experts.gate_and_up_projs",
            torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        ),
        (
            "llm.mtp.layers.1.mixer.experts.down_projs",
            torch.arange(24, dtype=torch.bfloat16).reshape(2, 4, 3),
        ),
        ("llm.mtp.layers.1.mixer.gate.e_score_correction_bias", bias),
        (
            "llm.mtp.layers.1._extra_state",
            torch.ones(1, dtype=torch.float32),
        ),
    ]
    mapped = dict(checkpoint_to_mtp_weights(weights))

    assert set(mapped) == {
        "mtp.layers.0.mixer.q_proj.weight",
        "mtp.layers.0.mixer.k_proj.weight",
        "mtp.layers.0.mixer.v_proj.weight",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "mtp.layers.1.mixer.experts.1.up_proj.weight",
        "mtp.layers.1.mixer.experts.0.down_proj.weight",
        "mtp.layers.1.mixer.experts.1.down_proj.weight",
        "mtp.layers.1.mixer.gate.e_score_correction_bias",
    }
    assert mapped["mtp.layers.1.mixer.experts.0.up_proj.weight"].shape == (4, 3)
    assert mapped["mtp.layers.1.mixer.experts.0.down_proj.weight"].shape == (3, 4)
    mapped_bias = mapped["mtp.layers.1.mixer.gate.e_score_correction_bias"]
    assert mapped_bias.dtype == bias.dtype
    assert torch.equal(
        mapped_bias,
        bias,
    )


def test_mtp_expert_conversion_preserves_ungated_relu2_computation() -> None:
    up_weights = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 16
    down_weights = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 16
    converted = dict(
        checkpoint_to_mtp_weights(
            [
                (
                    "llm.mtp.layers.1.mixer.experts.gate_and_up_projs",
                    up_weights,
                ),
                ("llm.mtp.layers.1.mixer.experts.down_projs", down_weights),
            ]
        )
    )
    hidden_states = torch.tensor([[1.0, -2.0, 3.0]])
    for expert_index in range(2):
        prefix = f"mtp.layers.1.mixer.experts.{expert_index}"
        reference = (
            F.relu(hidden_states @ up_weights[expert_index]).square()
            @ down_weights[expert_index]
        )
        actual = F.linear(
            F.relu(
                F.linear(
                    hidden_states,
                    converted[f"{prefix}.up_proj.weight"],
                )
            ).square(),
            converted[f"{prefix}.down_proj.weight"],
        )
        torch.testing.assert_close(actual, reference)


def test_mtp_loader_forwards_tensors_and_propagates_parent_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = NemotronTranscribeMTP.__new__(NemotronTranscribeMTP)
    adapter.config = SimpleNamespace(max_n_routed_experts=1)

    def named_parameters(
        model: NemotronTranscribeMTP,
    ) -> list[tuple[str, torch.Tensor]]:
        del model
        return [("model.layers.0.mixer.norm.weight", torch.empty(1))]

    monkeypatch.setattr(NemotronTranscribeMTP, "named_parameters", named_parameters)

    checkpoint_tensor = torch.ones(2, dtype=torch.float16)
    checkpoint_weights = [("llm.mtp.layers.0.mixer.norm.weight", checkpoint_tensor)]
    forwarded: list[tuple[str, torch.Tensor]] = []

    def capture_parent_weights(
        model: NemotronHForCausalLMMTP,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        del model
        forwarded.extend(weights)

    monkeypatch.setattr(NemotronHForCausalLMMTP, "load_weights", capture_parent_weights)
    adapter.load_weights(checkpoint_weights)

    assert len(forwarded) == 1
    assert forwarded[0][0] == "mtp.layers.0.mixer.norm.weight"
    assert forwarded[0][1] is checkpoint_tensor
    assert forwarded[0][1].dtype == torch.float16

    parent_error = RuntimeError("parent loader failed")

    def raise_parent_error(
        model: NemotronHForCausalLMMTP,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        del model, weights
        raise parent_error

    monkeypatch.setattr(NemotronHForCausalLMMTP, "load_weights", raise_parent_error)
    with pytest.raises(RuntimeError) as error:
        adapter.load_weights(checkpoint_weights)

    assert error.value is parent_error


def test_mtp_loader_writes_qkv_expert_and_ordinary_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = NemotronTranscribeMTP.__new__(NemotronTranscribeMTP)
    torch.nn.Module.__init__(adapter)
    adapter.config = SimpleNamespace(max_n_routed_experts=2)
    adapter.model = SimpleNamespace(start_layer=0, end_layer=2)
    adapter.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    qkv_parameter = torch.nn.Parameter(torch.zeros((6, 2)), requires_grad=False)
    norm_parameter = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    w13_parameter = torch.nn.Parameter(torch.zeros((2, 2, 3)), requires_grad=False)
    w2_parameter = torch.nn.Parameter(torch.zeros((2, 3, 2)), requires_grad=False)
    router_bias_parameter = torch.nn.Parameter(
        torch.zeros(2, dtype=torch.float32), requires_grad=False
    )

    def qkv_weight_loader(
        parameter: torch.Tensor, loaded_weight: torch.Tensor, shard_id: str
    ) -> None:
        shard_index = {"q": 0, "k": 1, "v": 2}[shard_id]
        shard_start = shard_index * loaded_weight.shape[0]
        parameter.data[shard_start : shard_start + loaded_weight.shape[0]].copy_(
            loaded_weight
        )

    def expert_weight_loader(
        parameter: torch.Tensor,
        loaded_weight: torch.Tensor,
        parameter_name: str,
        *,
        shard_id: str,
        expert_id: int,
    ) -> None:
        if "w13_weight" in parameter_name:
            assert shard_id == "w1"
        elif "w2_weight" in parameter_name:
            assert shard_id == "w2"
        else:
            raise AssertionError(f"unexpected expert parameter {parameter_name}")
        parameter.data[expert_id].copy_(loaded_weight)

    qkv_parameter.weight_loader = qkv_weight_loader
    w13_parameter.weight_loader = expert_weight_loader
    w2_parameter.weight_loader = expert_weight_loader

    parameters = [
        ("model.layers.0.mixer.qkv_proj.weight", qkv_parameter),
        ("model.layers.0.mixer.norm.weight", norm_parameter),
        ("model.layers.1.mixer.experts.w13_weight", w13_parameter),
        ("model.layers.1.mixer.experts.w2_weight", w2_parameter),
        (
            "model.layers.1.mixer.gate.e_score_correction_bias",
            router_bias_parameter,
        ),
    ]

    def named_parameters(
        model: NemotronTranscribeMTP,
    ) -> list[tuple[str, torch.Tensor]]:
        del model
        return parameters

    monkeypatch.setattr(NemotronTranscribeMTP, "named_parameters", named_parameters)

    query_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    key_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    value_weight = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    up_weights = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    down_weights = torch.arange(12, 24, dtype=torch.float32).reshape(2, 2, 3)
    norm_weight = torch.tensor([13.0, 14.0])
    router_bias = torch.tensor([0.25, -0.5], dtype=torch.float32)

    adapter.load_weights(
        [
            ("llm.mtp.layers.0.mixer.q_proj.weight", query_weight),
            ("llm.mtp.layers.0.mixer.k_proj.weight", key_weight),
            ("llm.mtp.layers.0.mixer.v_proj.weight", value_weight),
            ("llm.mtp.layers.0.mixer.norm.weight", norm_weight),
            ("llm.mtp.layers.1.mixer.experts.gate_and_up_projs", up_weights),
            ("llm.mtp.layers.1.mixer.experts.down_projs", down_weights),
            (
                "llm.mtp.layers.1.mixer.gate.e_score_correction_bias",
                router_bias,
            ),
        ]
    )

    torch.testing.assert_close(
        qkv_parameter, torch.cat((query_weight, key_weight, value_weight), dim=0)
    )
    torch.testing.assert_close(norm_parameter, norm_weight)
    torch.testing.assert_close(
        w13_parameter, torch.stack([up_weights[index].t() for index in range(2)])
    )
    torch.testing.assert_close(
        w2_parameter, torch.stack([down_weights[index].t() for index in range(2)])
    )
    torch.testing.assert_close(router_bias_parameter, router_bias)


def test_mtp_inventory_expands_qkv_and_moe_parameters_and_fails_closed() -> None:
    expected = expected_mtp_checkpoint_names(
        {
            "model.layers.0.mixer.qkv_proj.weight",
            "model.layers.1.mixer.experts.w13_weight",
            "model.layers.1.mixer.experts.w2_weight",
            "model.layers.1.mixer.gate.e_score_correction_bias",
            "lm_head.weight",
        },
        num_experts=2,
    )
    assert "mtp.layers.0.mixer.q_proj.weight" in expected
    assert "mtp.layers.0.mixer.k_proj.weight" in expected
    assert "mtp.layers.0.mixer.v_proj.weight" in expected
    assert "mtp.layers.1.mixer.experts.1.up_proj.weight" in expected
    assert "mtp.layers.1.mixer.experts.1.down_proj.weight" in expected
    assert "mtp.layers.1.mixer.gate.e_score_correction_bias" in expected
    validate_mtp_checkpoint_inventory(expected, expected, set(), set())
    with pytest.raises(ValueError, match="layout mismatch"):
        validate_mtp_checkpoint_inventory(
            expected,
            expected - {"mtp.layers.0.mixer.q_proj.weight"},
            {"mtp.layers.9.unexpected.weight"},
            set(),
        )
