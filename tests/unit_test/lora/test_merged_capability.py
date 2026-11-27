# SPDX-License-Identifier: Apache-2.0
"""Unit tests for merged LoRA capability specs and architecture registry audit."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sglang_omni.lora.capability import (
    MERGED_LORA_ARCHITECTURE_REGISTRY,
    MergedLoRACapabilityError,
    MergedLoRASpec,
    SupportsMergedLoRA,
    audit_merged_lora_registry,
    resolve_merged_lora_spec,
)
from sglang_omni.models.auk.dit import AuKDit
from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
)
from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker


class FakeShardedLinear(nn.Module):
    """Minimal stand-in for a TP-sharded linear module."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        output_sizes: tuple[int, ...] | None = None,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.output_size_per_partition = out_features
        if output_sizes is not None:
            self.output_sizes = list(output_sizes)


class FakePlainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def merged_lora_spec(self) -> MergedLoRASpec:
        return MergedLoRASpec(allow_plain_linear=False)


class FakePackedAttributeModel(nn.Module):
    packed_modules_mapping = {"qkv_proj": ("q_proj", "k_proj", "v_proj")}

    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = FakeShardedLinear(12, 4, output_sizes=(4, 4, 4))


class FakeBadPackedSpecModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def merged_lora_spec(self) -> MergedLoRASpec:
        return MergedLoRASpec(
            packed_modules_mapping={"nonexistent_qkv": ("q", "k", "v")},
            allow_plain_linear=True,
        )


def test_merged_lora_spec_validation() -> None:
    spec = MergedLoRASpec(
        target_module_root="model",
        packed_modules_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
        name_aliases={"to_out": "to_out.0"},
        strip_prefixes=("model.",),
        allow_plain_linear=False,
    )
    assert spec.packed_modules_mapping["qkv_proj"] == ("q_proj", "k_proj", "v_proj")

    with pytest.raises(ValueError, match="mapped to both"):
        MergedLoRASpec(
            packed_modules_mapping={"p1": ("q_proj",), "p2": ("q_proj",)},
        )

    with pytest.raises(ValueError, match="non-empty sequence"):
        MergedLoRASpec(packed_modules_mapping={"p1": "q_proj"})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="name_aliases"):
        MergedLoRASpec(name_aliases={"": "to_out.0"})

    with pytest.raises(ValueError, match="strip_prefixes"):
        MergedLoRASpec(strip_prefixes=("",))


def test_auk_dit_implements_supports_merged_lora() -> None:
    dit = AuKDit(
        dim=32,
        num_layers=1,
        num_single_layers=1,
        heads=2,
        dim_head=16,
        text_hidden_dim=16,
        latent_dim=8,
    )
    assert isinstance(dit, SupportsMergedLoRA)
    spec = resolve_merged_lora_spec(dit, architecture="AuKForConditionalGeneration")
    assert spec.allow_plain_linear is True
    assert spec.packed_modules_mapping["to_qkv"] == ("to_q", "to_k", "to_v")
    assert spec.packed_modules_mapping["to_qkv_c"] == ("to_q_c", "to_k_c", "to_v_c")
    assert spec.packed_modules_mapping["linear_in"] == ("gate_proj", "up_proj")
    assert spec.name_aliases == {"to_out": "to_out.0"}

    # Registered alias resolves to canonical architecture cleanly.
    alias_spec = resolve_merged_lora_spec(dit, architecture="AuK")
    assert alias_spec == spec

    # Single-stream-only AuKDit omits to_qkv_c automatically and still validates.
    single_only = AuKDit(
        dim=32,
        num_layers=0,
        num_single_layers=1,
        heads=2,
        dim_head=16,
        text_hidden_dim=16,
        latent_dim=8,
    )
    single_spec = resolve_merged_lora_spec(
        single_only, architecture="AuKForConditionalGeneration"
    )
    assert "to_qkv_c" not in single_spec.packed_modules_mapping


def test_qwen3_omni_thinker_and_talker_specs() -> None:
    class FakeARSubtree(nn.Module):
        def __init__(self, include_dense_mlp: bool) -> None:
            super().__init__()
            self.model = nn.Module()
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.qkv_proj = FakeShardedLinear(12, 4, output_sizes=(4, 4, 4))
            layer.self_attn.o_proj = FakeShardedLinear(4, 4)
            if include_dense_mlp:
                layer.mlp = nn.Module()
                layer.mlp.gate_up_proj = FakeShardedLinear(16, 4, output_sizes=(8, 8))
            self.model.layers = nn.ModuleList([layer])

    dense_thinker = FakeARSubtree(include_dense_mlp=True)
    dense_spec = Qwen3OmniThinkerForCausalLM.merged_lora_spec(dense_thinker)
    assert "gate_up_proj" in dense_spec.packed_modules_mapping
    assert dense_spec.allow_plain_linear is False

    moe_thinker = FakeARSubtree(include_dense_mlp=False)
    moe_spec = Qwen3OmniThinkerForCausalLM.merged_lora_spec(moe_thinker)
    assert "qkv_proj" in moe_spec.packed_modules_mapping
    assert "gate_up_proj" not in moe_spec.packed_modules_mapping

    talker_stub = FakeARSubtree(include_dense_mlp=True)
    talker_spec = Qwen3OmniTalker.merged_lora_spec(talker_stub)
    assert talker_spec.target_module_root == "model"
    assert "talker.model." in talker_spec.strip_prefixes


def test_capability_rejections_and_fallback() -> None:
    with pytest.raises(MergedLoRACapabilityError, match="plain torch.nn.Linear"):
        resolve_merged_lora_spec(FakePlainModel())

    with pytest.raises(MergedLoRACapabilityError, match="nonexistent_qkv"):
        resolve_merged_lora_spec(FakeBadPackedSpecModel())

    with pytest.raises(MergedLoRACapabilityError, match="does not implement"):
        resolve_merged_lora_spec(nn.Sequential(nn.Linear(4, 4)))

    fallback = FakePackedAttributeModel()
    spec = resolve_merged_lora_spec(fallback)
    assert spec.packed_modules_mapping == {"qkv_proj": ("q_proj", "k_proj", "v_proj")}

    with pytest.raises(MergedLoRACapabilityError, match="not_capable"):
        resolve_merged_lora_spec(
            fallback, architecture="WhisperForConditionalGeneration"
        )

    with pytest.raises(MergedLoRACapabilityError, match="runtime_only"):
        resolve_merged_lora_spec(fallback, architecture="MossTTSDelayModel")


def test_dots_tts_flow_head_spec() -> None:
    class TinyDotsFlowHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.velocity_field_predictor = nn.Module()
            self.velocity_field_predictor.q_proj = nn.Linear(8, 8, bias=False)
            self.velocity_field_predictor.k_proj = nn.Linear(8, 8, bias=False)
            self.velocity_field_predictor.v_proj = nn.Linear(8, 8, bias=False)

        def merged_lora_spec(self) -> MergedLoRASpec:
            return DotsTTSFlowHead.merged_lora_spec(self)

    flow_head = TinyDotsFlowHead()
    assert isinstance(flow_head, SupportsMergedLoRA)
    dots_spec = resolve_merged_lora_spec(
        flow_head, architecture="DotsTTSForConditionalGeneration"
    )
    assert dots_spec.target_module_root == "velocity_field_predictor"
    assert dots_spec.allow_plain_linear is True


def test_minimax_music3_dit_spec() -> None:
    try:
        from sglang_omni.models.minimax_music3.dit import MiniMaxMusic3DIT
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional minimax_music3 dependency missing: {exc}")

    class TinyMusic3DIT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.diffusion_transformer = nn.Module()
            self.diffusion_transformer.transformer = nn.Module()
            layer = nn.Module()
            layer.attn = nn.Module()
            layer.attn.to_qkv = nn.Linear(8, 24, bias=False)
            layer.attn.to_out = nn.Linear(8, 8, bias=False)
            self.diffusion_transformer.transformer.layers = nn.ModuleList([layer])

        def merged_lora_spec(self) -> MergedLoRASpec:
            return MiniMaxMusic3DIT.merged_lora_spec(self)

    music_dit = TinyMusic3DIT()
    assert isinstance(music_dit, SupportsMergedLoRA)
    music_spec = resolve_merged_lora_spec(
        music_dit, architecture="MiniMaxMusic3ForConditionalGeneration"
    )
    assert music_spec.target_module_root == "diffusion_transformer"
    assert music_spec.packed_modules_mapping == {"to_qkv": ("to_q", "to_k", "to_v")}
    assert music_spec.allow_plain_linear is True


def test_architecture_registry_audit_passes_and_reports_undeclared() -> None:
    assert audit_merged_lora_registry() == []
    assert "AuKForConditionalGeneration" in MERGED_LORA_ARCHITECTURE_REGISTRY
    assert "Qwen3OmniMoeForConditionalGeneration" in MERGED_LORA_ARCHITECTURE_REGISTRY
    assert (
        MERGED_LORA_ARCHITECTURE_REGISTRY[
            "MiniMaxMusic3ForConditionalGeneration"
        ].status
        == "supported"
    )
    assert (
        MERGED_LORA_ARCHITECTURE_REGISTRY["DotsTTSForConditionalGeneration"].status
        == "supported"
    )

    synthetic = dict(MERGED_LORA_ARCHITECTURE_REGISTRY)
    synthetic["BrandNewOmniModel"] = object()
    violations = audit_merged_lora_registry(synthetic)
    assert len(violations) == 1
    assert "BrandNewOmniModel" in violations[0]
    assert "MERGED_LORA_ARCHITECTURE_REGISTRY" in violations[0]
