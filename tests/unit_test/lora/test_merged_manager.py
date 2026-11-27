# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MergedLoRAManager, target matching, AuKDit, and TP-sharded merging."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sglang_omni.lora.capability import MergedLoRASpec
from sglang_omni.lora.merged_manager import (
    LoRAAdapterTensor,
    LoRARequestPayload,
    MergedLoRAManager,
    build_target_module_matcher,
    normalize_target_modules,
)
from sglang_omni.models.auk.dit import AuKDit


class ShardedColLinear(nn.Module):
    """Minimal TP column-parallel linear (standalone or packed)."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int,
        tp_rank: int,
        output_sizes: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.output_size_per_partition = int(weight.shape[0])
        self.output_size = int(weight.shape[0]) * tp_size
        self.input_size = int(weight.shape[1])
        if output_sizes is not None:
            self.output_sizes = list(output_sizes)


class ShardedRowLinear(nn.Module):
    """Minimal TP row-parallel linear."""

    def __init__(self, weight: torch.Tensor, *, tp_size: int, tp_rank: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_size_per_partition = int(weight.shape[1])
        self.output_size = int(weight.shape[0])
        self.input_size = int(weight.shape[1]) * tp_size


class ReplicatedShardedLinear(nn.Module):
    """Minimal replicated linear with tp_size > 1."""

    def __init__(self, weight: torch.Tensor, *, tp_size: int, tp_rank: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.output_size_per_partition = int(weight.shape[0])
        self.output_size = int(weight.shape[0])
        self.input_size = int(weight.shape[1])


class FakeTPARModel(nn.Module):
    """Toy TP-sharded AR model implementing SupportsMergedLoRA."""

    def __init__(
        self,
        *,
        qkv_weight: torch.Tensor,
        qkv_sizes: tuple[int, int, int],
        gate_up_weight: torch.Tensor,
        gate_up_sizes: tuple[int, int],
        standalone_col_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        replicated_weight: torch.Tensor,
        tp_size: int,
        tp_rank: int,
    ) -> None:
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = ShardedColLinear(
            qkv_weight, tp_size=tp_size, tp_rank=tp_rank, output_sizes=qkv_sizes
        )
        layer.self_attn.o_proj = ShardedRowLinear(
            o_proj_weight, tp_size=tp_size, tp_rank=tp_rank
        )
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = ShardedColLinear(
            gate_up_weight,
            tp_size=tp_size,
            tp_rank=tp_rank,
            output_sizes=gate_up_sizes,
        )
        layer.mlp.extra_col = ShardedColLinear(
            standalone_col_weight, tp_size=tp_size, tp_rank=tp_rank
        )
        layer.mlp.head_rep = ReplicatedShardedLinear(
            replicated_weight, tp_size=tp_size, tp_rank=tp_rank
        )
        self.model.layers = nn.ModuleList([layer])

    def merged_lora_spec(self) -> MergedLoRASpec:
        return MergedLoRASpec(
            target_module_root="model",
            packed_modules_mapping={
                "qkv_proj": ("q_proj", "k_proj", "v_proj"),
                "gate_up_proj": ("gate_proj", "up_proj"),
            },
            strip_prefixes=("model.",),
            allow_plain_linear=False,
        )


def test_normalize_target_modules_types_and_validation() -> None:
    assert normalize_target_modules("to_q") == "to_q"
    assert normalize_target_modules(["to_v", "to_q", "to_q"]) == ("to_v", "to_q")
    assert normalize_target_modules({"to_k", "to_q"}) == ("to_k", "to_q")
    assert normalize_target_modules(frozenset(["a", "b"])) == ("a", "b")

    with pytest.raises(ValueError, match="non-empty"):
        normalize_target_modules("")
    with pytest.raises(ValueError, match="empty"):
        normalize_target_modules([])
    with pytest.raises(ValueError, match="target_modules"):
        normalize_target_modules(123)  # type: ignore[arg-type]


def test_target_module_matcher_aliases_prefixes_and_anchored_regex() -> None:
    matcher = build_target_module_matcher(
        ["to_q", "to_out"],
        name_aliases={"to_out": "to_out.0"},
        strip_prefixes=("dit.", "transformer."),
    )
    assert matcher.matches("dit.transformer_blocks.0.attn.to_q")
    assert matcher.matches("transformer.transformer_blocks.0.attn.to_out")
    assert matcher.matches("transformer_blocks.0.attn.to_out.0")
    assert not matcher.matches("transformer_blocks.0.attn.to_k")

    anchored = build_target_module_matcher(
        r"^transformer_blocks\.0\.attn\.to_out\.0$",
        name_aliases={"to_out": "to_out.0"},
        strip_prefixes=("dit.",),
    )
    assert anchored.matches("dit.transformer_blocks.0.attn.to_out")
    assert anchored.matches("transformer_blocks.0.attn.to_out.0")
    assert not anchored.matches("transformer_blocks.1.attn.to_out")


def test_merged_lora_manager_on_real_auk_dit() -> None:
    torch.manual_seed(42)
    dit = AuKDit(
        dim=32,
        num_layers=1,
        num_single_layers=1,
        heads=2,
        dim_head=16,
        text_hidden_dim=16,
        latent_dim=8,
    )
    qkv_param = dit.transformer_blocks[0].attn.to_qkv.weight
    out_param = dit.transformer_blocks[0].attn.to_out[0].weight
    ff_param = dit.single_transformer_blocks[0].ff.linear_in.weight
    base_qkv = qkv_param.detach().clone()
    base_out = out_param.detach().clone()
    base_ff = ff_param.detach().clone()
    qkv_ptr = qkv_param.data_ptr()
    out_ptr = out_param.data_ptr()
    ff_ptr = ff_param.data_ptr()

    manager = MergedLoRAManager(
        dit,
        architecture="AuKForConditionalGeneration",
        prefer_pinned=False,
    )

    rank = 4
    a_q = torch.randn(rank, 32)
    b_q = torch.randn(32, rank)
    a_v = torch.randn(rank, 32)
    b_v = torch.randn(32, rank)
    a_out = torch.randn(rank, 32)
    b_out = torch.randn(32, rank)
    inner_dim = ff_param.shape[0] // 2
    a_up = torch.randn(rank, 32)
    b_up = torch.randn(inner_dim, rank)

    request = LoRARequestPayload(
        adapter_id="adapter_v1",
        target_modules=["to_q", "to_v", "to_out", "up_proj"],
        tensors=(
            LoRAAdapterTensor(
                module_name="dit.transformer_blocks.0.attn.to_q",
                lora_a=a_q,
                lora_b=b_q,
                scaling=0.5,
            ),
            LoRAAdapterTensor(
                module_name="transformer.transformer_blocks.0.attn.to_v",
                lora_a=a_v,
                lora_b=b_v,
                scaling=0.25,
            ),
            LoRAAdapterTensor(
                module_name="dit.transformer_blocks.0.attn.to_out",
                lora_a=a_out,
                lora_b=b_out,
                scaling=2.0,
            ),
            LoRAAdapterTensor(
                module_name="single_transformer_blocks.0.ff.up_proj",
                lora_a=a_up,
                lora_b=b_up,
                scaling=1.5,
            ),
        ),
        scale=1.0,
    )

    park_loc, matched = manager.prepare_adapter(request)
    assert park_loc == "cpu"
    assert matched == 4
    updated = manager.commit_adapter("adapter_v1")
    assert updated == 3
    assert manager.active_adapter_id == "adapter_v1"
    assert qkv_param.data_ptr() == qkv_ptr
    assert out_param.data_ptr() == out_ptr
    assert ff_param.data_ptr() == ff_ptr

    # Verify exact packed slice updates on Q (0:32), unchanged K (32:64), V (64:96).
    assert torch.allclose(qkv_param[:32], base_qkv[:32] + 0.5 * (b_q @ a_q), atol=1e-5)
    assert torch.equal(qkv_param[32:64], base_qkv[32:64])
    assert torch.allclose(
        qkv_param[64:96], base_qkv[64:96] + 0.25 * (b_v @ a_v), atol=1e-5
    )
    assert torch.allclose(out_param, base_out + 2.0 * (b_out @ a_out), atol=1e-5)
    assert torch.equal(ff_param[:inner_dim], base_ff[:inner_dim])
    assert torch.allclose(
        ff_param[inner_dim:], base_ff[inner_dim:] + 1.5 * (b_up @ a_up), atol=1e-5
    )

    # Switching to adapter_v2 targeting only to_k restores omitted to_out and linear_in.
    a_k = torch.randn(rank, 32)
    b_k = torch.randn(32, rank)
    request_v2 = LoRARequestPayload(
        adapter_id="adapter_v2",
        target_modules=["to_k"],
        tensors=(
            LoRAAdapterTensor(
                module_name="transformer_blocks.0.attn.to_k",
                lora_a=a_k,
                lora_b=b_k,
                scaling=1.0,
            ),
        ),
    )
    manager.prepare_adapter(request_v2, restore_omitted=True)
    manager.commit_adapter("adapter_v2")
    assert torch.equal(qkv_param[:32], base_qkv[:32])
    assert torch.allclose(qkv_param[32:64], base_qkv[32:64] + (b_k @ a_k), atol=1e-5)
    assert torch.equal(qkv_param[64:96], base_qkv[64:96])
    assert torch.equal(out_param, base_out)
    assert torch.equal(ff_param, base_ff)

    # Restore base resets everything back bit-for-bit.
    assert manager.restore_base() == 1
    assert manager.active_adapter_id is None
    assert torch.equal(qkv_param, base_qkv)
    assert torch.equal(out_param, base_out)
    assert torch.equal(ff_param, base_ff)


def test_merged_lora_manager_tp2_sharded_ar_numerics() -> None:
    torch.manual_seed(7)
    tp_size = 2
    in_dim = 8
    q_full, k_full, v_full = 8, 4, 4  # GQA asymmetric sizes
    gate_full, up_full = 6, 6
    extra_full = 8
    rep_out = 6
    rank = 2

    q_base = torch.randn(q_full, in_dim)
    k_base = torch.randn(k_full, in_dim)
    v_base = torch.randn(v_full, in_dim)
    gate_base = torch.randn(gate_full, in_dim)
    up_base = torch.randn(up_full, in_dim)
    extra_base = torch.randn(extra_full, in_dim)
    o_base = torch.randn(in_dim, q_full)
    rep_base = torch.randn(rep_out, in_dim)

    def make_rank_model(rank_idx: int) -> FakeTPARModel:
        q_slice = q_base.chunk(tp_size, dim=0)[rank_idx]
        k_slice = k_base.chunk(tp_size, dim=0)[rank_idx]
        v_slice = v_base.chunk(tp_size, dim=0)[rank_idx]
        gate_slice = gate_base.chunk(tp_size, dim=0)[rank_idx]
        up_slice = up_base.chunk(tp_size, dim=0)[rank_idx]
        extra_slice = extra_base.chunk(tp_size, dim=0)[rank_idx]
        o_slice = o_base.chunk(tp_size, dim=1)[rank_idx]
        return FakeTPARModel(
            qkv_weight=torch.cat([q_slice, k_slice, v_slice], dim=0),
            qkv_sizes=(q_full, k_full, v_full),
            gate_up_weight=torch.cat([gate_slice, up_slice], dim=0),
            gate_up_sizes=(gate_full, up_full),
            standalone_col_weight=extra_slice,
            o_proj_weight=o_slice,
            replicated_weight=rep_base,
            tp_size=tp_size,
            tp_rank=rank_idx,
        )

    models = [make_rank_model(0), make_rank_model(1)]
    managers = [MergedLoRAManager(m, prefer_pinned=False) for m in models]

    a_q, b_q = torch.randn(rank, in_dim), torch.randn(q_full, rank)
    a_k, b_k = torch.randn(rank, in_dim), torch.randn(k_full, rank)
    a_gate, b_gate = torch.randn(rank, in_dim), torch.randn(gate_full, rank)
    a_extra, b_extra = torch.randn(rank, in_dim), torch.randn(extra_full, rank)
    a_o, b_o = torch.randn(rank, q_full), torch.randn(in_dim, rank)
    a_rep, b_rep = torch.randn(rank, in_dim), torch.randn(rep_out, rank)

    payload = LoRARequestPayload(
        adapter_id="tp_adapter",
        target_modules=[
            "q_proj",
            "k_proj",
            "gate_proj",
            "extra_col",
            "o_proj",
            "head_rep",
        ],
        tensors=(
            LoRAAdapterTensor("model.layers.0.self_attn.q_proj", a_q, b_q, 0.5),
            LoRAAdapterTensor("model.layers.0.self_attn.k_proj", a_k, b_k, 1.0),
            LoRAAdapterTensor("model.layers.0.mlp.gate_proj", a_gate, b_gate, 0.25),
            LoRAAdapterTensor("model.layers.0.mlp.extra_col", a_extra, b_extra, 1.5),
            LoRAAdapterTensor("model.layers.0.self_attn.o_proj", a_o, b_o, 0.75),
            LoRAAdapterTensor("model.layers.0.mlp.head_rep", a_rep, b_rep, 2.0),
        ),
    )

    for manager in managers:
        _, matched = manager.prepare_adapter(payload)
        assert matched == 6
        manager.commit_adapter("tp_adapter")

    # Reconstruct full tensors from rank 0 + rank 1 and compare to full unsharded math.
    q_shards = [m.model.layers[0].self_attn.qkv_proj.weight[:4] for m in models]
    k_shards = [m.model.layers[0].self_attn.qkv_proj.weight[4:6] for m in models]
    v_shards = [m.model.layers[0].self_attn.qkv_proj.weight[6:8] for m in models]
    assert torch.allclose(
        torch.cat(q_shards, dim=0), q_base + 0.5 * (b_q @ a_q), atol=1e-5
    )
    assert torch.allclose(
        torch.cat(k_shards, dim=0), k_base + 1.0 * (b_k @ a_k), atol=1e-5
    )
    assert torch.allclose(torch.cat(v_shards, dim=0), v_base, atol=1e-6)

    gate_shards = [m.model.layers[0].mlp.gate_up_proj.weight[:3] for m in models]
    up_shards = [m.model.layers[0].mlp.gate_up_proj.weight[3:6] for m in models]
    assert torch.allclose(
        torch.cat(gate_shards, dim=0), gate_base + 0.25 * (b_gate @ a_gate), atol=1e-5
    )
    assert torch.allclose(torch.cat(up_shards, dim=0), up_base, atol=1e-6)

    extra_shards = [m.model.layers[0].mlp.extra_col.weight for m in models]
    assert torch.allclose(
        torch.cat(extra_shards, dim=0),
        extra_base + 1.5 * (b_extra @ a_extra),
        atol=1e-5,
    )

    o_shards = [m.model.layers[0].self_attn.o_proj.weight for m in models]
    assert torch.allclose(
        torch.cat(o_shards, dim=1), o_base + 0.75 * (b_o @ a_o), atol=1e-5
    )

    for m in models:
        assert torch.allclose(
            m.model.layers[0].mlp.head_rep.weight,
            rep_base + 2.0 * (b_rep @ a_rep),
            atol=1e-5,
        )
