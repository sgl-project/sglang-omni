# SPDX-License-Identifier: Apache-2.0
"""Checkpoint tensors land on the right Llama and Moshi parameters, without a GPU."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from torch import nn
from transformers import LlamaConfig

from sglang_omni.models.personaplex.architecture import (
    AUDIO_CARD,
    NUM_AUDIO_STREAMS,
    NUM_STREAMS,
    TEMPORAL_TRANSFORMER,
    TEXT_CARD,
)
from sglang_omni.models.personaplex.sglang_model import (
    PersonaPlexForCausalLM,
    backbone_weight,
)

DIM = 4


def kv_scale_model(
    scale_state: Literal["missing", "parameter", "tensor"]
) -> SimpleNamespace:
    layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = nn.Linear(DIM, DIM, bias=False, dtype=torch.bfloat16)
        layer.self_attn.attn = nn.Module()
        for scale_name in ("k_scale", "v_scale"):
            if scale_state == "missing":
                scale = None
            elif scale_state == "parameter":
                scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            else:
                scale = torch.tensor(1.0)
            setattr(layer.self_attn.attn, scale_name, scale)
            setattr(layer.self_attn.attn, f"{scale_name}_float", 1.0)
        layers.append(layer)
    return SimpleNamespace(
        llm=SimpleNamespace(
            model=SimpleNamespace(
                config=LlamaConfig(num_hidden_layers=len(layers)),
                layers=layers,
                _kv_cache_parallel_layout=(0, 1),
            )
        ),
        fusion_buffer=torch.empty(1, DIM),
    )


def write_kv_scales(
    path: Path,
    scales: dict[int, dict[int, float]],
    *,
    model_type: str = "llama",
    dtype: str = "float8_e4m3fn",
) -> str:
    path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "kv_cache": {"dtype": dtype, "scaling_factor": scales},
            }
        )
    )
    return str(path)


@pytest.mark.parametrize(
    "scale_state",
    ["missing", "parameter", "tensor"],
    ids=["bf16-weights", "fp8-weights", "existing-tensor"],
)
def test_kv_scales_support_attention_tensor_and_host_scalar_consumers(
    tmp_path: Path, scale_state: Literal["missing", "parameter", "tensor"]
) -> None:
    """Non-unit scales must reach both attention APIs without replacing loaded Parameters."""
    model = kv_scale_model(scale_state)
    scales = {0: {0: 0.1, 1: 0.2}}
    original_scales = [
        [
            (scale, scale.data_ptr() if scale is not None else None)
            for scale in (layer.self_attn.attn.k_scale, layer.self_attn.attn.v_scale)
        ]
        for layer in model.llm.model.layers
    ]
    PersonaPlexForCausalLM.load_kv_cache_scales(
        model, write_kv_scales(tmp_path / "scales.json", scales)
    )
    for layer_index, layer in enumerate(model.llm.model.layers):
        expected = torch.tensor(scales[0][layer_index], dtype=torch.float32)
        for scale_index, scale_name in enumerate(("k_scale", "v_scale")):
            scale = getattr(layer.self_attn.attn, scale_name)
            assert scale.dtype == torch.float32
            assert scale.device == model.fusion_buffer.device
            assert torch.equal(scale.expand(2, 3), expected.expand(2, 3))
            assert getattr(layer.self_attn.attn, f"{scale_name}_float") == scale.item()
            original, original_address = original_scales[layer_index][scale_index]
            if original is not None:
                assert scale is original
                assert scale.data_ptr() == original_address
            else:
                assert scale.ndim == 0


@pytest.mark.parametrize("scale_state", ["missing", "parameter"])
@pytest.mark.parametrize(
    "invalid_scale",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), 1e40, 1e-50],
    ids=[
        "zero",
        "negative",
        "nan",
        "inf",
        "negative-inf",
        "fp32-overflow",
        "fp32-underflow",
    ],
)
def test_invalid_last_kv_scale_preserves_every_layer(
    tmp_path: Path,
    scale_state: Literal["missing", "parameter"],
    invalid_scale: float,
) -> None:
    """A bad final scale must not leave earlier layers partly calibrated."""
    model = kv_scale_model(scale_state)
    original_scales = [
        (layer.self_attn.attn.k_scale, layer.self_attn.attn.v_scale)
        for layer in model.llm.model.layers
    ]
    path = write_kv_scales(tmp_path / "scales.json", {0: {0: 0.1, 1: invalid_scale}})
    with pytest.raises(ValueError, match="finite and positive"):
        PersonaPlexForCausalLM.load_kv_cache_scales(model, path)
    for layer_index, layer in enumerate(model.llm.model.layers):
        for scale_index, scale_name in enumerate(("k_scale", "v_scale")):
            scale = getattr(layer.self_attn.attn, scale_name)
            assert scale is original_scales[layer_index][scale_index]
            assert getattr(layer.self_attn.attn, f"{scale_name}_float") == 1.0
            if scale is not None:
                assert scale.item() == 1.0
            else:
                pass


@pytest.mark.parametrize(
    "invalid_case,error_type,message",
    [
        ("missing-file", FileNotFoundError, "scales.json"),
        ("invalid-json", ValueError, "Invalid JSON"),
        ("model", ValueError, "Model type is llama"),
        ("dtype", ValueError, "rather than float8_e4m3fn"),
        ("rank", ValueError, "TP rank 0 not found"),
        ("layer", ValueError, "scales for layer 1"),
        ("extra-rank", ValueError, "Loaded dictionary has TP size 2"),
        ("extra-layer", ValueError, "Expected 2 layers, got 3"),
    ],
)
def test_kv_scale_file_errors_do_not_fall_back_to_unit_scales(
    tmp_path: Path, invalid_case: str, error_type: type[Exception], message: str
) -> None:
    """An invalid calibration must fail for the expected reason before any update."""
    model = kv_scale_model("missing")
    scales = {0: {0: 0.1, 1: 0.2}}
    path = tmp_path / "scales.json"
    if invalid_case == "rank":
        scales[1] = scales.pop(0)
    elif invalid_case == "layer":
        scales[0][2] = scales[0].pop(1)
    elif invalid_case == "extra-rank":
        scales[1] = {0: 0.5, 1: 0.6}
    elif invalid_case == "extra-layer":
        scales[0][2] = 0.5
    else:
        pass
    write_kv_scales(
        path,
        scales,
        model_type="personaplex" if invalid_case == "model" else "llama",
        dtype="bfloat16" if invalid_case == "dtype" else "float8_e4m3fn",
    )
    if invalid_case == "missing-file":
        path.unlink()
    elif invalid_case == "invalid-json":
        path.write_text("{")
    else:
        pass
    with pytest.raises(error_type, match=message):
        PersonaPlexForCausalLM.load_kv_cache_scales(model, str(path))
    assert all(
        layer.self_attn.attn.k_scale is None and layer.self_attn.attn.v_scale is None
        for layer in model.llm.model.layers
    )


@pytest.mark.parametrize("invalid_target", ["float", "vector", "dtype", "device"])
def test_invalid_last_kv_scale_target_preserves_existing_parameters(
    tmp_path: Path, invalid_target: str
) -> None:
    """Validate every destination before copying into any loaded scale Parameter."""
    model = kv_scale_model("parameter")
    last_attention = model.llm.model.layers[-1].self_attn.attn
    delattr(last_attention, "v_scale")
    replacements = {
        "float": 1.0,
        "vector": torch.ones(1),
        "dtype": torch.tensor(1.0, dtype=torch.bfloat16),
        "device": torch.tensor(1.0, device="meta"),
    }
    last_attention.v_scale = replacements[invalid_target]
    first_attention = model.llm.model.layers[0].self_attn.attn
    first_key_scale = first_attention.k_scale
    with pytest.raises(ValueError, match="scalar float32 tensor"):
        PersonaPlexForCausalLM.load_kv_cache_scales(
            model,
            write_kv_scales(tmp_path / "scales.json", {0: {0: 0.1, 1: 0.2}}),
        )
    assert first_attention.k_scale is first_key_scale
    for layer in model.llm.model.layers:
        assert layer.self_attn.attn.k_scale.item() == 1.0
        assert layer.self_attn.attn.k_scale_float == 1.0
        assert layer.self_attn.attn.v_scale_float == 1.0


@pytest.mark.parametrize("scale_state", ["missing", "parameter", "tensor"])
def test_kv_scales_do_not_use_offloaded_projection_device(
    tmp_path: Path, scale_state: str
) -> None:
    model = kv_scale_model(scale_state)
    for layer in model.llm.model.layers:
        layer.self_attn.qkv_proj.to("meta")
    PersonaPlexForCausalLM.load_kv_cache_scales(
        model, write_kv_scales(tmp_path / "scales.json", {0: {0: 0.3, 1: 0.4}})
    )
    for layer in model.llm.model.layers:
        assert layer.self_attn.attn.k_scale.device == model.fusion_buffer.device
        assert layer.self_attn.attn.k_scale.item() == layer.self_attn.attn.k_scale_float


def test_kv_scales_preserve_offloaded_cpu_parameters(tmp_path: Path) -> None:
    model = kv_scale_model("parameter")
    model.fusion_buffer = torch.empty(1, DIM, device="meta")
    original = model.llm.model.layers[0].self_attn.attn.k_scale
    address = original.data_ptr()
    PersonaPlexForCausalLM.load_kv_cache_scales(
        model, write_kv_scales(tmp_path / "scales.json", {0: {0: 0.3, 1: 0.4}})
    )
    assert model.llm.model.layers[0].self_attn.attn.k_scale is original
    assert original.data_ptr() == address
    assert original.item() == torch.tensor(0.3, dtype=torch.float32).item()


def test_missing_kv_scales_use_execution_device(tmp_path: Path) -> None:
    model = kv_scale_model("missing")
    model.fusion_buffer = torch.empty(1, DIM, device="meta")
    PersonaPlexForCausalLM.load_kv_cache_scales(
        model, write_kv_scales(tmp_path / "scales.json", {0: {0: 0.3, 1: 0.4}})
    )
    for layer in model.llm.model.layers:
        assert layer.self_attn.attn.k_scale.device == model.fusion_buffer.device


def test_plain_cpu_scale_cannot_follow_offloader(tmp_path: Path) -> None:
    model = kv_scale_model("tensor")
    model.fusion_buffer = torch.empty(1, DIM, device="meta")
    with pytest.raises(ValueError, match="offloaded CPU Parameter"):
        PersonaPlexForCausalLM.load_kv_cache_scales(
            model, write_kv_scales(tmp_path / "scales.json", {0: {0: 0.3, 1: 0.4}})
        )
    assert model.llm.model.layers[0].self_attn.attn.k_scale.item() == 1.0


def test_custom_kv_scales_reject_fnuz_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sglang_omni.models.personaplex.sglang_model.is_fp8_fnuz", lambda: True
    )
    model = kv_scale_model("parameter")
    with pytest.raises(NotImplementedError, match="do not support FP8 FNUZ"):
        PersonaPlexForCausalLM.load_kv_cache_scales(
            model, str(tmp_path / "missing.json")
        )
    assert model.llm.model.layers[0].self_attn.attn.k_scale.item() == 1.0


def test_backbone_tensors_map_onto_llama_names():
    in_proj = torch.arange(12.0).view(6, 2)
    mapped = dict(
        backbone_weight("transformer.layers.3.self_attn.in_proj_weight", in_proj)
    )
    q, k, v = in_proj.chunk(3)
    assert torch.equal(mapped["model.layers.3.self_attn.q_proj.weight"], q)
    assert torch.equal(mapped["model.layers.3.self_attn.k_proj.weight"], k)
    assert torch.equal(mapped["model.layers.3.self_attn.v_proj.weight"], v)

    linear_in = torch.arange(8.0).view(4, 2)
    mapped = dict(
        backbone_weight("transformer.layers.0.gating.linear_in.weight", linear_in)
    )
    assert torch.equal(mapped["model.layers.0.mlp.gate_proj.weight"], linear_in[:2])
    assert torch.equal(mapped["model.layers.0.mlp.up_proj.weight"], linear_in[2:])

    alpha = torch.ones(1, 1, DIM)
    ((name, tensor),) = backbone_weight("transformer.layers.1.norm2.alpha", alpha)
    assert name == "model.layers.1.post_attention_layernorm.weight"
    assert tensor.shape == (DIM,)
    ((name, _),) = backbone_weight(
        "transformer.layers.1.self_attn.out_proj.weight", torch.zeros(DIM, DIM)
    )
    assert name == "model.layers.1.self_attn.o_proj.weight"
    with pytest.raises(KeyError):
        list(backbone_weight("transformer.layers.1.unknown.weight", alpha))


def fake_model() -> SimpleNamespace:
    loaded = SimpleNamespace(backbone=None, depformer=None)
    return SimpleNamespace(
        loaded=loaded,
        temporal=replace(TEMPORAL_TRANSFORMER, num_layers=1),
        llm=SimpleNamespace(load_weights=lambda w: setattr(loaded, "backbone", w)),
        audio_emb=nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, DIM) for _ in range(NUM_AUDIO_STREAMS)
        ),
        text_emb=nn.Embedding(TEXT_CARD + 1, DIM),
        depformer=SimpleNamespace(
            load_reference_weights=lambda w: setattr(loaded, "depformer", w)
        ),
    )


def test_load_weights_routes_every_checkpoint_group():
    model = fake_model()
    text_emb = torch.randn(TEXT_CARD + 1, DIM)
    audio = {f"emb.{k}.weight": torch.randn(AUDIO_CARD + 1, DIM) for k in range(16)}
    weights = {
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(DIM, DIM),
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * DIM, DIM),
        "transformer.layers.0.norm1.alpha": torch.ones(1, 1, DIM),
        "transformer.layers.0.norm2.alpha": torch.ones(1, 1, DIM),
        "transformer.layers.0.gating.linear_in.weight": torch.zeros(2 * DIM, DIM),
        "transformer.layers.0.gating.linear_out.weight": torch.zeros(DIM, DIM),
        "out_norm.alpha": torch.ones(1, 1, DIM),
        "text_linear.weight": torch.randn(TEXT_CARD, DIM),
        "text_emb.weight": text_emb,
        "depformer_in.0.weight": torch.zeros(1),
        "linears.0.weight": torch.zeros(1),
        **audio,
    }

    for missing in (
        "transformer.layers.0.self_attn.in_proj_weight",
        "transformer.layers.0.gating.linear_out.weight",
        "out_norm.alpha",
        "text_linear.weight",
        "text_emb.weight",
    ):
        incomplete = {
            name: tensor for name, tensor in weights.items() if name != missing
        }
        rejected = fake_model()
        with pytest.raises(ValueError, match="missing PersonaPlex backbone weights"):
            PersonaPlexForCausalLM.load_weights(rejected, incomplete.items())
        assert rejected.loaded.backbone is None

    PersonaPlexForCausalLM.load_weights(model, weights.items())

    backbone = dict(model.loaded.backbone)
    assert set(backbone) == {
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    }
    assert torch.equal(backbone["model.embed_tokens.weight"], text_emb[:TEXT_CARD])
    assert torch.equal(model.text_emb.weight, text_emb)
    assert torch.equal(model.audio_emb[5].weight, audio["emb.5.weight"])
    assert set(model.loaded.depformer) == {"depformer_in.0.weight", "linears.0.weight"}

    with pytest.raises(KeyError, match="unexpected PersonaPlex tensor"):
        PersonaPlexForCausalLM.load_weights(fake_model(), [("mystery", torch.zeros(1))])


def test_embed_rows_reads_text_from_column_zero_and_codebook_k_from_column_k_plus_one():
    model = SimpleNamespace(
        audio_emb=nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, NUM_STREAMS) for _ in range(NUM_AUDIO_STREAMS)
        ),
        text_emb=nn.Embedding(TEXT_CARD + 1, NUM_STREAMS),
    )
    with torch.no_grad():
        for k, table in enumerate(model.audio_emb):
            table.weight.zero_()
            table.weight[:, 1 + k] = torch.arange(AUDIO_CARD + 1, dtype=torch.float32)
        model.text_emb.weight.zero_()
        model.text_emb.weight[:, 0] = torch.arange(TEXT_CARD + 1, dtype=torch.float32)

    rows = torch.randint(0, AUDIO_CARD + 1, (3, NUM_STREAMS))
    rows[:, 0] = torch.tensor([5, 31999, 32000])
    with torch.no_grad():
        embedded = PersonaPlexForCausalLM.embed_rows(model, rows)
    assert torch.equal(embedded, rows.float())


@pytest.mark.parametrize("query_count", [1, 3], ids=["decode", "prefill"])
@pytest.mark.parametrize("boundary_offset", [-2, -1, 0, 1, 3001])
def test_attention_matches_streaming_ring_at_wraparound(
    query_count: int, boundary_offset: int
) -> None:
    capacity = TEMPORAL_TRANSFORMER.context
    last_position = capacity + boundary_offset
    positions = torch.arange(last_position + 1)
    queries = positions[-query_count:]
    model = SimpleNamespace(temporal=TEMPORAL_TRANSFORMER)
    window = PersonaPlexForCausalLM.get_attention_sliding_window_size(model)
    mask = (queries[:, None] >= positions) & (positions >= queries[:, None] - window)
    keys = torch.zeros(1, 1, len(positions), 4)
    values = positions.float().view(1, 1, -1, 1) / capacity
    query = torch.zeros(1, 1, query_count, 4)
    actual = torch.nn.functional.scaled_dot_product_attention(
        query, keys, values, attn_mask=mask
    )
    slots = torch.arange(capacity)
    for index, position in enumerate(queries.tolist()):
        end_offset = position + 1
        # note (LinzeShi): The ring labels its next overwrite slot as a future key.
        delta = slots - end_offset % capacity
        ring_positions = torch.where(
            delta <= 0, end_offset + delta, end_offset + delta - capacity
        )
        visible = (slots < end_offset) & (ring_positions <= position)
        stored_positions = position - (position - slots) % capacity
        reference_positions = stored_positions[visible].sort().values
        assert torch.equal(positions[mask[index]], reference_positions)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[:, :, index : index + 1],
            keys[:, :, reference_positions],
            values[:, :, reference_positions],
        )
        torch.testing.assert_close(actual[:, :, index : index + 1], reference)
