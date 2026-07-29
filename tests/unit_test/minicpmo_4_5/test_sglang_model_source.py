# SPDX-License-Identifier: Apache-2.0
"""Dependency-light checks for the native MiniCPM-o SGLang model."""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_MODEL_PATH = _ROOT / "sglang_omni/models/minicpmo_4_5/sglang_model.py"
_RUNNER_PATH = _ROOT / "sglang_omni/model_runner/sglang_model_runner.py"


def _routing_namespace() -> dict[str, object]:
    tree = ast.parse(_MODEL_PATH.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_WEIGHT_COMPONENT_PREFIXES"
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_route_weight_name",
            "_packed_weight_target",
            "_required_packed_shards",
            "_validate_loaded_understanding_weights",
        }:
            selected.append(node)

    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(_MODEL_PATH), "exec"),
        namespace,
    )
    return namespace


def test_minicpmo_architecture_is_registered_to_native_model() -> None:
    tree = ast.parse(_RUNNER_PATH.read_text(encoding="utf-8"))
    registrations = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "sglang_omni_models"
            for target in node.targets
        ):
            registrations = ast.literal_eval(node.value)
            break

    assert registrations is not None
    assert registrations["MiniCPMO"] == (
        "sglang_omni.models.minicpmo_4_5.sglang_model:" "MiniCPMO45ForCausalLM"
    )


def test_checkpoint_prefix_routing_excludes_tts() -> None:
    route = _routing_namespace()["_route_weight_name"]

    assert route("llm.model.layers.0.self_attn.q_proj.weight") == (
        "llm",
        "llm.model.layers.0.self_attn.q_proj.weight",
    )
    assert route("vpm.encoder.layers.0.self_attn.q_proj.weight")[0] == "vpm"
    assert route("resampler.query") == ("resampler", "resampler.query")
    assert route("apm.layers.0.fc1.weight")[0] == "apm"
    assert route("audio_projection_layer.linear1.weight")[0] == (
        "audio_projection_layer"
    )
    assert route("tts.model.layers.0.self_attn.q_proj.weight") is None
    try:
        route("unknown.weight")
    except ValueError as exc:
        assert "unsupported MiniCPM-o checkpoint tensor prefix" in str(exc)
    else:
        raise AssertionError("unknown checkpoint prefixes must fail closed")


def test_packed_weight_routing_is_component_scoped() -> None:
    packed = _routing_namespace()["_packed_weight_target"]

    assert packed("llm", "llm.model.layers.0.self_attn.k_proj.weight") == (
        "llm.model.layers.0.self_attn.qkv_proj.weight",
        "k",
    )
    assert packed("llm", "llm.model.layers.0.mlp.up_proj.weight") == (
        "llm.model.layers.0.mlp.gate_up_proj.weight",
        1,
    )
    assert packed("vpm", "vpm.encoder.layers.0.self_attn.v_proj.weight") == (
        "vpm.encoder.layers.0.self_attn.qkv_proj.weight",
        "v",
    )
    # Whisper keeps separate projections, and resampler.kv_proj must never be
    # mistaken for a split v_proj tensor.
    assert packed("apm", "apm.layers.0.self_attn.v_proj.weight") is None
    assert packed("resampler", "resampler.kv_proj.weight") is None


def test_weight_manifest_validation_rejects_missing_parameter_or_shard() -> None:
    validate = _routing_namespace()["_validate_loaded_understanding_weights"]
    parameters = {
        "llm.model.layers.0.self_attn.qkv_proj.weight",
        "llm.model.layers.0.mlp.gate_up_proj.weight",
        "apm.layers.0.fc1.weight",
    }
    complete_shards = {
        "llm.model.layers.0.self_attn.qkv_proj.weight": {"q", "k", "v"},
        "llm.model.layers.0.mlp.gate_up_proj.weight": {0, 1},
    }

    validate(parameters, set(parameters), complete_shards)

    try:
        validate(parameters, parameters - {"apm.layers.0.fc1.weight"}, complete_shards)
    except RuntimeError as exc:
        assert "missing parameters: apm.layers.0.fc1.weight" in str(exc)
    else:
        raise AssertionError("missing local parameters must fail closed")

    incomplete_shards = {**complete_shards}
    incomplete_shards["llm.model.layers.0.self_attn.qkv_proj.weight"] = {"q", "v"}
    try:
        validate(parameters, set(parameters), incomplete_shards)
    except RuntimeError as exc:
        assert "missing shards k" in str(exc)
    else:
        raise AssertionError("missing packed shards must fail closed")


def test_native_model_has_no_demo_or_rpc_runtime_dependency() -> None:
    source = _MODEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MiniCPMO45ForCausalLM"
    )

    assert any(
        isinstance(base, ast.Name) and base.id == "MiniCPMV4_5"
        for base in model_class.bases
    )
    assert {
        node.name for node in model_class.body if isinstance(node, ast.FunctionDef)
    } >= {
        "forward",
        "get_input_embeddings",
        "pad_input_ids",
        "load_weights",
    }
    for forbidden in (
        "process_runtime",
        "runtime_worker",
        "subprocess",
        "MiniCPM-o-Demo",
    ):
        assert forbidden not in source
