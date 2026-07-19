# SPDX-License-Identifier: Apache-2.0
"""SemanticConditioner.project() unit tests.

These tests construct a real :class:`SemanticConditioner` and inject CPU
fakes onto its private attributes, bypassing ``load()`` entirely so no Ming
model files, GPU, or network are required. The fake connector is an identity
module (returns its ``inputs_embeds`` as the last hidden state) so the
projection chain is exercised end-to-end on CPU.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from sglang_omni.models.ming_omni.diffusion.semantic_conditioner import (  # noqa: E402
    SemanticConditioner,
)


class _IdentityConnector:
    """Stands in for the Qwen2 connector; returns inputs_embeds unchanged."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, inputs_embeds, attention_mask, output_hidden_states):
        self.calls.append(
            {
                "n": inputs_embeds.shape[1],
                "attention_mask_shape": tuple(attention_mask.shape),
                "output_hidden_states": output_hidden_states,
            }
        )
        return SimpleNamespace(hidden_states=[inputs_embeds])


def _make_conditioner(*, scales, scale_indices, proj_mid=1536):
    """Build a SemanticConditioner with CPU/float32 fakes (no load())."""
    cond = SemanticConditioner()
    cond._device = "cpu"
    cond._dtype = torch.float32
    cond._proj_in = nn.Linear(4096, proj_mid)
    cond._proj_out = nn.Linear(proj_mid, 2560)
    connector = _IdentityConnector()
    cond._connector = connector
    cond._img_gen_scales = scales
    cond._scale_indices = scale_indices
    return cond, connector


class _FakeSelfAttention:
    def __init__(self) -> None:
        self.is_causal = True


class _FakeLayer:
    def __init__(self) -> None:
        self.self_attn = _FakeSelfAttention()


class _FakeConnector:
    instances: list["_FakeConnector"] = []

    def __init__(self) -> None:
        self.model = SimpleNamespace(layers=[_FakeLayer(), _FakeLayer()])
        self.to_calls = []
        self.eval_called = False
        _FakeConnector.instances.append(self)

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return iter(())


class _FakeAutoModelForCausalLM:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_path, *, subfolder, torch_dtype):
        cls.calls.append(
            {
                "model_path": model_path,
                "subfolder": subfolder,
                "torch_dtype": torch_dtype,
            }
        )
        return _FakeConnector()


def _install_fake_transformers(monkeypatch):
    module = ModuleType("transformers")
    module.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", module)
    _FakeAutoModelForCausalLM.calls.clear()
    _FakeConnector.instances.clear()


def _write_tiny_conditioner_model(tmp_path, *, include_32x32: bool = True):
    from safetensors.torch import save_file

    model_root = tmp_path / "ming"
    mlp_dir = model_root / "mlp"
    mlp_dir.mkdir(parents=True)
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "llm_config": {
                    "image_patch_token": 101,
                    "image_start_token": 102,
                }
            }
        )
    )
    (mlp_dir / "config.json").write_text(json.dumps({"img_gen_scales": [1, 2]}))

    state = {
        "proj_in.weight": torch.ones(2, 3),
        "proj_in.bias": torch.zeros(2),
        "proj_out.weight": torch.ones(4, 2),
        "proj_out.bias": torch.zeros(4),
        "query_tokens_dict.1x1": torch.ones(1, 3),
    }
    if include_32x32:
        state["query_tokens_dict.2x2"] = torch.full((4, 3), 2.0)
    save_file(state, str(mlp_dir / "model.safetensors"))
    return model_root


def test_load_reads_config_connector_projection_and_query_tokens(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("safetensors.torch")
    _install_fake_transformers(monkeypatch)
    model_root = _write_tiny_conditioner_model(tmp_path)
    cond = SemanticConditioner()

    cond.load(str(model_root), torch.device("cpu"), dtype=torch.float32)

    assert cond._proj_in is not None
    assert cond._proj_out is not None
    assert cond._proj_in.weight.shape == (2, 3)
    assert cond._proj_out.weight.shape == (4, 2)
    assert cond._proj_in.weight.dtype == torch.float32
    assert cond._proj_out.weight.dtype == torch.float32
    assert cond._proj_in.weight.device == torch.device("cpu")
    assert cond._proj_out.weight.device == torch.device("cpu")
    torch.testing.assert_close(cond._proj_in.weight, torch.ones(2, 3))
    torch.testing.assert_close(cond._proj_in.bias, torch.zeros(2))
    torch.testing.assert_close(cond._proj_out.weight, torch.ones(4, 2))
    torch.testing.assert_close(cond._proj_out.bias, torch.zeros(4))
    assert _FakeAutoModelForCausalLM.calls == [
        {
            "model_path": str(model_root),
            "subfolder": "connector",
            "torch_dtype": torch.float32,
        }
    ]
    connector = _FakeConnector.instances[0]
    assert all(layer.self_attn.is_causal is False for layer in connector.model.layers)
    assert connector.to_calls == [torch.device("cpu")]
    assert connector.eval_called is True
    assert cond.image_patch_token == 101
    assert cond.image_start_token == 102
    assert cond.image_end_token == 103
    assert cond.img_gen_scales == [1, 2]
    assert cond._scale_indices == [1, 5]
    assert cond.query_tokens.shape == (5, 3)
    torch.testing.assert_close(cond.query_tokens[0], torch.ones(3))
    torch.testing.assert_close(cond.query_tokens[-1], torch.full((3,), 2.0))


def test_load_fails_when_configured_query_token_key_is_missing(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("safetensors.torch")
    _install_fake_transformers(monkeypatch)
    model_root = _write_tiny_conditioner_model(tmp_path, include_32x32=False)
    cond = SemanticConditioner()

    with pytest.raises(KeyError, match="query_tokens_dict.2x2"):
        cond.load(str(model_root), torch.device("cpu"), dtype=torch.float32)


def test_project_output_shape_single_scale() -> None:
    # scales=[16] -> scale_indices=[256]; last scale slices h[:, 0:256].
    cond, _ = _make_conditioner(scales=[16], scale_indices=[256])
    hidden = torch.randn(2, 256, 4096)

    out = cond.project(hidden)

    assert out.shape == (2, 256, 2560)


def test_project_output_is_l2_normalized() -> None:
    cond, _ = _make_conditioner(scales=[16], scale_indices=[256])
    hidden = torch.randn(2, 256, 4096)

    out = cond.project(hidden)

    norms = torch.linalg.norm(out, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)


def test_project_multi_scale_slices_last_scale() -> None:
    # scales=[16, 32] -> scale_indices=[256, 1280]; last scale is rows
    # [256:1280], i.e. 1024 query tokens.
    cond, connector = _make_conditioner(scales=[16, 32], scale_indices=[256, 1280])
    hidden = torch.randn(3, 1280, 4096)

    out = cond.project(hidden)

    # Output row count equals the LAST scale's length (1024), not 256 or 1280.
    assert out.shape == (3, 1024, 2560)
    # The connector received exactly the last scale's slice.
    assert connector.calls[0]["n"] == 1024
    sliced = hidden[:, 256:1280, :]
    expected = cond._proj_in(sliced)
    expected = cond._proj_out(expected)
    expected = torch.nn.functional.normalize(expected, dim=-1)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_project_non_3d_input_raises_value_error() -> None:
    cond, _ = _make_conditioner(scales=[16], scale_indices=[256])
    hidden = torch.randn(256, 4096)  # 2D, missing batch dim

    with pytest.raises(ValueError):
        cond.project(hidden)


def test_project_not_loaded_raises_runtime_error() -> None:
    cond = SemanticConditioner()  # _connector is None
    assert cond._connector is None
    hidden = torch.randn(1, 256, 4096)

    with pytest.raises(RuntimeError):
        cond.project(hidden)
