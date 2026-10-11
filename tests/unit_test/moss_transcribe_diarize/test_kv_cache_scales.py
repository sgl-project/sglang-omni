# SPDX-License-Identifier: Apache-2.0
"""load_kv_cache_scales delegation on the MTD entry class.

sglang's model_runner calls ``load_kv_cache_scales`` on the EntryClass when
``kv_cache_dtype=fp8_e4m3`` and ``quantization_param_path`` are set; without
the method the server hard-fails at load. The upstream JSON loader also
assigns Python floats to ``k_scale``/``v_scale``, while the FA3 decode path
calls ``.expand()`` on them, so the delegation must convert to 0-d tensors.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
    MossTranscribeDiarizeForConditionalGeneration,
)


def _fake_language_model(n_layers: int) -> SimpleNamespace:
    calls: list[str] = []
    layers = []
    for _ in range(n_layers):
        attn = SimpleNamespace(
            k_scale=None, v_scale=None, k_scale_float=None, v_scale_float=None
        )
        layers.append(SimpleNamespace(self_attn=SimpleNamespace(attn=attn)))

    def load_kv_cache_scales(path: str) -> None:
        calls.append(path)
        # Mirror the upstream JSON loader: assigns plain Python floats.
        for i, layer in enumerate(layers):
            layer.self_attn.attn.k_scale = 0.01 * (i + 1)
            layer.self_attn.attn.v_scale = 0.01 * (i + 1)

    lm = SimpleNamespace(
        load_kv_cache_scales=load_kv_cache_scales,
        model=SimpleNamespace(layers=layers),
        parameters=lambda: iter([torch.zeros(1)]),
    )
    return SimpleNamespace(lm=lm, calls=calls)


def test_delegates_and_converts_float_scales_to_tensors():
    fake = _fake_language_model(3)
    inst = MossTranscribeDiarizeForConditionalGeneration.__new__(
        MossTranscribeDiarizeForConditionalGeneration
    )
    inst.language_model = fake.lm

    inst.load_kv_cache_scales("/tmp/kv_scales.json")

    assert fake.calls == ["/tmp/kv_scales.json"]
    for i, layer in enumerate(fake.lm.model.layers):
        attn = layer.self_attn.attn
        expected = 0.01 * (i + 1)
        # FA3 calls .expand() on these; floats would crash at decode time.
        assert isinstance(attn.k_scale, torch.Tensor) and attn.k_scale.ndim == 0
        assert isinstance(attn.v_scale, torch.Tensor) and attn.v_scale.ndim == 0
        assert attn.k_scale.dtype == torch.float32
        assert attn.v_scale.dtype == torch.float32
        assert attn.k_scale.device.type == "cpu"
        assert attn.v_scale.device.type == "cpu"
        assert torch.isclose(attn.k_scale, torch.tensor(expected))
        assert torch.isclose(attn.v_scale, torch.tensor(expected))
        assert attn.k_scale_float == expected
        assert attn.v_scale_float == expected


def test_tensor_scales_pass_through_unconverted():
    fake = _fake_language_model(1)
    attn = fake.lm.model.layers[0].self_attn.attn
    assigned: dict[str, torch.Tensor] = {}

    def load_with_tensors(path: str) -> None:
        fake.calls.append(path)
        assigned["k"] = attn.k_scale = torch.tensor(0.5)
        assigned["v"] = attn.v_scale = torch.tensor(0.5)

    fake.lm.load_kv_cache_scales = load_with_tensors
    inst = MossTranscribeDiarizeForConditionalGeneration.__new__(
        MossTranscribeDiarizeForConditionalGeneration
    )
    inst.language_model = fake.lm

    inst.load_kv_cache_scales("/tmp/kv_scales.json")

    assert attn.k_scale is assigned["k"] and attn.v_scale is assigned["v"]
    assert torch.isclose(attn.k_scale, torch.tensor(0.5))
