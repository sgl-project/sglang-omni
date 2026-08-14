# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch


def _platform(*, rocm: bool) -> SimpleNamespace:
    return SimpleNamespace(is_rocm=lambda: rocm)


def test_fishaudio_uses_aiter_for_slow_ar_on_rocm(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder

    monkeypatch.setattr(engine_builder, "current_platform", _platform(rocm=True))
    monkeypatch.setattr(
        engine_builder,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: (_ for _ in ()).throw(AssertionError("CUDA SM queried")),
    )

    assert engine_builder._resolve_fast_ar_attention_backend(gpu_id=0) == "aiter"


def test_fishaudio_torch_fast_ar_updates_cache() -> None:
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
        audio_decoder,
    )

    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 2, 8)
    v = torch.randn(2, 1, 2, 8)
    k_cache = torch.zeros(2, 4, 2, 8)
    v_cache = torch.zeros(2, 4, 2, 8)

    output = audio_decoder._torch_kvcache_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        causal=True,
        cache_position=2,
    )

    assert output.shape == q.shape
    assert torch.equal(k_cache[:, 2:3], k)
    assert torch.equal(v_cache[:, 2:3], v)


def test_llada_selects_platform_attention_backend(monkeypatch) -> None:
    from sglang_omni.models.llada2_uni import stages

    monkeypatch.setattr(stages, "current_platform", _platform(rocm=True))
    assert stages._dllm_attention_backend() == "aiter"

    monkeypatch.setattr(stages, "current_platform", _platform(rocm=False))
    assert stages._dllm_attention_backend() == "flashinfer"


def test_ming_audio_vae_uses_sdpa_on_rocm(monkeypatch) -> None:
    from sglang_omni.models.ming_omni.talker.audio_vae import vae_modules

    model_args = {"_attn_implementation": "flash_attention_2"}
    monkeypatch.setattr(vae_modules, "current_platform", _platform(rocm=True))
    assert vae_modules._qwen2_config(model_args)._attn_implementation == "sdpa"

    monkeypatch.setattr(vae_modules, "current_platform", _platform(rocm=False))
    assert (
        vae_modules._qwen2_config(model_args)._attn_implementation
        == "flash_attention_2"
    )


def test_ming_talker_accepts_tensor_cache_length() -> None:
    from sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker import (
        _cache_positions,
    )

    positions = _cache_positions(torch.tensor(7), 3, torch.device("cpu"))

    assert torch.equal(positions, torch.tensor([7, 8, 9]))
