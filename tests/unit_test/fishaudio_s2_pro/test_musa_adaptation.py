# SPDX-License-Identifier: Apache-2.0
"""Focused tests for FishAudio S2-Pro MUSA compatibility."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.models.fishaudio_s2_pro import engine_builder
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
    audio_decoder,
)


def test_fast_ar_musa_uses_fa3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_builder.current_platform, "is_musa", lambda: True)
    assert engine_builder.resolve_fast_ar_attention_backend(gpu_id=0) == "fa3"

    calls = []
    flash_attn = ModuleType("flash_attn_interface")

    def flash_attn_with_kvcache(**kwargs):
        calls.append(kwargs)
        return kwargs["q"]

    flash_attn.flash_attn_with_kvcache = flash_attn_with_kvcache
    monkeypatch.setitem(sys.modules, "flash_attn_interface", flash_attn)

    q = SimpleNamespace(device=SimpleNamespace(type="musa", index=0))
    sentinels = {
        name: object() for name in ("k_cache", "v_cache", "k", "v", "cache_seqlens")
    }

    assert (
        audio_decoder.musa_kvcache_attention(
            q=q, **sentinels, causal=True, num_splits=1, cache_position=-1
        )
        is q
    )
    assert calls[0]["q"] is q
    assert all(calls[0][name] is value for name, value in sentinels.items())
    assert calls[0]["causal"] is True
    assert calls[0]["num_splits"] == 1


def test_fast_ar_musa_dispatches_to_mate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def musa_kvcache_attention(**kwargs):
        calls.append(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(audio_decoder, "musa_kvcache_attention", musa_kvcache_attention)

    q = SimpleNamespace(device=SimpleNamespace(type="musa", index=0))
    k_cache = object()
    v_cache = object()
    k = object()
    v = object()
    cache_seqlens = SimpleNamespace()
    cache_seqlens.contiguous = lambda: cache_seqlens
    init_fn = audio_decoder.flash_attn_kvcache_op._init_fn  # noqa: leading-underscore

    assert (
        init_fn(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
            num_splits=2,
            cache_position=-1,
        )
        is q
    )
    assert calls[0] == {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k": k,
        "v": v,
        "cache_seqlens": cache_seqlens,
        "causal": True,
        "num_splits": 2,
        "cache_position": -1,
    }
