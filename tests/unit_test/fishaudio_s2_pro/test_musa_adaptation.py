# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from sglang_omni.models.fishaudio_s2_pro import engine_builder
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
    audio_decoder,
)


def test_fast_ar_musa_uses_fa3(monkeypatch) -> None:
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
        audio_decoder.cuda_kvcache_attention(
            q=q, **sentinels, causal=True, num_splits=1, cache_position=-1
        )
        is q
    )
    assert calls[0]["q"] is q
    assert all(calls[0][name] is value for name, value in sentinels.items())
    assert calls[0]["causal"] is True
    assert calls[0]["num_splits"] == 1
