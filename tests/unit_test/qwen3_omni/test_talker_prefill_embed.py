# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_omni.components import talker_prefill
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder


def _builder(cache: dict[int, torch.Tensor]) -> TalkerPrefillBuilder:
    builder = object.__new__(TalkerPrefillBuilder)
    builder._thinker_embed_cache = cache
    builder._model_path = "unused"
    builder._device = torch.device("cpu")
    builder._dtype = torch.float32
    return builder


def test_single_token_fast_path_matches_multi_token_path_bitwise() -> None:
    torch.manual_seed(0)
    cache = {7: torch.randn(4), 9: torch.randn(4)}
    builder = _builder(dict(cache))

    fast = builder._load_prompt_token_embeddings(torch.tensor([7], dtype=torch.long))
    general = builder._load_prompt_token_embeddings(
        torch.tensor([7, 9], dtype=torch.long)
    )

    assert fast.shape == (1, 4)
    assert fast.dtype == general.dtype
    assert torch.equal(fast[0], general[0])
    assert torch.equal(fast[0], cache[7])


def test_single_token_fast_path_does_not_alias_cache() -> None:
    cache = {7: torch.zeros(4)}
    builder = _builder(cache)

    fast = builder._load_prompt_token_embeddings(torch.tensor([7], dtype=torch.long))
    fast[0] = 5.0

    assert torch.equal(cache[7], torch.zeros(4))


def test_single_token_fast_path_loads_and_caches_missing_row(monkeypatch) -> None:
    loaded_row = torch.arange(4, dtype=torch.float32)
    calls: list[list[int]] = []

    def _fake_load(model_path: str, row_ids: list[int]) -> torch.Tensor:
        calls.append(row_ids)
        return loaded_row.unsqueeze(0)

    monkeypatch.setattr(talker_prefill, "load_thinker_embedding_rows", _fake_load)
    builder = _builder({})

    out = builder._load_prompt_token_embeddings(torch.tensor([3], dtype=torch.long))
    again = builder._load_prompt_token_embeddings(torch.tensor([3], dtype=torch.long))

    assert calls == [[3]]
    assert torch.equal(out, loaded_row.unsqueeze(0))
    assert torch.equal(again, out)
    assert 3 in builder._thinker_embed_cache
