# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.qwen3_omni.components import talker_prefill

VOCAB, HIDDEN = 32, 8


@pytest.fixture()
def model_dir(tmp_path):
    weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    shard = "model-00001-of-00001.safetensors"
    save_file({"thinker.model.embed_tokens.weight": weight}, str(tmp_path / shard))
    index = {"weight_map": {"thinker.model.embed_tokens.weight": shard}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    talker_prefill._EMBED_SOURCE_CACHE.clear()
    talker_prefill._EMBED_HANDLE_CACHE.clear()
    try:
        yield tmp_path
    finally:
        talker_prefill._EMBED_SOURCE_CACHE.clear()
        talker_prefill._EMBED_HANDLE_CACHE.clear()


def test_rows_correct(model_dir):
    rows = talker_prefill.load_thinker_embedding_rows(str(model_dir), [3, 0, 7])
    expected = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    assert torch.equal(rows, expected[[3, 0, 7]])


def test_index_parsed_once_across_calls(model_dir, monkeypatch):
    calls = {"n": 0}
    real_loads = json.loads

    def counting_loads(*args, **kwargs):
        calls["n"] += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(talker_prefill.json, "loads", counting_loads)
    talker_prefill.load_thinker_embedding_rows(str(model_dir), [1])
    talker_prefill.load_thinker_embedding_rows(str(model_dir), [2])
    talker_prefill.load_thinker_embedding_rows(str(model_dir), [3, 4])
    assert calls["n"] == 1


def test_shard_opened_once_across_calls(model_dir, monkeypatch):
    opens = {"n": 0}
    real_safe_open = talker_prefill.safe_open

    def counting_safe_open(*args, **kwargs):
        opens["n"] += 1
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(talker_prefill, "safe_open", counting_safe_open)
    talker_prefill.load_thinker_embedding_rows(str(model_dir), [1])
    talker_prefill.load_thinker_embedding_rows(str(model_dir), [2, 9])
    talker_prefill.load_thinker_embedding_rows(str(model_dir), [15])
    assert opens["n"] == 1


def test_repeated_rows_stay_identical(model_dir):
    first = talker_prefill.load_thinker_embedding_rows(str(model_dir), [4, 11])
    again = talker_prefill.load_thinker_embedding_rows(str(model_dir), [11, 4])
    assert torch.equal(again, first[[1, 0]])


def test_no_index_fallback_cached(model_dir, monkeypatch):
    (model_dir / "model.safetensors.index.json").unlink()
    talker_prefill._EMBED_SOURCE_CACHE.clear()
    talker_prefill._EMBED_HANDLE_CACHE.clear()
    rows = talker_prefill.load_thinker_embedding_rows(str(model_dir), [5])
    expected = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    assert torch.equal(rows, expected[[5]])
    globs = {"n": 0}
    real_glob = talker_prefill.Path.glob

    def counting_glob(self, pattern):
        globs["n"] += 1
        return real_glob(self, pattern)

    monkeypatch.setattr(talker_prefill.Path, "glob", counting_glob)
    talker_prefill.load_thinker_embedding_rows(str(model_dir), [6])
    assert globs["n"] == 0


def test_missing_weights_raises(tmp_path):
    talker_prefill._EMBED_SOURCE_CACHE.clear()
    with pytest.raises(KeyError):
        talker_prefill.load_thinker_embedding_rows(str(tmp_path), [0])
