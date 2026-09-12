# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
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


def test_fallback_scan_ignores_directories_named_safetensors(
    tmp_path, clear_embed_caches
):
    weight = _dense_table()
    (tmp_path / ".downloads" / "partial.safetensors").mkdir(parents=True)
    save_file(
        {"thinker.model.embed_tokens.weight": weight},
        str(tmp_path / "model.safetensors"),
    )

    rows = talker_prefill.load_thinker_embedding_rows(str(tmp_path), [3, 0, 7])

    assert torch.equal(rows, weight[[3, 0, 7]])


@pytest.fixture()
def clear_embed_caches():
    talker_prefill._EMBED_SOURCE_CACHE.clear()
    talker_prefill._EMBED_HANDLE_CACHE.clear()
    try:
        yield
    finally:
        talker_prefill._EMBED_SOURCE_CACHE.clear()
        talker_prefill._EMBED_HANDLE_CACHE.clear()


def _dense_table(offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
        + offset
    )


def test_converted_component_export_reads_the_thinker_shard(
    tmp_path, clear_embed_caches
):
    """A converted MLX export keeps each component in its own subdirectory.

    Stripped of the component prefix the thinker and the talker both own
    ``model.embed_tokens.weight``, so ownership has to come from the shard's
    directory. A root-level scan finds neither shard; a recursive scan that
    trusts key text alone can splice the *talker's* embedding table into the
    talker prompt without any error.
    """

    thinker = _dense_table()
    talker = _dense_table(offset=10_000.0)
    (tmp_path / "thinker").mkdir()
    (tmp_path / "talker").mkdir()
    save_file(
        {"model.embed_tokens.weight": thinker},
        str(tmp_path / "thinker" / "model.safetensors"),
    )
    save_file(
        {"model.embed_tokens.weight": talker},
        str(tmp_path / "talker" / "model.safetensors"),
    )

    rows = talker_prefill.load_thinker_embedding_rows(str(tmp_path), [3, 0, 7])

    assert torch.equal(rows, thinker[[3, 0, 7]])


def test_converted_4bit_thinker_embedding_is_dequantized(tmp_path, clear_embed_caches):
    """A converted 4-bit export stores the embedding packed, not as floats.

    ``mlx.nn.quantize`` replaces the embedding with a ``QuantizedEmbedding``
    whose ``weight`` is a ``uint32`` bit-packed tensor plus ``scales`` and
    ``biases``. Reading rows out of that tensor as if it were the float table
    yields packed integers, so the prompt embeddings must be dequantized.
    """

    mx = pytest.importorskip("mlx.core")

    # MLX affine quantization supports group sizes 32/64/128 only, so the
    # probe table is as wide as the smallest supported group.
    group_size = 32
    dense = torch.arange(VOCAB * group_size, dtype=torch.float32).reshape(
        VOCAB, group_size
    ) / (VOCAB * group_size)
    packed, scales, biases = mx.quantize(
        mx.array(dense.numpy()), group_size=group_size, bits=4
    )
    (tmp_path / "thinker").mkdir()
    mx.save_safetensors(
        str(tmp_path / "thinker" / "model.safetensors"),
        {
            "model.embed_tokens.weight": packed,
            "model.embed_tokens.scales": scales,
            "model.embed_tokens.biases": biases,
        },
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": group_size}})
    )

    rows = talker_prefill.load_thinker_embedding_rows(str(tmp_path), [3, 0, 7])

    assert rows.dtype is torch.float32
    assert rows.shape == (3, group_size)
    torch.testing.assert_close(rows, dense[[3, 0, 7]], atol=0.02, rtol=0.02)


def test_native_mlx_prefill_dequantizes_only_requested_thinker_rows(
    tmp_path, clear_embed_caches, monkeypatch
):
    mx = pytest.importorskip("mlx.core")
    from sglang_omni.models.qwen3_omni.mlx.talker_prefill import (
        _load_mlx_embedding_rows,
    )

    group_size = 32
    dense = torch.arange(VOCAB * group_size, dtype=torch.float32).reshape(
        VOCAB, group_size
    ) / (VOCAB * group_size)
    packed, scales, biases = mx.quantize(
        mx.array(dense.numpy()), group_size=group_size, bits=4
    )
    component = tmp_path / "thinker"
    component.mkdir()
    mx.save_safetensors(
        str(component / "model.safetensors"),
        {
            "model.embed_tokens.weight": packed,
            "model.embed_tokens.scales": scales,
            "model.embed_tokens.biases": biases,
        },
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": group_size}})
    )
    monkeypatch.setattr(
        mx,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "row-selective embedding loading must not materialize a full shard"
        ),
    )

    rows = _load_mlx_embedding_rows(str(tmp_path), [3, 0, 7])

    assert rows.shape == (3, group_size)
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(rows.astype(mx.float32))),
        dense[[3, 0, 7]],
        atol=0.02,
        rtol=0.02,
    )


def test_bfloat16_packed_metadata_rows_work_for_torch_and_mlx_consumers(
    tmp_path, clear_embed_caches, monkeypatch
):
    mx = pytest.importorskip("mlx.core")
    from sglang_omni.models.qwen3_omni.mlx.talker_prefill import (
        _load_mlx_embedding_rows,
    )

    group_size = 32
    row_ids = [3, 0, 7]
    dense = torch.arange(VOCAB * group_size, dtype=torch.float32).reshape(
        VOCAB, group_size
    ) / (VOCAB * group_size)
    packed, scales, biases = mx.quantize(
        mx.array(dense.numpy()), group_size=group_size, bits=4
    )
    bf16_scales = scales.astype(mx.bfloat16)
    bf16_biases = biases.astype(mx.bfloat16)
    component = tmp_path / "thinker"
    component.mkdir()
    save_file(
        {
            "model.embed_tokens.weight": torch.from_numpy(np.asarray(packed)),
            "model.embed_tokens.scales": torch.from_numpy(np.asarray(scales)).to(
                torch.bfloat16
            ),
            "model.embed_tokens.biases": torch.from_numpy(np.asarray(biases)).to(
                torch.bfloat16
            ),
        },
        str(component / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": group_size}})
    )
    monkeypatch.setattr(
        mx,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "packed embedding rows must remain selectively sliced"
        ),
    )

    source = talker_prefill._resolve_embed_source(str(tmp_path))
    torch_rows = talker_prefill._packed_embedding_rows(str(tmp_path), source, row_ids)
    mlx_rows = _load_mlx_embedding_rows(str(tmp_path), row_ids)
    reference = mx.dequantize(
        packed[mx.array(row_ids)],
        bf16_scales[mx.array(row_ids)],
        bf16_biases[mx.array(row_ids)],
        group_size=group_size,
        bits=4,
        mode="affine",
    )

    assert torch_rows.dtype is torch.float32
    assert mlx_rows.dtype == reference.dtype == mx.bfloat16
    reference_torch = torch.from_numpy(np.asarray(reference.astype(mx.float32)))
    torch.testing.assert_close(torch_rows, reference_torch, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(mlx_rows.astype(mx.float32))),
        reference_torch,
        rtol=0,
        atol=0,
    )


def test_native_mlx_prefill_reads_bfloat16_thinker_rows(tmp_path, clear_embed_caches):
    mx = pytest.importorskip("mlx.core")
    from sglang_omni.models.qwen3_omni.mlx.talker_prefill import (
        _load_mlx_embedding_rows,
    )

    weight = _dense_table().to(torch.bfloat16)
    save_file(
        {"thinker.model.embed_tokens.weight": weight},
        str(tmp_path / "model.safetensors"),
    )

    rows = _load_mlx_embedding_rows(str(tmp_path), [3, 0, 7])

    assert rows.dtype == mx.bfloat16
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(rows.astype(mx.float32))),
        weight[[3, 0, 7]].float(),
    )


def test_native_mlx_prefill_uses_supported_torch_safetensors_slicing(
    tmp_path, clear_embed_caches, monkeypatch
):
    mx = pytest.importorskip("mlx.core")
    from sglang_omni.models.qwen3_omni.mlx import talker_prefill as mlx_prefill

    weight = _dense_table().to(torch.bfloat16)
    save_file(
        {"thinker.model.embed_tokens.weight": weight},
        str(tmp_path / "model.safetensors"),
    )

    calls = []
    real_safe_open = talker_prefill.safe_open

    def recording_safe_open(*args, **kwargs):
        calls.append(kwargs)
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(talker_prefill, "safe_open", recording_safe_open)

    rows = mlx_prefill._load_mlx_embedding_rows(str(tmp_path), [3, 0, 7])

    assert calls == [
        {"framework": "pt", "device": "cpu"},
        {"framework": "pt", "device": "cpu"},
    ]
    assert rows.dtype == mx.bfloat16
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(rows.astype(mx.float32))),
        weight[[3, 0, 7]].float(),
    )


def test_native_mlx_prefill_reuses_supported_safetensors_handle_for_unseen_rows(
    tmp_path, clear_embed_caches, monkeypatch
):
    pytest.importorskip("mlx.core")
    from sglang_omni.models.qwen3_omni.mlx import talker_prefill as mlx_prefill

    weight = _dense_table().to(torch.bfloat16)
    shard = tmp_path / "model.safetensors"
    save_file({"thinker.model.embed_tokens.weight": weight}, str(shard))
    talker_prefill._resolve_embed_source(str(tmp_path))

    calls = {"safe_open": 0}
    real_safe_open = talker_prefill.safe_open

    def counting_safe_open(*args, **kwargs):
        calls["safe_open"] += 1
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(talker_prefill, "safe_open", counting_safe_open)
    builder = mlx_prefill.Qwen3OmniMlxTalkerPrefillBuilder.from_talker(
        SimpleNamespace(text_projection=lambda rows: rows),
        model_path=str(tmp_path),
        special_token_ids={"im_end_token_id": -1},
        speaker_map={},
    )

    first = builder._project_assistant_chunk(
        SimpleNamespace(data=None, metadata={"token_id": 3})
    )
    second = builder._project_assistant_chunk(
        SimpleNamespace(data=None, metadata={"token_id": 7})
    )

    torch.testing.assert_close(first, weight[3].float())
    torch.testing.assert_close(second, weight[7].float())
    assert calls == {"safe_open": 1}


def test_mlx_community_indexed_thinker_embedding_is_dequantized(
    tmp_path, clear_embed_caches
):
    mx = pytest.importorskip("mlx.core")

    group_size = 64
    dense = torch.arange(VOCAB * group_size, dtype=torch.float32).reshape(
        VOCAB, group_size
    ) / (VOCAB * group_size)
    packed, scales, biases = mx.quantize(
        mx.array(dense.numpy()), group_size=group_size, bits=4
    )
    prefix = "thinker.language_model.model.embed_tokens"
    shard = "model-00001-of-00001.safetensors"
    tensors = {
        f"{prefix}.weight": packed,
        f"{prefix}.scales": scales,
        f"{prefix}.biases": biases,
    }
    mx.save_safetensors(str(tmp_path / shard), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "bits": 4,
                    "group_size": group_size,
                    "mode": "affine",
                }
            }
        ),
        encoding="utf-8",
    )

    rows = talker_prefill.load_thinker_embedding_rows(str(tmp_path), [3, 0, 7])

    assert rows.dtype is torch.float32
    assert rows.shape == (3, group_size)
    torch.testing.assert_close(rows, dense[[3, 0, 7]], atol=0.02, rtol=0.02)


def test_indexed_packed_embedding_tensors_can_use_different_shards(
    tmp_path, clear_embed_caches
):
    mx = pytest.importorskip("mlx.core")

    group_size = 64
    dense = torch.arange(VOCAB * group_size, dtype=torch.float32).reshape(
        VOCAB, group_size
    ) / (VOCAB * group_size)
    packed, scales, biases = mx.quantize(
        mx.array(dense.numpy()), group_size=group_size, bits=4
    )
    prefix = "thinker.language_model.model.embed_tokens"
    tensor_shards = {
        f"{prefix}.weight": "weight.safetensors",
        f"{prefix}.scales": "scales.safetensors",
        f"{prefix}.biases": "biases.safetensors",
    }
    mx.save_safetensors(
        str(tmp_path / tensor_shards[f"{prefix}.weight"]),
        {f"{prefix}.weight": packed},
    )
    mx.save_safetensors(
        str(tmp_path / tensor_shards[f"{prefix}.scales"]),
        {f"{prefix}.scales": scales},
    )
    mx.save_safetensors(
        str(tmp_path / tensor_shards[f"{prefix}.biases"]),
        {f"{prefix}.biases": biases},
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": tensor_shards}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "bits": 4,
                    "group_size": group_size,
                    "mode": "affine",
                }
            }
        ),
        encoding="utf-8",
    )

    rows = talker_prefill.load_thinker_embedding_rows(str(tmp_path), [3, 0, 7])

    torch.testing.assert_close(rows, dense[[3, 0, 7]], atol=0.02, rtol=0.02)


def test_component_indexed_packed_embedding_tensors_can_use_different_shards(
    tmp_path, clear_embed_caches
):
    mx = pytest.importorskip("mlx.core")

    group_size = 64
    dense = torch.arange(VOCAB * group_size, dtype=torch.float32).reshape(
        VOCAB, group_size
    ) / (VOCAB * group_size)
    packed, scales, biases = mx.quantize(
        mx.array(dense.numpy()), group_size=group_size, bits=4
    )
    component = tmp_path / "thinker"
    component.mkdir()
    prefix = "model.embed_tokens"
    tensor_shards = {
        f"{prefix}.weight": "weight.safetensors",
        f"{prefix}.scales": "scales.safetensors",
        f"{prefix}.biases": "biases.safetensors",
    }
    mx.save_safetensors(
        str(component / tensor_shards[f"{prefix}.weight"]),
        {f"{prefix}.weight": packed},
    )
    mx.save_safetensors(
        str(component / tensor_shards[f"{prefix}.scales"]),
        {f"{prefix}.scales": scales},
    )
    mx.save_safetensors(
        str(component / tensor_shards[f"{prefix}.biases"]),
        {f"{prefix}.biases": biases},
    )
    (component / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": tensor_shards}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "bits": 4,
                    "group_size": group_size,
                    "mode": "affine",
                }
            }
        ),
        encoding="utf-8",
    )

    rows = talker_prefill.load_thinker_embedding_rows(str(tmp_path), [3, 0, 7])

    torch.testing.assert_close(rows, dense[[3, 0, 7]], atol=0.02, rtol=0.02)
