# SPDX-License-Identifier: Apache-2.0
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

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


def _prefill_builder(model_dir, dtype=torch.float32):
    builder = object.__new__(talker_prefill.TalkerPrefillBuilder)
    builder._model_path = str(model_dir)
    builder._device = torch.device("cpu")
    builder._dtype = dtype
    builder._thinker_embed_cache = {}
    builder._model = SimpleNamespace(
        text_projection=torch.nn.Sequential(
            torch.nn.Linear(HIDDEN, HIDDEN),
            torch.nn.SiLU(),
            torch.nn.Linear(HIDDEN, HIDDEN),
        ).to(dtype=dtype)
    )
    return builder


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("token_id", [3, "3", 3.5])
def test_project_cached_chunk_preserves_singleton_projection(
    model_dir, monkeypatch, dtype, token_id
):
    builder = _prefill_builder(model_dir, dtype)
    original_input = builder._load_prompt_token_embeddings(torch.tensor([3]))
    cached_row = builder._thinker_embed_cache[3]
    saved_row = cached_row.clone()
    expected = builder._model.text_projection(original_input)[0].detach()

    def reject_loader(_ids):
        pytest.fail("A cached singleton must not rebuild the embedding matrix")

    monkeypatch.setattr(builder, "_load_prompt_token_embeddings", reject_loader)
    seen = []
    handle = builder._model.text_projection.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0])
    )
    try:
        chunk = SimpleNamespace(metadata={"token_id": token_id})
        first = builder.project_assistant_chunk(chunk)
        assert torch.equal(first, expected)
        first.zero_()
        again = builder.project_assistant_chunk(chunk)
    finally:
        handle.remove()

    assert torch.equal(again, expected)
    assert not again.requires_grad
    assert again.stride() == expected.stride()
    assert torch.equal(cached_row, saved_row)
    for projection_input in seen:
        assert projection_input.shape == original_input.shape == (1, HIDDEN)
        assert projection_input.stride() == original_input.stride()
        assert projection_input.dtype == dtype
        assert projection_input.device == original_input.device
        assert projection_input.data_ptr() == cached_row.data_ptr()


def test_project_cache_miss_keeps_loader_and_then_reuses_row(model_dir, monkeypatch):
    builder = _prefill_builder(model_dir)
    load = builder._load_prompt_token_embeddings
    calls = []

    def observe_load(ids):
        assert ids.dtype == torch.long and ids.device.type == "cpu"
        calls.append(ids.tolist())
        return load(ids)

    monkeypatch.setattr(builder, "_load_prompt_token_embeddings", observe_load)
    chunk = SimpleNamespace(metadata={"token_id": "7"})
    first = builder.project_assistant_chunk(chunk)
    again = builder.project_assistant_chunk(chunk)
    assert calls == [[7]]
    assert torch.equal(first, again)
    assert torch.equal(builder._thinker_embed_cache[7], torch.arange(56.0, 64.0))


@pytest.mark.parametrize("metadata", [None, {}, {"token_id": None}])
def test_project_chunk_without_token_id_keeps_data_conversion(model_dir, metadata):
    builder = _prefill_builder(model_dir, torch.bfloat16)
    data = torch.arange(HIDDEN * 2, dtype=torch.float64)[::2]
    expected = builder._model.text_projection(data.to(torch.bfloat16).unsqueeze(0))
    actual = builder.project_assistant_chunk(
        SimpleNamespace(metadata=metadata, data=data)
    )
    assert torch.equal(actual, expected[0])
    assert actual.dtype == torch.bfloat16 and actual.device.type == "cpu"
    assert not actual.requires_grad
    assert builder._thinker_embed_cache == {}


@pytest.mark.parametrize("token_id,error", [("invalid", ValueError), ([], TypeError)])
def test_project_chunk_keeps_token_conversion_errors(model_dir, token_id, error):
    builder = _prefill_builder(model_dir)
    with pytest.raises(error):
        builder.project_assistant_chunk(
            SimpleNamespace(metadata={"token_id": token_id})
        )
    assert builder._thinker_embed_cache == {}


def test_project_chunk_keeps_cache_miss_error(model_dir, monkeypatch):
    builder = _prefill_builder(model_dir)
    failure = RuntimeError("embedding load failed")

    def fail(_ids):
        raise failure

    monkeypatch.setattr(builder, "_load_prompt_token_embeddings", fail)
    with pytest.raises(RuntimeError) as caught:
        builder.project_assistant_chunk(SimpleNamespace(metadata={"token_id": 7}))
    assert caught.value is failure


@pytest.mark.parametrize("ids", [[3], [3, 3, 7]])
def test_generic_embedding_matrix_still_does_not_alias_cache(model_dir, ids):
    builder = _prefill_builder(model_dir)
    token_ids = torch.tensor(ids)
    original = builder._load_prompt_token_embeddings(token_ids)
    expected = original.clone()
    saved_cache = {
        token: row.clone() for token, row in builder._thinker_embed_cache.items()
    }
    matrix = builder._load_prompt_token_embeddings(token_ids)
    matrix[0].add_(1000)
    if len(ids) > 1:
        assert torch.equal(matrix[1], expected[1])
    assert torch.equal(original, expected)
    assert all(
        torch.equal(row, saved_cache[token])
        for token, row in builder._thinker_embed_cache.items()
    )
    assert torch.equal(builder._load_prompt_token_embeddings(token_ids), expected)


@pytest.mark.accelerator
@pytest.mark.parametrize("device_index", [0, 1])
@pytest.mark.parametrize("side_stream", [False, True])
def test_project_cached_chunk_cuda_unquantized_projection(device_index, side_stream):
    """Component equivalence on real CUDA, with synthetic BF16 projection weights.

    Run separately on the serving runtime. This does not load the full model or
    establish serving latency, speech quality, or arbitrary quantized behavior.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() <= device_index:
        pytest.skip("requires the selected CUDA device")
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    from sglang_omni.models.qwen3_omni.components.talker import ResizeMLP
    from sglang_omni.models.qwen3_omni.hf_config import Qwen3OmniMoeTalkerConfig

    config = Qwen3OmniMoeTalkerConfig()
    device = torch.device("cuda", device_index)
    dtype = torch.bfloat16
    with torch.cuda.device(device), torch.inference_mode():
        projection = ResizeMLP(
            config.thinker_hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
            quant_config=None,
        ).to(device=device, dtype=dtype)
        for parameter in projection.parameters():
            parameter.uniform_(-0.02, 0.02)
        assert isinstance(projection.linear_fc1.quant_method, UnquantizedLinearMethod)
        assert isinstance(projection.linear_fc2.quant_method, UnquantizedLinearMethod)
        row = torch.linspace(
            -1, 1, config.thinker_hidden_size, device=device, dtype=dtype
        )
        saved_row = row.clone()
        ready = torch.cuda.Event()
        ready.record()
        stream = (
            torch.cuda.Stream(device=device)
            if side_stream
            else torch.cuda.default_stream(device)
        )
    builder = object.__new__(talker_prefill.TalkerPrefillBuilder)
    builder._device, builder._dtype = device, dtype
    builder._thinker_embed_cache = {3: row}
    builder._model = SimpleNamespace(text_projection=projection)

    def compare():
        with (
            torch.cuda.device(device),
            torch.cuda.stream(stream),
            torch.inference_mode(),
        ):
            stream.wait_event(ready)
            baseline_input = builder._load_prompt_token_embeddings(torch.tensor([3]))
            candidate_input = row.unsqueeze(0)
            assert candidate_input.shape == baseline_input.shape
            assert candidate_input.stride() == baseline_input.stride()
            assert candidate_input.dtype == baseline_input.dtype == dtype
            assert candidate_input.device == baseline_input.device == device
            expected = projection(baseline_input)[0].detach()
            actual = builder.project_assistant_chunk(
                SimpleNamespace(metadata={"token_id": 3})
            )
            assert torch.cuda.current_stream(device) == stream
            assert torch.cuda.current_device() == device_index
            assert actual.shape == expected.shape
            assert actual.stride() == expected.stride()
            assert actual.dtype == dtype and actual.device == device
            assert not actual.requires_grad
            # Synchronization is test-only, after both paths have enqueued work.
            stream.synchronize()
            assert torch.equal(actual, expected)
            assert torch.equal(row, saved_row)

    compare()
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(compare).result(timeout=60)
