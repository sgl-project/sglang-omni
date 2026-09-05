# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig, Qwen3Config

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_tts_realtime.sglang_model import (
    MossTTSRealtimeSGLangModel,
    _normalize_config,
    expected_moss_tts_realtime_checkpoint_keys,
)


def _runtime_config() -> PretrainedConfig:
    config = PretrainedConfig(
        architectures=["MossTTSRealtime"],
        language_config=Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=40960,
            tie_word_embeddings=False,
        ),
        local_config=PretrainedConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=33,
            rvq=16,
        ),
        rvq=16,
        audio_pad_token=1024,
        audio_vocab_size=1027,
        reference_audio_pad=151654,
        text_pad=151655,
    )
    return _normalize_config(config)


def test_model_config_derives_sglang_channel_metadata() -> None:
    config = _runtime_config()
    keys = expected_moss_tts_realtime_checkpoint_keys(config)

    assert config.channels == 17
    assert config.vocab_size_list == [32, *([1027] * 16)]
    assert config.pad_token == [151655, *([1024] * 16)]
    assert len(keys) == 73
    assert "embed_tokens.16.weight" in keys
    assert "local_transformer.model.embed_tokens.14.weight" in keys
    assert "local_transformer.local_lm_heads.15.weight" in keys


def test_model_worker_arch_override_uses_language_config() -> None:
    outer = _runtime_config()
    model_config = SimpleNamespace(hf_config=outer)

    ModelWorker._apply_arch_override(model_config, "MossTTSRealtimeSGLangModel")

    assert outer.architectures == ["MossTTSRealtimeSGLangModel"]
    assert model_config.hf_text_config is outer.language_config
    assert model_config.hidden_size == outer.language_config.hidden_size
    assert model_config.num_hidden_layers == outer.language_config.num_hidden_layers


def _embedding_only_model() -> MossTTSRealtimeSGLangModel:
    model = MossTTSRealtimeSGLangModel.__new__(MossTTSRealtimeSGLangModel)
    torch.nn.Module.__init__(model)
    model.config = _runtime_config()
    model.hidden_size = 8
    model.embed_tokens = torch.nn.ModuleList(
        [torch.nn.Embedding(32, 8)] + [torch.nn.Embedding(1027, 8) for _ in range(16)]
    )
    with torch.no_grad():
        for channel, embedding in enumerate(model.embed_tokens):
            embedding.weight.zero_()
            embedding.weight[:, channel % 8] = torch.arange(
                embedding.num_embeddings,
                dtype=embedding.weight.dtype,
            )
    return model


def test_mixed_embedding_sums_all_seventeen_columns() -> None:
    model = _embedding_only_model()
    rows = torch.tensor(
        [
            [3, *range(1, 17)],
            [4, *range(17, 33)],
        ],
        dtype=torch.long,
    )

    actual = model.get_input_embeddings(rows)
    expected = sum(
        embedding(rows[:, channel])
        for channel, embedding in enumerate(model.embed_tokens)
    )

    torch.testing.assert_close(actual, expected)
    changed = rows.clone()
    changed[0, 7] += 1
    assert not torch.equal(model.get_input_embeddings(changed), actual)


def test_rank_one_embedding_uses_audio_pad_rows() -> None:
    model = _embedding_only_model()
    token_ids = torch.tensor([1, 2], dtype=torch.long)

    actual = model.get_input_embeddings(token_ids)
    rows = torch.full((2, 17), 1024, dtype=torch.long)
    rows[:, 0] = token_ids
    expected = model.get_input_embeddings(rows)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "rows",
    [
        torch.zeros((2, 16), dtype=torch.long),
        torch.tensor([[32, *([0] * 16)]], dtype=torch.long),
        torch.tensor([[0, 1027, *([0] * 15)]], dtype=torch.long),
        torch.zeros((1, 17), dtype=torch.float32),
    ],
)
def test_mixed_embedding_rejects_invalid_rows(rows: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        _embedding_only_model().get_input_embeddings(rows)


def test_forward_reads_decode_embeddings_from_stable_table() -> None:
    captured: dict[str, object] = {}

    class FakeBackbone:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return kwargs["input_embeds"]

    model = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        language_model=FakeBackbone(),
        _decode_input_embedding=torch.nn.Embedding.from_pretrained(
            torch.arange(24, dtype=torch.float32).reshape(3, 8),
            freeze=True,
        ),
        _select_sample_hidden_states=(
            MossTTSRealtimeSGLangModel._select_sample_hidden_states
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: True,
            is_extend=lambda: False,
        ),
    )

    output = MossTTSRealtimeSGLangModel.forward(
        model,
        input_ids=torch.tensor([2, 0]),
        positions=torch.tensor([3, 4]),
        forward_batch=forward_batch,
    )

    projected = model._decode_input_embedding(torch.tensor([2, 0]))
    assert captured["input_ids"] is None
    torch.testing.assert_close(captured["input_embeds"], projected)
    torch.testing.assert_close(output.hidden_states, projected)
    assert output.next_token_logits.shape == (2, 1)


def test_forward_returns_last_extend_hidden_per_request() -> None:
    class FakeBackbone:
        def __call__(self, **kwargs):
            del kwargs
            return torch.arange(40, dtype=torch.float32).reshape(5, 8)

    model = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        language_model=FakeBackbone(),
        _select_sample_hidden_states=(
            MossTTSRealtimeSGLangModel._select_sample_hidden_states
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_extend=lambda: True,
        ),
        extend_seq_lens=torch.tensor([2, 3]),
    )
    projected = torch.zeros((5, 8))

    output = MossTTSRealtimeSGLangModel.forward(
        model,
        input_ids=torch.arange(5),
        positions=torch.arange(5),
        forward_batch=forward_batch,
        input_embeds=projected,
    )

    expected = torch.arange(40, dtype=torch.float32).reshape(5, 8)[[1, 4]]
    torch.testing.assert_close(output.hidden_states, expected)
    assert output.next_token_logits.shape == (2, 1)


def test_decode_local_frame_falls_back_for_unsupported_graph_batch() -> None:
    calls: list[object] = []

    class FakeGraphRunner:
        def supports(self, batch_size: int) -> bool:
            calls.append(("supports", batch_size))
            return False

        def record_fallback(self) -> None:
            calls.append("fallback")

    class FakeLocalTransformer:
        def decode_frame(self, hidden_states, *, sample_audio, compute_logits=None):
            calls.append(("decode", hidden_states.shape, compute_logits))
            return sample_audio(torch.zeros(hidden_states.shape[0], 4), 0).view(-1, 1)

    model = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True),
        _local_cuda_graph_runner=FakeGraphRunner(),
        local_transformer=FakeLocalTransformer(),
    )
    hidden_states = torch.zeros(2, 8)

    result = MossTTSRealtimeSGLangModel.decode_local_frame(
        model,
        hidden_states,
        sample_audio=lambda logits, codebook: torch.full(
            (logits.shape[0],),
            codebook + 3,
            dtype=torch.long,
        ),
    )

    assert result.tolist() == [[3], [3]]
    assert calls == [
        ("supports", 2),
        "fallback",
        ("decode", torch.Size([2, 8]), None),
    ]


def _loader_shell(*, first_rank: bool) -> MossTTSRealtimeSGLangModel:
    model = MossTTSRealtimeSGLangModel.__new__(MossTTSRealtimeSGLangModel)
    torch.nn.Module.__init__(model)
    model.config = _runtime_config()
    model.pp_group = SimpleNamespace(
        is_first_rank=first_rank,
        is_last_rank=False,
    )
    model.language_model = SimpleNamespace(start_layer=0, end_layer=1)
    return model


def test_strict_loader_rejects_unknown_duplicate_and_missing_keys() -> None:
    model = _loader_shell(first_rank=False)
    tensor = torch.zeros(1)

    with pytest.raises(ValueError, match="unexpected"):
        model.load_weights([("unknown.weight", tensor)], strict=False)
    with pytest.raises(ValueError, match="duplicate"):
        model.load_weights(
            [
                ("embed_tokens.0.weight", tensor),
                ("embed_tokens.0.weight", tensor),
            ],
            strict=False,
        )
    with pytest.raises(ValueError, match="missing"):
        model.load_weights([], strict=True)


def test_loader_wraps_direct_shape_mismatch() -> None:
    parameter = torch.nn.Parameter(torch.zeros((2, 3)))
    with pytest.raises(ValueError, match="failed loading"):
        MossTTSRealtimeSGLangModel._load_param(
            "weight",
            parameter,
            torch.zeros((4, 3)),
        )


def test_loader_routes_qkv_shards_to_packed_parameter() -> None:
    calls: list[tuple[torch.Tensor, object]] = []
    parameter = torch.nn.Parameter(torch.zeros((6, 4)))

    def weight_loader(
        param: torch.nn.Parameter,
        loaded: torch.Tensor,
        shard_id: object,
    ) -> None:
        assert param is parameter
        calls.append((loaded, shard_id))

    parameter.weight_loader = weight_loader
    loaded = torch.ones((2, 4))
    params = {"language_model.layers.0.self_attn.qkv_proj.weight": parameter}

    assert MossTTSRealtimeSGLangModel._load_stacked_language_weight(
        "language_model.layers.0.self_attn.k_proj.weight",
        loaded,
        params,
    )
    assert calls == [(loaded, "k")]
