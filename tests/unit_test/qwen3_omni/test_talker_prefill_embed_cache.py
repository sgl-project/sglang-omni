# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.qwen3_omni.components import talker_prefill
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload

VOCAB, HIDDEN = 32, 8


@pytest.fixture()
def model_dir(tmp_path):
    weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    shard = "model-00001-of-00001.safetensors"
    save_file({"thinker.model.embed_tokens.weight": weight}, str(tmp_path / shard))
    index = {"weight_map": {"thinker.model.embed_tokens.weight": shard}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    talker_prefill._EMBED_SOURCE_CACHE.clear()  # noqa: leading-underscore  # production name
    talker_prefill._EMBED_HANDLE_CACHE.clear()  # noqa: leading-underscore  # production name
    try:
        yield tmp_path
    finally:
        talker_prefill._EMBED_SOURCE_CACHE.clear()  # noqa: leading-underscore  # production name
        talker_prefill._EMBED_HANDLE_CACHE.clear()  # noqa: leading-underscore  # production name


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
    talker_prefill._EMBED_SOURCE_CACHE.clear()  # noqa: leading-underscore  # production name
    talker_prefill._EMBED_HANDLE_CACHE.clear()  # noqa: leading-underscore  # production name
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
    talker_prefill._EMBED_SOURCE_CACHE.clear()  # noqa: leading-underscore  # production name
    with pytest.raises(KeyError):
        talker_prefill.load_thinker_embedding_rows(str(tmp_path), [0])


@pytest.fixture()
def prompt_builder(model_dir):
    codec_embedding = torch.nn.Embedding(VOCAB, HIDDEN)
    with torch.no_grad():
        codec_embedding.weight.zero_()
    model = SimpleNamespace(
        model=SimpleNamespace(codec_embedding=codec_embedding),
        activation_dtype=torch.float32,
        text_projection=lambda rows: rows * 2,
        hidden_projection=lambda rows: rows + 100,
        get_input_embeddings=lambda: codec_embedding,
    )
    return talker_prefill.TalkerPrefillBuilder(
        model=model,
        model_path=str(model_dir),
        audio_token_id=3,
        image_token_id=4,
        video_token_id=5,
        tts_bos_token_id=10,
        tts_eos_token_id=11,
        tts_pad_token_id=12,
        im_start_token_id=1,
        im_end_token_id=9,
        system_token_id=19,
        user_token_id=2,
        assistant_token_id=6,
        codec_bos_id=13,
        codec_nothink_id=14,
        codec_think_bos_id=15,
        codec_think_eos_id=16,
        codec_pad_id=17,
    )


def build_streamed_prompt(builder, prompt_ids, chunks):
    payload = StagePayload(
        request_id="prompt-capture",
        request=OmniRequest(inputs=[], params={}),
        data={"prompt": {"input_ids": torch.tensor([prompt_ids])}},
    )
    return builder.build_prompt_prefill(payload, chunks, thinker_done=False)


def test_multimodal_prompt_uses_captured_thinker_states(prompt_builder):
    prompt_ids = [1, 2, 3, 4, 5, 8, 9, 1, 6, 7]
    captured = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN) + 7
    first_chunk = StreamItem(
        chunk_id=0,
        from_stage="thinker",
        data=captured,
        metadata={"token_id": 20, "prompt_hidden_positions": [2, 3, 4]},
    )
    second_chunk = StreamItem(
        chunk_id=1,
        from_stage="thinker",
        data=torch.tensor([21]),
        metadata={"token_id": 21},
    )
    result = build_streamed_prompt(
        prompt_builder, prompt_ids, [first_chunk, second_chunk]
    )
    expected_user = (
        torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)[
            prompt_ids[:7]
        ]
        * 2
    )
    expected_user[2:5] = captured + 100
    torch.testing.assert_close(result["input_embeds"][:7], expected_user)
    expected_next_text = torch.arange(21 * HIDDEN, 22 * HIDDEN).float() * 2
    torch.testing.assert_close(
        result["pending_text_queue"].popleft(), expected_next_text
    )


def test_prompt_rows_at_other_than_the_multimodal_rows_are_rejected(prompt_builder):
    chunk = StreamItem(
        chunk_id=0,
        from_stage="thinker",
        data=torch.ones((3, HIDDEN)),
        metadata={"token_id": 20, "prompt_hidden_positions": [2, 3, 6]},
    )
    with pytest.raises(AssertionError, match="multimodal rows"):
        build_streamed_prompt(prompt_builder, [1, 2, 3, 4, 5, 8, 9, 1, 6, 7], [chunk])


def test_a_session_turn_without_prompt_rows_keeps_zeros_at_multimodal_rows(
    prompt_builder,
):
    prompt_ids = [1, 2, 3, 4, 5, 8, 9, 1, 6, 7]
    chunk = StreamItem(
        chunk_id=0,
        from_stage="thinker",
        data=torch.tensor([20]),
        metadata={"token_id": 20},
    )
    result = build_streamed_prompt(prompt_builder, prompt_ids, [chunk])
    expected_user = (
        torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)[
            prompt_ids[:7]
        ]
        * 2
    )
    expected_user[2:5] = 100.0
    torch.testing.assert_close(result["input_embeds"][:7], expected_user)


def test_text_prompt_needs_only_streamed_token_ids(prompt_builder):
    prompt_ids = [1, 2, 8, 9, 1, 6, 7]
    chunk = StreamItem(
        chunk_id=0,
        from_stage="thinker",
        data=torch.tensor([20]),
        metadata={"token_id": 20},
    )
    result = build_streamed_prompt(prompt_builder, prompt_ids, [chunk])
    expected_user = (
        torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)[
            prompt_ids[:4]
        ]
        * 2
    )
    torch.testing.assert_close(result["input_embeds"][:4], expected_user)
