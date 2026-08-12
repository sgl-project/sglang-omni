# SPDX-License-Identifier: Apache-2.0
"""Differential tests: vectorized M-RoPE positions vs sglang HF port oracle."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
    get_rope_index_qwen3_omni,
)

from sglang_omni.models.qwen3_omni.mrope_positions import (
    _feat_extract_output_lengths,
    get_rope_index_qwen3_omni_vectorized,
)

# Defaults from Qwen3OmniMoeThinkerConfig
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
AUDIO_TOKEN_ID = 151646
AUDIO_START_TOKEN_ID = 151647
AUDIO_END_TOKEN_ID = 151648
POSITION_ID_PER_SECONDS = 25
SPATIAL_MERGE_SIZE = 2


def _oracle_and_fast(
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    audio_seqlens: torch.Tensor | None = None,
    use_audio_in_video: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    kwargs: dict[str, Any] = {
        "audio_token_id": AUDIO_TOKEN_ID,
        "audio_start_token_id": AUDIO_START_TOKEN_ID,
        "position_id_per_seconds": POSITION_ID_PER_SECONDS,
        "use_audio_in_video": use_audio_in_video,
        "audio_seqlens": audio_seqlens,
    }
    common = dict(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        tokens_per_second=None,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        **kwargs,
    )
    return get_rope_index_qwen3_omni(**common), get_rope_index_qwen3_omni_vectorized(
        **common
    )


def _assert_bit_identical(
    oracle: tuple[torch.Tensor, torch.Tensor],
    fast: tuple[torch.Tensor, torch.Tensor],
) -> None:
    o_pos, o_delta = oracle
    f_pos, f_delta = fast
    assert o_pos.shape == f_pos.shape, (o_pos.shape, f_pos.shape)
    assert torch.equal(o_pos.float(), f_pos.float()), (
        f"position mismatch: max abs diff="
        f"{(o_pos.float() - f_pos.float()).abs().max().item()}"
    )
    assert o_delta.shape == f_delta.shape, (o_delta.shape, f_delta.shape)
    assert torch.equal(
        o_delta.float(), f_delta.float()
    ), f"delta mismatch: oracle={o_delta} fast={f_delta}"


def _image_span(grid_thw: list[int]) -> list[int]:
    t, h, w = grid_thw
    image_len = (t * h * w) // (SPATIAL_MERGE_SIZE**2)
    return (
        [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * image_len + [VISION_END_TOKEN_ID]
    )


def _video_span(grid_thw: list[int]) -> list[int]:
    t, h, w = grid_thw
    video_len = (t * h * w) // (SPATIAL_MERGE_SIZE**2)
    return (
        [VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * video_len + [VISION_END_TOKEN_ID]
    )


def _audio_span(audio_seqlen: int) -> list[int]:
    audio_len = _feat_extract_output_lengths(audio_seqlen)
    return [AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * audio_len + [AUDIO_END_TOKEN_ID]


def _audio_in_video_span(grid_thw: list[int], audio_seqlen: int) -> list[int]:
    t, h, w = grid_thw
    video_len = (t * h * w) // (SPATIAL_MERGE_SIZE**2)
    audio_len = _feat_extract_output_lengths(audio_seqlen)
    # bos_len=2: vision_start, audio_start; then interleaved placeholders;
    # eos_len=2: vision_end, audio_end (order only affects st cursor length)
    return (
        [VISION_START_TOKEN_ID, AUDIO_START_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * video_len
        + [AUDIO_TOKEN_ID] * audio_len
        + [VISION_END_TOKEN_ID, AUDIO_END_TOKEN_ID]
    )


def test_feat_extract_lengths_matches_sglang() -> None:
    from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
        _get_feat_extract_output_lengths,
    )

    for n in (1, 50, 100, 101, 250, 1000):
        assert _feat_extract_output_lengths(n) == int(
            _get_feat_extract_output_lengths(torch.tensor(n)).item()
        )


def test_text_only_no_grids_falls_back_to_arange() -> None:
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    oracle, fast = _oracle_and_fast(ids)
    _assert_bit_identical(oracle, fast)


def test_single_image() -> None:
    grid = [1, 4, 4]
    tokens = [10, 11] + _image_span(grid) + [12, 13]
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([grid], dtype=torch.long)
    oracle, fast = _oracle_and_fast(ids, image_grid_thw=image_grid)
    _assert_bit_identical(oracle, fast)


def test_single_video() -> None:
    grid = [2, 4, 4]
    tokens = [10] + _video_span(grid) + [11]
    ids = torch.tensor([tokens], dtype=torch.long)
    video_grid = torch.tensor([grid], dtype=torch.long)
    seconds = torch.tensor([0.5], dtype=torch.float)
    oracle, fast = _oracle_and_fast(
        ids, video_grid_thw=video_grid, second_per_grid_ts=seconds
    )
    _assert_bit_identical(oracle, fast)


def test_image_then_audio() -> None:
    grid = [1, 8, 8]
    audio_seqlen = 200
    tokens = [1, 2] + _image_span(grid) + [3] + _audio_span(audio_seqlen) + [4]
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([grid], dtype=torch.long)
    audio_seqlens = torch.tensor([audio_seqlen], dtype=torch.long)
    oracle, fast = _oracle_and_fast(
        ids, image_grid_thw=image_grid, audio_seqlens=audio_seqlens
    )
    _assert_bit_identical(oracle, fast)


def test_audio_then_video() -> None:
    grid = [3, 4, 4]
    audio_seqlen = 150
    tokens = _audio_span(audio_seqlen) + _video_span(grid) + [99]
    ids = torch.tensor([tokens], dtype=torch.long)
    video_grid = torch.tensor([grid], dtype=torch.long)
    seconds = torch.tensor([1.0], dtype=torch.float)
    audio_seqlens = torch.tensor([audio_seqlen], dtype=torch.long)
    oracle, fast = _oracle_and_fast(
        ids,
        video_grid_thw=video_grid,
        second_per_grid_ts=seconds,
        audio_seqlens=audio_seqlens,
    )
    _assert_bit_identical(oracle, fast)


def test_two_images() -> None:
    g0, g1 = [1, 4, 4], [1, 6, 6]
    tokens = [0] + _image_span(g0) + [1, 2] + _image_span(g1) + [3]
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([g0, g1], dtype=torch.long)
    oracle, fast = _oracle_and_fast(ids, image_grid_thw=image_grid)
    _assert_bit_identical(oracle, fast)


def test_audio_in_video_interleaved() -> None:
    grid = [4, 4, 4]
    audio_seqlen = 300
    tokens = [7, 8] + _audio_in_video_span(grid, audio_seqlen) + [9]
    ids = torch.tensor([tokens], dtype=torch.long)
    video_grid = torch.tensor([grid], dtype=torch.long)
    seconds = torch.tensor([0.25], dtype=torch.float)
    audio_seqlens = torch.tensor([audio_seqlen], dtype=torch.long)
    oracle, fast = _oracle_and_fast(
        ids,
        video_grid_thw=video_grid,
        second_per_grid_ts=seconds,
        audio_seqlens=audio_seqlens,
        use_audio_in_video=True,
    )
    _assert_bit_identical(oracle, fast)


def test_mixed_image_video_audio_in_video() -> None:
    img = [1, 4, 4]
    vid = [2, 4, 4]
    audio_seqlen = 180
    tokens = (
        [1] + _image_span(img) + [2] + _audio_in_video_span(vid, audio_seqlen) + [3, 4]
    )
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([img], dtype=torch.long)
    video_grid = torch.tensor([vid], dtype=torch.long)
    seconds = torch.tensor([0.5], dtype=torch.float)
    audio_seqlens = torch.tensor([audio_seqlen], dtype=torch.long)
    oracle, fast = _oracle_and_fast(
        ids,
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        second_per_grid_ts=seconds,
        audio_seqlens=audio_seqlens,
        use_audio_in_video=True,
    )
    _assert_bit_identical(oracle, fast)


@pytest.mark.parametrize(
    "grid,audio_seqlen,seconds",
    [
        ([1, 4, 4], 50, 1.0),
        ([8, 2, 2], 400, 0.125),
        ([2, 8, 8], 100, 0.5),
    ],
)
def test_audio_in_video_parametrized(
    grid: list[int], audio_seqlen: int, seconds: float
) -> None:
    tokens = _audio_in_video_span(grid, audio_seqlen)
    ids = torch.tensor([tokens], dtype=torch.long)
    oracle, fast = _oracle_and_fast(
        ids,
        video_grid_thw=torch.tensor([grid], dtype=torch.long),
        second_per_grid_ts=torch.tensor([seconds], dtype=torch.float),
        audio_seqlens=torch.tensor([audio_seqlen], dtype=torch.long),
        use_audio_in_video=True,
    )
    _assert_bit_identical(oracle, fast)


def test_video_non_integer_timescale_25fps() -> None:
    """(arange * sec) * pps order must match oracle (25 FPS)."""
    grid = [12, 4, 4]
    tokens = [1] + _video_span(grid) + [2]
    ids = torch.tensor([tokens], dtype=torch.long)
    oracle, fast = _oracle_and_fast(
        ids,
        video_grid_thw=torch.tensor([grid], dtype=torch.long),
        second_per_grid_ts=torch.tensor([1.0 / 25.0], dtype=torch.float),
    )
    _assert_bit_identical(oracle, fast)


def test_audio_in_video_eos_uses_last_emitted_column() -> None:
    """AIV eos st_idx follows last emitted column, not max(merged)."""
    grid = [1, 56, 56]
    audio_seqlen = 100
    tokens = [1] + _audio_in_video_span(grid, audio_seqlen) + [2]
    ids = torch.tensor([tokens], dtype=torch.long)
    oracle, fast = _oracle_and_fast(
        ids,
        video_grid_thw=torch.tensor([grid], dtype=torch.long),
        second_per_grid_ts=torch.tensor([1.0], dtype=torch.float),
        audio_seqlens=torch.tensor([audio_seqlen], dtype=torch.long),
        use_audio_in_video=True,
    )
    _assert_bit_identical(oracle, fast)


def test_compute_mrope_positions_wires_vectorized_path(monkeypatch) -> None:
    """_compute_mrope_positions calls the vectorized path, [3, seq] layout."""
    from types import SimpleNamespace

    from sglang_omni.models.qwen3_omni import mrope_positions as mp
    from sglang_omni.models.qwen3_omni.request_builders import _compute_mrope_positions

    real_vectorized = mp.get_rope_index_qwen3_omni_vectorized
    calls: list[int] = []

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_vectorized(*args, **kwargs)

    monkeypatch.setattr(mp, "get_rope_index_qwen3_omni_vectorized", _spy)

    grid = [1, 4, 4]
    tokens = [10] + _image_span(grid) + [11]
    input_ids = torch.tensor(tokens, dtype=torch.long)
    model_inputs = {"image_grid_thw": torch.tensor([grid], dtype=torch.long)}
    thinker_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=None),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
    )
    result = _compute_mrope_positions(input_ids, model_inputs, thinker_config)
    assert calls, "hot path must call get_rope_index_qwen3_omni_vectorized"
    assert result is not None
    positions, delta = result
    assert positions.shape == (3, len(tokens))
    oracle_pos, oracle_delta = get_rope_index_qwen3_omni(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        tokens_per_second=None,
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs["image_grid_thw"],
        video_grid_thw=None,
        second_per_grid_ts=None,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
        use_audio_in_video=False,
        audio_seqlens=None,
    )
    assert torch.equal(positions.float(), oracle_pos.squeeze(1).float())
    assert torch.equal(delta.float(), oracle_delta.float())


# Part B: talker linearization equivalence


def _thinker_config_ns():
    from types import SimpleNamespace

    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=None),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
    )


def _decode_pos_from_delta(delta: torch.Tensor, seq_len: int) -> float:
    """(delta - 1) + seq_len (ForwardBatch._expand_mrope_from_input)."""
    return float(delta.reshape(-1)[0].item()) - 1.0 + float(seq_len)


def test_talker_mm_prompt_not_equivalent_to_linear() -> None:
    """Mm talker prompts diverge from arange+delta0 (prefill + decode)."""
    from sglang_omni.models.qwen3_omni.mrope_positions import (
        get_rope_index_qwen3_omni_vectorized,
        linear_mrope_positions,
        talker_can_use_linear_mrope,
    )

    grid = [1, 8, 8]
    user = [151644, 872] + _image_span(grid) + [198, 151645]
    assistant = [151675] * 20
    tokens = user + assistant
    ids = torch.tensor([tokens], dtype=torch.long)
    image_grid = torch.tensor([grid], dtype=torch.long)
    model_inputs = {"image_grid_thw": image_grid}

    mm_pos, mm_delta = get_rope_index_qwen3_omni_vectorized(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        tokens_per_second=None,
        input_ids=ids,
        image_grid_thw=image_grid,
        video_grid_thw=None,
        second_per_grid_ts=None,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
        use_audio_in_video=False,
        audio_seqlens=None,
    )
    lin_pos, lin_delta = linear_mrope_positions(len(tokens), dtype=torch.float)

    mm_pos_2d = mm_pos.squeeze(1).float()
    assert not torch.equal(mm_pos_2d, lin_pos.float())
    assert not torch.equal(mm_pos_2d[0], torch.arange(len(tokens), dtype=torch.float))
    assert not torch.equal(mm_delta.float(), lin_delta.float())

    seq_after_prefill = len(tokens) + 1
    assert _decode_pos_from_delta(
        mm_delta, seq_after_prefill
    ) != _decode_pos_from_delta(lin_delta, seq_after_prefill)

    cfg = _thinker_config_ns()
    assert talker_can_use_linear_mrope(ids.view(-1), model_inputs, cfg) is False


def test_talker_text_only_can_use_linear_and_matches_full_path() -> None:
    """No grids → linear short-circuit matches full path."""
    from sglang_omni.models.qwen3_omni.mrope_positions import (
        get_rope_index_qwen3_omni_vectorized,
        linear_mrope_positions,
        talker_can_use_linear_mrope,
    )

    tokens = list(range(100, 140))
    ids = torch.tensor([tokens], dtype=torch.long)
    model_inputs: dict = {}
    cfg = _thinker_config_ns()
    assert talker_can_use_linear_mrope(ids.view(-1), model_inputs, cfg) is True

    full_pos, full_delta = get_rope_index_qwen3_omni_vectorized(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        tokens_per_second=None,
        input_ids=ids,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
        use_audio_in_video=False,
        audio_seqlens=None,
    )
    lin_pos, lin_delta = linear_mrope_positions(len(tokens))
    assert torch.equal(full_pos.squeeze(1).long(), lin_pos)
    assert int(full_delta.reshape(-1)[0].item()) == 0
    assert torch.equal(lin_delta.view(-1), torch.zeros(1, dtype=torch.long))


def test_talker_grids_without_mm_markers_can_use_linear() -> None:
    """Grids without vision/audio markers → multimodal_nums=0 → linear."""
    from sglang_omni.models.qwen3_omni.mrope_positions import (
        get_rope_index_qwen3_omni_vectorized,
        linear_mrope_positions,
        talker_can_use_linear_mrope,
    )

    tokens = [1, 2, 3, 4, 5]
    ids = torch.tensor([tokens], dtype=torch.long)
    model_inputs = {"image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long)}
    cfg = _thinker_config_ns()
    assert talker_can_use_linear_mrope(ids.view(-1), model_inputs, cfg) is True

    full_pos, full_delta = get_rope_index_qwen3_omni_vectorized(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        tokens_per_second=None,
        input_ids=ids,
        image_grid_thw=model_inputs["image_grid_thw"],
        video_grid_thw=None,
        second_per_grid_ts=None,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
        use_audio_in_video=False,
        audio_seqlens=None,
    )
    lin_pos, _ = linear_mrope_positions(len(tokens), dtype=full_pos.dtype)
    assert torch.equal(full_pos.squeeze(1).float(), lin_pos.float())
    assert float(full_delta.reshape(-1)[0].item()) == 0.0


def test_build_talker_request_keeps_mm_mrope_for_image_prompt(monkeypatch) -> None:
    """Talker builder keeps mm MRoPE when image segments exist."""
    from unittest.mock import MagicMock

    from sglang_omni.models.qwen3_omni import request_builders as rb
    from sglang_omni.models.qwen3_omni.mrope_positions import (
        get_rope_index_qwen3_omni_vectorized,
    )

    grid = [1, 4, 4]
    tokens = [10] + _image_span(grid) + [11] + [151675] * 4
    input_ids = torch.tensor(tokens, dtype=torch.long)
    embeds = torch.zeros(len(tokens), 8)
    model_inputs = {"image_grid_thw": torch.tensor([grid], dtype=torch.long)}
    cfg = _thinker_config_ns()

    # Avoid importing heavy sglang Req machinery beyond MultimodalInputs.
    fake_req = MagicMock()
    fake_req.output_ids = []
    fake_sampling = MagicMock()
    monkeypatch.setattr(
        "sglang.srt.managers.schedule_batch.Req",
        lambda **kwargs: fake_req,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams",
        lambda **kwargs: fake_sampling,
    )
    fake_sampling.normalize = MagicMock()
    fake_sampling.verify = MagicMock()

    # MultimodalInputs is a real dataclass — keep it.
    data = rb.build_sglang_talker_request(
        thinker_hidden_states=torch.empty(0),
        tokenizer=MagicMock(),
        codec_vocab_size=3072,
        talker_input_embeds=embeds,
        talker_input_ids=input_ids,
        input_embeds_are_projected=True,
        thinker_config=cfg,
        talker_model_inputs=model_inputs,
    )
    mm = data.req.multimodal_inputs
    assert mm is not None
    expected_pos, expected_delta = get_rope_index_qwen3_omni_vectorized(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        tokens_per_second=None,
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs["image_grid_thw"],
        video_grid_thw=None,
        second_per_grid_ts=None,
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
        use_audio_in_video=False,
        audio_seqlens=None,
    )
    assert torch.equal(mm.mrope_positions.float(), expected_pos.squeeze(1).float())
    assert torch.equal(mm.mrope_position_delta.float(), expected_delta.float())
    assert not torch.equal(
        mm.mrope_positions[0].float(),
        torch.arange(len(tokens), dtype=torch.float),
    )


def test_build_talker_request_uses_linear_mrope_without_mm_markers(
    monkeypatch,
) -> None:
    """Grids but no mm markers → builder installs linear positions directly."""
    from unittest.mock import MagicMock

    from sglang_omni.models.qwen3_omni import request_builders as rb

    tokens = [1, 2, 3, 4, 5] + [151675] * 4
    input_ids = torch.tensor(tokens, dtype=torch.long)
    embeds = torch.zeros(len(tokens), 8)
    model_inputs = {"image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long)}
    cfg = _thinker_config_ns()

    fake_req = MagicMock()
    fake_req.output_ids = []
    fake_sampling = MagicMock()
    monkeypatch.setattr(
        "sglang.srt.managers.schedule_batch.Req",
        lambda **kwargs: fake_req,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams",
        lambda **kwargs: fake_sampling,
    )
    fake_sampling.normalize = MagicMock()
    fake_sampling.verify = MagicMock()

    def _fail_compute(*args, **kwargs):
        raise AssertionError("linear gate must skip _compute_mrope_positions")

    monkeypatch.setattr(rb, "_compute_mrope_positions", _fail_compute)

    data = rb.build_sglang_talker_request(
        thinker_hidden_states=torch.empty(0),
        tokenizer=MagicMock(),
        codec_vocab_size=3072,
        talker_input_embeds=embeds,
        talker_input_ids=input_ids,
        input_embeds_are_projected=True,
        thinker_config=cfg,
        talker_model_inputs=model_inputs,
    )
    mm = data.req.multimodal_inputs
    assert mm is not None
    assert torch.equal(
        mm.mrope_positions,
        torch.arange(len(tokens), dtype=torch.long).unsqueeze(0).expand(3, -1),
    )
    assert torch.equal(mm.mrope_position_delta, torch.zeros((1, 1), dtype=torch.long))
