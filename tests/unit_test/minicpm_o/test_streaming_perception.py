# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for native duplex image and audio unit plans."""

from io import BytesIO
from unittest.mock import Mock

import pytest
import torch
from PIL import Image, UnidentifiedImageError

from sglang_omni.models.minicpm_o.components.streaming_perception import (
    MiniCPMOPerceptionState,
)


@pytest.fixture
def state() -> MiniCPMOPerceptionState:
    tokenizer = Mock(unk_token_id=0)
    tokenizer.convert_tokens_to_ids.side_effect = {
        "<unit>": 1,
        "<image>": 2,
        "</image>": 3,
        "<slice>": 4,
        "</slice>": 5,
    }.__getitem__
    return MiniCPMOPerceptionState(
        tokenizer=tokenizer,
        processor=Mock(),
        audio_encoder=Mock(),
        image_encoder=Mock(),
        max_slice_nums=1,
    )


@pytest.mark.parametrize(
    ("chunk_index", "reference_audio"), [(1, False), (1, True), (2, True)]
)
def test_audio_plan_without_image(
    state: MiniCPMOPerceptionState, chunk_index: int, reference_audio: bool
) -> None:
    state.audio_chunk_index = chunk_index
    if reference_audio:
        state.prefix_token_ids = [7, 0, 0, 8]
        state.prefix_schema = [("token", 1), ("audio", 2), ("token", 1)]
        state.prefix_embeds = torch.full((2, 4), 5.0)
        prefix_spans = [
            dict(
                modality="audio",
                token_start=1,
                token_end=3,
                embedding_start=0,
                embedding_end=2,
            )
        ]
    else:
        state.prefix_token_ids = [7, 8]
        state.prefix_schema = [("token", 2)]
        prefix_spans = []
    audio = torch.full((10, 4), 9.0)
    plan = state.build_step_plan(audio)
    is_first_unit = chunk_index == 1
    prefix_ids = state.prefix_token_ids if is_first_unit else []
    spans = prefix_spans if is_first_unit else []
    embedding_start = 2 if is_first_unit and reference_audio else 0
    token_start = len(prefix_ids) + 1
    assert plan["token_ids"] == prefix_ids + [1] + [0] * 10
    assert plan["embedding_spans"] == spans + [
        dict(
            modality="audio",
            token_start=token_start,
            token_end=token_start + 10,
            embedding_start=embedding_start,
            embedding_end=embedding_start + 10,
        )
    ]
    assert plan["input_embeds"].shape[0] == embedding_start + 10
    assert torch.equal(plan["input_embeds"][embedding_start:], audio)


@pytest.mark.parametrize("chunk_index", [1, 2])
def test_image_plan_follows_official_slice_order(
    state: MiniCPMOPerceptionState, chunk_index: int
) -> None:
    state.audio_chunk_index = chunk_index
    state.prefix_token_ids = [7, 0, 0, 8]
    state.prefix_schema = [("token", 1), ("audio", 2), ("token", 1)]
    state.prefix_embeds = torch.full((2, 4), 5.0)
    first = torch.cat([torch.full((64, 4), float(i)) for i in (1, 2, 3)])
    second = torch.full((64, 4), 4.0)
    audio = torch.full((10, 4), 9.0)
    plan = state.build_step_plan(audio, (first, second))
    prefix = state.prefix_token_ids if chunk_index == 1 else []
    offset = len(prefix)
    # note (Junnan Li): Overview, its two slices, then the second frame's overview.
    expected = prefix + [1]
    for open_id, close_id in [(2, 3), (4, 5), (4, 5), (2, 3)]:
        expected += [open_id, *[0] * 64, close_id]
    assert plan["token_ids"] == expected + [0] * 10
    unit_spans = plan["embedding_spans"][-5:]
    assert [span["modality"] for span in unit_spans] == ["image"] * 4 + ["audio"]
    assert [(span["token_start"], span["token_end"]) for span in unit_spans] == [
        (offset + 2 + 66 * index, offset + 66 + 66 * index) for index in range(4)
    ] + [(offset + 265, offset + 275)]
    blocks = [state.prefix_embeds] if prefix else []
    blocks += [*first.split(64), second, audio]
    assert torch.equal(plan["input_embeds"], torch.cat(blocks))
    for span, block in zip(plan["embedding_spans"], blocks, strict=True):
        assert torch.equal(
            plan["input_embeds"][span["embedding_start"] : span["embedding_end"]], block
        )


def encoded_image(format: str) -> bytes:
    encoded = BytesIO()
    Image.new("RGB", (16, 16)).save(encoded, format=format)
    return encoded.getvalue()


@pytest.mark.parametrize(
    ("encoded", "error", "match", "pixel_limit"),
    [
        (encoded_image("GIF"), ValueError, "JPEG or PNG", None),
        (b"\x89PNG\r\n\x1a\ninvalid", UnidentifiedImageError, None, None),
        (encoded_image("PNG")[:45], OSError, None, None),
        (encoded_image("PNG"), ValueError, "pixel limit", 15),
    ],
)
def test_reject_frame_before_processor(
    state: MiniCPMOPerceptionState,
    monkeypatch: pytest.MonkeyPatch,
    encoded: bytes,
    error: type[Exception],
    match: str | None,
    pixel_limit: int | None,
) -> None:
    if pixel_limit is not None:
        monkeypatch.setattr(
            "sglang_omni.models.minicpm_o.components.streaming_perception.MAX_FRAME_PIXELS",
            pixel_limit,
        )
    else:
        pass
    with pytest.raises(error, match=match):
        state.encode_image(encoded)
    state.processor.process_image.assert_not_called()
