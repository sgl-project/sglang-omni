# SPDX-License-Identifier: Apache-2.0
"""A Qwen session prefill takes the M-RoPE positions of the whole conversation computed at once, over retained audio and image rows."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

from sglang_omni.models.qwen3_omni import thinker_session
from sglang_omni.models.qwen3_omni.mrope_positions import feat_extract_output_lengths
from sglang_omni.models.qwen3_omni.request_builders import compute_mrope_positions
from sglang_omni.models.qwen3_omni.thinker_session import (
    CODEC_FRAME_MS,
    QwenThinkerSessionAdapter,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.qwen3_omni.test_thinker_session import (
    END,
    IDENTITY,
    NEWLINE,
    PROMPT_TEXT,
    SYSTEM_PREFIX,
    abort,
    truncate,
    unit_chunk,
)

IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
AUDIO_TOKEN_ID = 151646
AUDIO_START_TOKEN_ID = 151647
AUDIO_END_TOKEN_ID = 151648
USER = 7
ASSISTANT = 8
IMAGE_GRID = [1, 4, 4]
HIDDEN = 4


def thinker_config() -> Qwen3OmniMoeThinkerConfig:
    return Qwen3OmniMoeThinkerConfig(
        audio_token_id=AUDIO_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        position_id_per_seconds=25,
        vision_config={"spatial_merge_size": 2, "tokens_per_second": 25},
    )


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> QwenThinkerSessionAdapter:
    tokenizer = Mock()
    tokenizer.convert_tokens_to_ids.return_value = END
    tokenizer.encode.side_effect = lambda text, add_special_tokens: (
        [NEWLINE] if text == "\n" else SYSTEM_PREFIX
    )

    def build_request(
        pipeline_state,
        *,
        params,
        tokenizer,
        vocab_size,
        request_id,
        capture_prompt_hidden_states,
    ):
        return SimpleNamespace(
            req=SimpleNamespace(
                origin_input_ids=pipeline_state.prompt["input_ids"].tolist()
            ),
            output_ids=[],
            stage_payload=None,
            finish_reason=None,
        )

    monkeypatch.setattr(thinker_session, "build_sglang_thinker_request", build_request)
    result = QwenThinkerSessionAdapter(tokenizer, 200, thinker_config())
    result.open(IDENTITY, OmniRequest(None))
    return result


def audio_span(feature_length: int) -> list[int]:
    rows = feat_extract_output_lengths(feature_length)
    return [AUDIO_START_TOKEN_ID, *[AUDIO_TOKEN_ID] * rows, AUDIO_END_TOKEN_ID]


def image_span() -> list[int]:
    t, h, w = IMAGE_GRID
    return [
        VISION_START_TOKEN_ID,
        *[IMAGE_TOKEN_ID] * (t * h * w // 4),
        VISION_END_TOKEN_ID,
    ]


def turn_payload(
    turn: int, feature_length: int, with_image: bool = False
) -> StagePayload:
    """One encoded user turn as the audio encoder hands it on: the prompt ids and this turn's media rows."""
    prompt_ids = [USER]
    model_inputs: dict[str, object] = {
        "audio_embeds": torch.full(
            (feat_extract_output_lengths(feature_length), HIDDEN), float(turn)
        ),
        "audio_feature_lengths": torch.tensor([feature_length]),
        "feature_attention_mask": torch.ones(1, feature_length, dtype=torch.long),
    }
    if with_image:
        rows = len(image_span()) - 2
        prompt_ids.extend(image_span())
        model_inputs["image_embeds"] = torch.full((rows, HIDDEN), -float(turn))
        model_inputs["image_grid_thw"] = torch.tensor([IMAGE_GRID])
        model_inputs["deepstack_visual_embeds"] = [
            torch.full((rows, HIDDEN), float(layer)) for layer in range(3)
        ]
    else:
        pass
    prompt_ids.extend([*audio_span(feature_length), ASSISTANT])
    return StagePayload(
        f"unit-{turn}",
        OmniRequest(None, params={"max_new_tokens": 8}),
        {
            "session": {"turn": turn},
            "prompt": {
                "input_ids": torch.tensor(SYSTEM_PREFIX + prompt_ids),
                "prompt_text": PROMPT_TEXT,
            },
            "thinker_inputs": {"model_inputs": model_inputs},
        },
    )


class NativeSession:
    """What the engine's streaming session holds: the ids of every committed request, which it prepends to the next unit's ids."""

    def __init__(self) -> None:
        self.ids: list[int] = []

    def bind(self, adapter: QwenThinkerSessionAdapter, request_data) -> SimpleNamespace:
        request = SimpleNamespace(
            origin_input_ids=[*self.ids, *request_data.req.origin_input_ids]
        )
        request_data.req = request
        adapter.bind_request(IDENTITY, request_data, SimpleNamespace())
        return request

    def finish(self, adapter: QwenThinkerSessionAdapter, request_data, output_ids):
        self.ids = [*request_data.req.origin_input_ids, *output_ids]
        request_data.output_ids = output_ids
        adapter.result(IDENTITY, request_data)


def one_shot(
    ids: list[int], feature_lengths: list[int], image_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    model_inputs: dict[str, object] = {
        "audio_feature_lengths": torch.tensor(feature_lengths),
    }
    if image_count:
        model_inputs["image_grid_thw"] = torch.tensor([IMAGE_GRID] * image_count)
    else:
        pass
    return compute_mrope_positions(
        torch.tensor(ids, dtype=torch.long), model_inputs, thinker_config()
    )


def assert_positions_match(request, feature_lengths: list[int], image_count: int):
    positions, delta = one_shot(request.origin_input_ids, feature_lengths, image_count)
    assert torch.equal(request.multimodal_inputs.mrope_positions, positions)
    assert torch.equal(request.multimodal_inputs.mrope_position_delta, delta)


def test_every_prefill_takes_the_positions_of_the_whole_conversation(
    adapter: QwenThinkerSessionAdapter,
) -> None:
    """Turns of audio, audio with an image, an interrupted turn refilled with what was heard, and a rebuild after a truncate."""
    native = NativeSession()
    lengths = [4000, 2400, 8000, 1600, 3200]

    first = adapter.build(IDENTITY, unit_chunk(1), turn_payload(1, lengths[0]))
    request = native.bind(adapter, first)
    assert request.origin_input_ids[: len(SYSTEM_PREFIX)] == SYSTEM_PREFIX
    assert_positions_match(request, lengths[:1], 0)
    native.finish(adapter, first, [51, 52, END])

    second = adapter.build(
        IDENTITY, unit_chunk(2), turn_payload(2, lengths[1], with_image=True)
    )
    request = native.bind(adapter, second)
    assert request.origin_input_ids[: len(native.ids)] == native.ids
    assert_positions_match(request, lengths[:2], 1)
    assert request.omni_model_inputs["image_grid_thw"].tolist() == [IMAGE_GRID]
    assert len(request.omni_model_inputs["image_deepstack_visual_embeds"]) == 3
    assert "feature_attention_mask" not in request.omni_model_inputs
    native.finish(adapter, second, [61, END])

    third = adapter.build(IDENTITY, unit_chunk(3), turn_payload(3, lengths[2]))
    native.bind(adapter, third)
    abort(adapter, [71, 72, 73, 74, 75, 76])
    adapter.control(IDENTITY, truncate(3, t_ms=2 * CODEC_FRAME_MS))

    fourth = adapter.build(IDENTITY, unit_chunk(4), turn_payload(4, lengths[3]))
    request = native.bind(adapter, fourth)
    heard = [71, 72, 73, 74, 75]
    assert request.origin_input_ids[len(native.ids) :] == [
        NEWLINE,
        USER,
        *audio_span(lengths[2]),
        ASSISTANT,
        *heard,
        END,
        NEWLINE,
        USER,
        *audio_span(lengths[3]),
        ASSISTANT,
    ]
    assert_positions_match(request, lengths[:4], 1)
    audio_rows = request.omni_model_inputs["audio_embeds"]
    assert audio_rows.shape[0] == sum(
        feat_extract_output_lengths(length) for length in lengths[:4]
    )
    assert audio_rows[-1].tolist() == [4.0] * HIDDEN
    native.finish(adapter, fourth, [81, 82, 83, 84, END])

    adapter.control(IDENTITY, truncate(4, t_ms=0))
    preparation = adapter.prepare_unit(
        IDENTITY, unit_chunk(5), turn_payload(5, lengths[4])
    )
    assert preparation.reset_history is True
    native.ids = []
    fifth = adapter.build(IDENTITY, unit_chunk(5), turn_payload(5, lengths[4]))
    request = native.bind(adapter, fifth)
    assert request.origin_input_ids[: len(SYSTEM_PREFIX)] == SYSTEM_PREFIX
    assert request.origin_input_ids.count(AUDIO_START_TOKEN_ID) == 5
    assert request.origin_input_ids.count(VISION_START_TOKEN_ID) == 1
    assert_positions_match(request, lengths, 1)
    assert request.omni_model_inputs["audio_embeds"].shape[0] == sum(
        feat_extract_output_lengths(length) for length in lengths
    )
