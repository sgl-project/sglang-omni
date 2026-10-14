# SPDX-License-Identifier: Apache-2.0
"""Real request retirement/re-prefill contracts without an attention model."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from tests.unit_test.qwen3_omni.test_thinker_prefill_contract import (
    AUDIO_ID,
    HIDDEN,
    IMAGE_ID,
    VIDEO_ID,
    VOCAB,
    make_batch,
    make_runner,
)


def media_request(modality: str) -> tuple[Req, SGLangARRequestData]:
    token_id = {"audio": AUDIO_ID, "image": IMAGE_ID, "video": VIDEO_ID}[modality]
    inputs = {
        f"{modality}_embeds": torch.arange(2 * HIDDEN).reshape(2, HIDDEN).float() + 100
    }
    if modality != "audio":
        inputs[f"{modality}_deepstack_visual_embeds"] = [
            torch.full((2, HIDDEN), 200.0 + layer) + torch.arange(2).reshape(2, 1)
            for layer in range(3)
        ]
    req = Req(
        rid=f"retract-{modality}",
        origin_input_text="",
        origin_input_ids=[7, token_id, 8, token_id, 9],
        sampling_params=SamplingParams(max_new_tokens=17, temperature=0),
        vocab_size=VOCAB,
    )
    req.omni_model_inputs = inputs
    req._omni_consumed = None  # noqa: leading-underscore
    req._omni_mm_positions = None  # noqa: leading-underscore
    req.multimodal_inputs = MultimodalInputs(mm_items=[])
    req.multimodal_inputs.mrope_positions = torch.arange(5).repeat(3, 1)
    req.multimodal_inputs.mrope_position_delta = torch.tensor([0])
    data = SGLangARRequestData(req=req, model_inputs=inputs, output_ids=req.output_ids)
    req.omni_data = data
    return req, data


def prefill(runner, req: Req, tokens: list[int], prefix: int = 0):
    forward_batch, schedule_batch = make_batch([req], [tokens], prefix_lens=[prefix])
    captured = []

    def capture(_batch, embeds, deepstack, mask):
        captured.append((embeds.clone(), deepstack, mask))
        return SimpleNamespace(logits_output=None)

    runner.forward_with_omni_embeds = capture
    runner.before_prefill(forward_batch, schedule_batch, [req])
    runner.custom_prefill_forward(forward_batch, schedule_batch, [req])
    sidecar = get_omni_prefill_inputs(forward_batch)
    if sidecar is not None:
        return sidecar.input_embeds, None, None
    if captured:
        return captured[0]
    return forward_batch.input_embeds, None, None


def retire_for_recompute(req: Req) -> None:
    # The allocator is absent: the real native reset requires no held KV.
    assert not req.kv.holds_kv
    req.reset_for_retract()
    # Native prepare_for_extend clears this before the model's prefill hook.
    req.is_retracted = False


@pytest.mark.parametrize("modality", ["audio", "image", "video"])
@pytest.mark.parametrize("middle_chunk", [False, True])
@pytest.mark.parametrize("cached_prefix", [0, 2])
def test_retraction_replays_original_media_and_auxiliary_rows(
    modality: str, middle_chunk: bool, cached_prefix: int
) -> None:
    runner = make_runner()
    req, data = media_request(modality)
    original = data.model_inputs
    delta = req.multimodal_inputs.mrope_position_delta
    positions = req.multimodal_inputs.mrope_positions
    req.inflight_middle_chunks = int(middle_chunk)
    initial = list(req.origin_input_ids[:3] if middle_chunk else req.origin_input_ids)
    prefill(runner, req, initial)
    req.output_ids.extend([11, 12])
    output_ids = req.output_ids
    retire_for_recompute(req)
    replay_tokens = list(req.origin_input_ids) + list(req.output_ids)
    actual, auxiliary, mask = prefill(
        runner, req, replay_tokens[cached_prefix:], cached_prefix
    )
    media_positions = [
        index - cached_prefix for index in (1, 3) if index >= cached_prefix
    ]
    expected_rows = original[f"{modality}_embeds"][cached_prefix // 2 :]
    assert torch.equal(actual[media_positions], expected_rows)
    if modality != "audio":
        assert auxiliary is not None and len(auxiliary) == 3
        assert mask.nonzero(as_tuple=True)[0].tolist() == media_positions
        for layer, values in enumerate(auxiliary):
            assert torch.equal(
                values,
                original[f"{modality}_deepstack_visual_embeds"][layer][
                    cached_prefix // 2 :
                ],
            )
    assert req.output_ids is output_ids and list(output_ids) == [11, 12]
    assert req.multimodal_inputs.mrope_position_delta is delta
    assert req.multimodal_inputs.mrope_positions is positions
    assert data.model_inputs is original
    assert req.omni_model_inputs is None


@pytest.mark.parametrize("cached_prefix", [3, 5])
def test_retraction_replays_combined_visual_rows_after_cached_prefix(
    cached_prefix: int,
) -> None:
    runner = make_runner()
    req, data = media_request("image")
    req.origin_input_ids = [7, IMAGE_ID, VIDEO_ID, 8, IMAGE_ID, VIDEO_ID, 9]
    original = data.model_inputs
    original.pop("image_deepstack_visual_embeds")
    original["video_embeds"] = torch.full((2, HIDDEN), 150.0)
    original["deepstack_visual_embeds"] = [
        torch.arange(4).reshape(4, 1).expand(4, HIDDEN).float() + 200 + layer * 10
        for layer in range(3)
    ]
    prefill(runner, req, list(req.origin_input_ids))
    retire_for_recompute(req)
    actual, auxiliary, mask = prefill(
        runner, req, req.origin_input_ids[cached_prefix:], cached_prefix
    )
    row_start = 2 if cached_prefix == 3 else 3
    assert auxiliary is not None
    for layer, values in enumerate(auxiliary):
        assert torch.equal(
            values, original["deepstack_visual_embeds"][layer][row_start:]
        )
    assert int(mask.sum()) == 4 - row_start


def test_each_retraction_restores_once_without_resetting_middle_chunk_progress() -> (
    None
):
    runner = make_runner()
    req, data = media_request("video")
    original = data.model_inputs
    prefill(runner, req, list(req.origin_input_ids))
    for generation in (1, 2):
        retire_for_recompute(req)
        req.inflight_middle_chunks = 1
        first, _, _ = prefill(runner, req, list(req.origin_input_ids[:3]))
        assert torch.equal(first[1], original["video_embeds"][0])
        assert data.prefill_retraction_count == generation
        assert req._omni_consumed["video"] == 1  # noqa: leading-underscore
        req.inflight_middle_chunks = 0
        second, _, _ = prefill(runner, req, list(req.origin_input_ids[3:]), 3)
        assert torch.equal(second[0], original["video_embeds"][1])
        assert data.prefill_retraction_count == generation
        assert req.omni_model_inputs is None


def test_retraction_cannot_restore_another_requests_media() -> None:
    runner = make_runner()
    req, data = media_request("video")
    other, _ = media_request("image")
    retire_for_recompute(req)
    data.req = other
    with pytest.raises(RuntimeError, match="request-data owner"):
        prefill(runner, req, list(req.origin_input_ids))


def test_retraction_generation_cannot_move_backwards() -> None:
    runner = make_runner()
    req, data = media_request("audio")
    retire_for_recompute(req)
    data.prefill_retraction_count = 2
    with pytest.raises(RuntimeError, match="generation moved backwards"):
        prefill(runner, req, list(req.origin_input_ids))


def test_completed_nonretracted_tail_keeps_existing_text_route() -> None:
    runner = make_runner()
    req, data = media_request("video")
    prefill(runner, req, list(req.origin_input_ids))
    assert req.omni_model_inputs is None
    actual, auxiliary, _ = prefill(runner, req, [11], len(req.origin_input_ids))
    assert req.omni_model_inputs is None and data.prefill_retraction_count == 0
    assert torch.equal(actual, runner.embed_tokens(torch.tensor([11])))
    assert auxiliary is None
