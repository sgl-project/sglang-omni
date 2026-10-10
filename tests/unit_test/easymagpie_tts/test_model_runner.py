# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.easymagpie_tts.decode_state import EMIT_COLUMN, STOP_COLUMN
from sglang_omni.models.easymagpie_tts.model_runner import EasyMagpieTTSModelRunner
from sglang_omni.models.easymagpie_tts.payload_types import EasyMagpieTTSState
from sglang_omni.models.easymagpie_tts.request_builders import (
    EasyMagpieSGLangRequestData,
)


@pytest.fixture
def runner(talker) -> EasyMagpieTTSModelRunner:
    runner = EasyMagpieTTSModelRunner.__new__(EasyMagpieTTSModelRunner)
    runner.model = talker
    return runner


def make_request(
    *, prompt_rows: int = 9, slot: int = 1, voice: str = "eng"
) -> SimpleNamespace:
    state = EasyMagpieTTSState(
        text_token_ids=[11, 12, 13, 14, 15, 63],
        context_token_ids=[7, 8],
        voice=voice,
        speaker_frames=3,
        phoneme_delay=3,
        speech_delay=5,
        text_prefill_num=4,
    )
    data = EasyMagpieSGLangRequestData(
        state=state,
        sampling_seed=7,
        req=SimpleNamespace(
            prefix_indices=[],
            extend_range=SimpleNamespace(length=prompt_rows),
            kv=SimpleNamespace(req_pool_idx=slot),
            finished=lambda: False,
            is_retracted=False,
        ),
    )
    return SimpleNamespace(data=data)


def test_prefill_folds_speaker_context_and_text_lead_in(runner, talker) -> None:
    heads = talker.heads
    request = make_request()
    batch = SimpleNamespace(
        input_ids=torch.zeros(9, dtype=torch.long), replace_embeds=None
    )

    assert runner.custom_prefill_forward(batch, None, [request]) is None

    sidecar = get_omni_prefill_inputs(batch)
    assert sidecar.input_embeds_are_projected is True
    embeds = sidecar.input_embeds
    text = heads.text_embedding.weight.detach()
    bos = heads.embed_phonemes(torch.tensor([[17]])).detach()[0]
    assert embeds.shape == (9, 8)
    torch.testing.assert_close(embeds[:3], torch.ones((3, 8)))
    torch.testing.assert_close(embeds[3:5], text[[7, 8]])
    torch.testing.assert_close(embeds[5:8], text[[11, 12, 13]])
    torch.testing.assert_close(embeds[8], text[14] + bos)


def test_batched_prefill_packs_each_requests_own_voice(runner, talker) -> None:
    text = talker.heads.text_embedding.weight.detach()
    bos = talker.heads.embed_phonemes(torch.tensor([[17]])).detach()[0]
    eng, alt = make_request(), make_request(prompt_rows=8, voice="alt")
    batch = SimpleNamespace(input_ids=torch.zeros(17), replace_embeds=None)
    runner.custom_prefill_forward(batch, None, [eng, alt])

    embeds = get_omni_prefill_inputs(batch).input_embeds
    assert embeds.shape == (17, 8)
    torch.testing.assert_close(embeds[:3], torch.ones((3, 8)))
    torch.testing.assert_close(embeds[9:11], torch.full((2, 8), 2.0))
    torch.testing.assert_close(embeds[11:13], text[[7, 8]])
    torch.testing.assert_close(embeds[13:16], text[[11, 12, 13]])
    torch.testing.assert_close(embeds[16], text[14] + bos)


def test_prefill_slices_rows_already_held_as_prefix(runner, talker) -> None:
    text = talker.heads.text_embedding.weight.detach()
    bos = talker.heads.embed_phonemes(torch.tensor([[17]])).detach()[0]
    request = make_request(prompt_rows=4)
    request.data.req.prefix_indices = [0, 1, 2, 3, 4]
    batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long), replace_embeds=None
    )
    runner.custom_prefill_forward(batch, None, [request])
    embeds = get_omni_prefill_inputs(batch).input_embeds
    assert embeds.shape == (4, 8)
    torch.testing.assert_close(embeds[:3], text[[11, 12, 13]])
    torch.testing.assert_close(embeds[3], text[14] + bos)

    request.data.req.extend_range.length = 6
    with pytest.raises(RuntimeError, match="scheduler expects 6"):
        runner.custom_prefill_forward(
            SimpleNamespace(input_ids=torch.zeros(6), replace_embeds=None),
            None,
            [request],
        )


def test_post_prefill_seeds_each_requests_slot(runner, talker) -> None:
    talker.last_phoneme_tokens = torch.tensor([[9], [18]])
    first, second = make_request(slot=3), make_request(slot=1)
    second.data.sampling_seed = 22
    runner.post_prefill(None, None, None, [first, second])

    inputs = talker.decode_state.read(torch.tensor([1, 3]))
    assert inputs.phoneme_tokens.tolist() == [[18], [9]]
    assert inputs.phoneme_ended.tolist() == [True, False]
    assert inputs.seeds.tolist() == [22, 7]
    assert inputs.steps.tolist() == [4, 4]


def stage_step(talker, codes: list[list[int]], emits: list[int], stops: list[int]):
    output = talker.decode_state.step_output
    rows = len(codes)
    output[:rows, :4] = torch.tensor(codes)
    output[:rows, EMIT_COLUMN] = torch.tensor(emits)
    output[:rows, STOP_COLUMN] = torch.tensor(stops)


def test_post_decode_collects_audio_frames_and_stop_tokens(runner, talker) -> None:
    speaking, stopping, warming = make_request(), make_request(), make_request()
    stage_step(talker, [[0, 1, 2, 3], [17] * 4, [4] * 4], [1, 0, 0], [0, 1, 0])
    result = SimpleNamespace(next_token_ids=None)

    runner.post_decode(result, None, None, [speaking, stopping, warming])

    assert result.next_token_ids.tolist() == [0, 1, 0]
    assert [codes.tolist() for codes in speaking.data.output_codes] == [[0, 1, 2, 3]]
    assert stopping.data.output_codes == []
    assert warming.data.output_codes == []


def test_async_resolve_reads_the_launched_step(runner, talker) -> None:
    runner.next_host_staging = lambda shape, dtype: torch.empty(shape, dtype=dtype)
    request = make_request()
    stage_step(talker, [[1, 1, 1, 1]], [1], [0])
    result = SimpleNamespace(next_token_ids=None)

    host = runner.post_decode_launch(result, None, [request])
    # The next launch overwrites the device output before this step resolves.
    stage_step(talker, [[2, 2, 2, 2]], [1], [1])
    runner.post_decode_resolve(host, result, None, None, [request])

    assert result.next_token_ids.tolist() == [0]
    assert [codes.tolist() for codes in request.data.output_codes] == [[1] * 4]
    assert runner.post_decode_launch(result, None, []) is None


def test_async_resolve_drops_requests_finished_a_step_earlier(runner, talker) -> None:
    runner.next_host_staging = lambda shape, dtype: torch.empty(shape, dtype=dtype)
    finished, live = make_request(), make_request()
    finished.data.req.finished = lambda: True
    stage_step(talker, [[1] * 4, [2] * 4], [1, 1], [0, 0])
    result = SimpleNamespace(next_token_ids=None)

    host = runner.post_decode_launch(result, None, [finished, live])
    runner.post_decode_resolve(host, result, None, None, [finished, live])

    assert finished.data.output_codes == []
    assert [codes.tolist() for codes in live.data.output_codes] == [[2] * 4]
