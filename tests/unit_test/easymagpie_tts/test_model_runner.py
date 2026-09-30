# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
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


def make_request(step: int, prompt_rows: int = 9) -> SimpleNamespace:
    state = EasyMagpieTTSState(
        text_token_ids=[11, 12, 13, 14, 15, 63],
        context_token_ids=[7, 8],
        speaker_embedding=torch.ones((3, 8)),
        phoneme_delay=3,
        speech_delay=5,
        text_prefill_num=4,
    )
    data = EasyMagpieSGLangRequestData(
        state=state,
        decode_offset=step,
        sampling_seed=7,
        req=SimpleNamespace(
            prefix_indices=[], extend_range=SimpleNamespace(length=prompt_rows)
        ),
    )
    return SimpleNamespace(data=data)


def decode_batch(rows: int) -> SimpleNamespace:
    return SimpleNamespace(input_ids=torch.zeros(rows, dtype=torch.long))


def test_prefill_folds_speaker_context_and_text_lead_in(runner, talker) -> None:
    heads = talker.heads
    request = make_request(step=4)
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


def test_prefill_slices_rows_already_held_as_prefix(runner) -> None:
    request = make_request(step=4, prompt_rows=4)
    request.data.req.prefix_indices = [0, 1, 2, 3, 4]
    batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long), replace_embeds=None
    )
    runner.custom_prefill_forward(batch, None, [request])
    assert get_omni_prefill_inputs(batch).input_embeds.shape == (4, 8)

    request.data.req.extend_range.length = 6
    with pytest.raises(RuntimeError, match="scheduler expects 6"):
        runner.custom_prefill_forward(
            SimpleNamespace(input_ids=torch.zeros(6), replace_embeds=None),
            None,
            [request],
        )


def test_post_prefill_keeps_each_rows_phoneme_feedback(runner, talker) -> None:
    talker.last_phoneme_tokens = torch.tensor([[9], [18]])
    first, second = make_request(4), make_request(4)
    runner.post_prefill(None, None, None, [first, second])
    assert first.data.last_phoneme_tokens.tolist() == [9]
    assert (first.data.last_phoneme_is_eos, second.data.last_phoneme_is_eos) == (
        False,
        True,
    )


def spy_conditioning(talker) -> list[dict]:
    calls = []
    compose = talker.compose_conditioning

    def spy(**kwargs):
        calls.append(kwargs)
        return compose(**kwargs)

    talker.compose_conditioning = spy
    return calls


def test_decode_applies_phoneme_and_speech_delays(runner, talker) -> None:
    calls = spy_conditioning(talker)

    runner.before_decode(decode_batch(1), None, [make_request(2)])
    assert calls[-1]["phoneme_valid"].tolist() == [False]
    assert calls[-1]["audio_valid"].tolist() == [False]

    runner.before_decode(decode_batch(1), None, [make_request(3)])
    assert calls[-1]["phoneme_tokens"].tolist() == [[17]]
    assert calls[-1]["audio_valid"].tolist() == [False]

    request = make_request(5)
    request.data.last_audio_codes = torch.full((4,), 3)
    runner.before_decode(decode_batch(1), None, [request])
    assert calls[-1]["previous_audio_codes"].tolist() == [[16] * 4]
    assert calls[-1]["text_tokens"].tolist() == [63]
    assert talker.decode_step.audio_valid.tolist() == [True]

    runner.before_decode(decode_batch(1), None, [request])
    assert calls[-1]["previous_audio_codes"].tolist() == [[3] * 4]
    assert calls[-1]["text_valid"].tolist() == [False]


def test_decode_feeds_a_phoneme_eos_once_then_closes_the_channel(
    runner, talker
) -> None:
    calls = spy_conditioning(talker)
    request = make_request(4)
    request.data.last_phoneme_tokens = torch.tensor([18])
    request.data.last_phoneme_is_eos = True

    runner.before_decode(decode_batch(1), None, [request])
    assert calls[-1]["phoneme_tokens"].tolist() == [[18]]
    assert request.data.phoneme_ended is True

    runner.before_decode(decode_batch(1), None, [request])
    assert calls[-1]["phoneme_valid"].tolist() == [False]


def test_decode_sampling_controls_stay_request_local(runner, talker) -> None:
    first, second = make_request(5), make_request(7)
    first.data.state.temperature, first.data.state.top_k = 0.5, 3
    second.data.state.temperature, second.data.state.top_k = 0.9, 6
    second.data.sampling_seed = 22
    batch = decode_batch(2)

    runner.before_decode(batch, None, [first, second])

    step = talker.decode_step
    assert batch.input_embeds.shape == (2, 8)
    assert step.temperatures.tolist() == pytest.approx([0.5, 0.9])
    assert (step.top_ks.tolist(), step.max_top_k) == ([3, 6], 6)
    assert step.seeds.tolist() == [7, 22]
    assert step.positions.tolist() == [6 * 4, 8 * 4]
    assert (first.data.decode_offset, second.data.decode_offset) == (6, 8)


def test_post_decode_emits_audio_frames_but_not_warmup_or_eos(runner, talker) -> None:
    talker.last_audio_codes = torch.stack((torch.arange(4), torch.full((4,), 17)))
    talker.last_phoneme_tokens = torch.tensor([[9], [18]])
    talker.last_audio_eos = torch.tensor([False, True])
    speaking, stopping = make_request(6), make_request(6)
    warming = make_request(5)

    runner.post_decode(None, None, None, [speaking, stopping])
    talker.last_audio_eos = torch.tensor([False])
    runner.post_decode(None, None, None, [warming])

    assert [codes.tolist() for codes in speaking.data.output_codes] == [[0, 1, 2, 3]]
    assert stopping.data.output_codes == []
    assert warming.data.output_codes == []
    assert stopping.data.last_phoneme_is_eos is True
    assert speaking.data.last_audio_codes.tolist() == [0, 1, 2, 3]
