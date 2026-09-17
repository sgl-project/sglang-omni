# SPDX-License-Identifier: Apache-2.0
"""Prefill and decode hooks feed timeline rows to the model and frames to the codec."""

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.personaplex.architecture import (
    AGENT_STREAM_OFFSET,
    NUM_STREAMS,
    USER_STREAM_OFFSET,
)
from sglang_omni.models.personaplex.model_runner import PersonaPlexModelRunner
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.request_builders import build_lm_request
from sglang_omni.models.personaplex.timeline import output_frame
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

VOICE_FRAMES = 4


class _Depformer:
    """Returns base + step codes, keeping forced ones, and records every call."""

    spec = SimpleNamespace(steps=8)

    def __init__(self):
        self.calls = []

    def generate(self, text_token_B, transformer_out_BD, forced_BK, sample):
        base = 1000 + 10 * len(self.calls)
        codes = torch.where(forced_BK >= 0, forced_BK, base + torch.arange(8))
        self.calls.append(
            SimpleNamespace(
                text=text_token_B.clone(),
                hidden=transformer_out_BD.clone(),
                forced=forced_BK.clone(),
                codes=codes[0].clone(),
            )
        )
        return codes


class _Model:
    """Embeds a row as its own token ids, so fused inputs can be read back."""

    def __init__(self, max_batch: int = 2):
        self._fusion_buffer = torch.zeros(max_batch, NUM_STREAMS)
        self._hidden_out = torch.arange(max_batch * NUM_STREAMS, dtype=torch.float32)
        self._hidden_out = self._hidden_out.view(max_batch, NUM_STREAMS)
        self.depformer = _Depformer()

    def embed_rows(self, rows_NK: torch.Tensor) -> torch.Tensor:
        return rows_NK.to(torch.float32)


def _runner(model: _Model) -> PersonaPlexModelRunner:
    runner = PersonaPlexModelRunner.__new__(PersonaPlexModelRunner)
    runner.model = model
    return runner


def _request(num_frames: int, *, voice: bool = False, params=None):
    state = PersonaPlexState(
        text_prompt_ids=[11, 12, 13],
        user_codes=torch.arange(num_frames * 8).view(num_frames, 8) + 500,
    )
    if voice:
        state.voice_frames = VOICE_FRAMES
        state.voice_embeddings = torch.randn(VOICE_FRAMES - 1, NUM_STREAMS)
        state.voice_tail_codes = torch.arange(16).view(2, 8) + 300
    payload = StagePayload(
        "r", request=OmniRequest(inputs={}, params=params or {}), data=state.to_dict()
    )
    return SimpleNamespace(data=build_lm_request(payload, vocab_size=32000))


def test_prefill_uses_stored_voice_rows_and_embeds_the_rest():
    runner = _runner(_Model())
    with_voice, without_voice = _request(3, voice=True), _request(2)
    voice_timeline = with_voice.data.talker_model_inputs["timeline"]
    plain_timeline = without_voice.data.talker_model_inputs["timeline"]
    total = voice_timeline.num_prompt_positions + plain_timeline.num_prompt_positions
    forward_batch = SimpleNamespace(replace_embeds=None, input_ids=torch.zeros(total))

    fresh = SimpleNamespace(output_ids=[])
    runner.before_prefill(
        forward_batch,
        SimpleNamespace(reqs=[fresh, fresh]),
        [with_voice, without_voice],
    )

    inputs = get_omni_prefill_inputs(forward_batch)
    assert inputs.input_embeds_are_projected
    voice_rows = inputs.input_embeds[: voice_timeline.num_prompt_positions]
    plain_rows = inputs.input_embeds[voice_timeline.num_prompt_positions :]
    stored = VOICE_FRAMES - 1
    torch.testing.assert_close(voice_rows[:stored], voice_timeline.prefill_embeddings)
    assert torch.equal(
        voice_rows[stored:], voice_timeline.prefill_tokens[stored:].float()
    )
    assert torch.equal(plain_rows, plain_timeline.prefill_tokens.float())


def test_decode_rows_chain_text_agent_codes_and_caller_frames():
    model = _Model()
    runner = _runner(model)
    request = _request(3, voice=True)
    data = request.data
    timeline = data.talker_model_inputs["timeline"]
    first_position = timeline.num_prompt_positions

    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor([77])), None, None, [request]
    )
    prefill_call = model.depformer.calls[0]
    assert prefill_call.text.tolist() == [77]
    assert torch.equal(prefill_call.hidden, model._hidden_out[0:1])
    assert torch.equal(prefill_call.forced[0], timeline.forced_agent_at_start)
    assert torch.equal(
        data.talker_model_inputs["frames"][0],
        output_frame(timeline.agent_row_before_start, prefill_call.codes),
    )

    runner.before_decode(
        None, SimpleNamespace(reqs=[SimpleNamespace(output_ids=[77])]), [request]
    )
    row = model._fusion_buffer[0].long()
    assert row[0].item() == 77
    assert torch.equal(row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET], prefill_call.codes)
    assert torch.equal(row[USER_STREAM_OFFSET:], timeline.user_rows[first_position])

    runner.post_decode(
        SimpleNamespace(next_token_ids=torch.tensor([78])), None, None, [request]
    )
    decode_call = model.depformer.calls[1]
    assert decode_call.text.tolist() == [78]
    assert (decode_call.forced == -1).all()
    frames = data.talker_model_inputs["frames"]
    assert torch.equal(frames[1], output_frame(prefill_call.codes, decode_call.codes))
    assert len(data.talker_model_inputs["pending_frames"]) == 2

    runner.before_decode(
        None, SimpleNamespace(reqs=[SimpleNamespace(output_ids=[77, 78])]), [request]
    )
    row = model._fusion_buffer[0].long()
    assert row[0].item() == 78
    assert torch.equal(row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET], decode_call.codes)
    assert torch.equal(row[USER_STREAM_OFFSET:], timeline.user_rows[first_position + 1])


def test_seeded_audio_sampler_draws_reproducibly_from_one_generator():
    runner = _runner(_Model())
    logits = torch.randn(1, 64)

    def draws(request):
        sampler = runner._audio_sampler(request.data)
        return [int(sampler(logits)) for _ in range(5)]

    params = {"seed": 7, "audio_temperature": 1.0, "audio_top_k": 0}
    first, second = _request(1, params=params), _request(1, params=params)
    assert draws(first) == draws(second)
    generator = first.data.talker_model_inputs["audio_generator"]
    runner._audio_sampler(first.data)
    assert first.data.talker_model_inputs["audio_generator"] is generator

    unseeded = _request(1, params={"audio_temperature": 1.0})
    runner._audio_sampler(unseeded.data)
    assert "audio_generator" not in unseeded.data.talker_model_inputs


def test_resume_after_a_retract_replays_the_generated_positions():
    model = _Model()
    runner = _runner(model)
    request = _request(5)
    data = request.data
    timeline = data.talker_model_inputs["timeline"]
    prompt = timeline.num_prompt_positions

    runner.before_prefill(
        SimpleNamespace(replace_embeds=None, input_ids=torch.zeros(prompt)),
        SimpleNamespace(reqs=[SimpleNamespace(output_ids=[])]),
        [request],
    )
    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor([77])), None, None, [request]
    )
    for token, before in ((78, [77]), (79, [77, 78])):
        runner.before_decode(
            None, SimpleNamespace(reqs=[SimpleNamespace(output_ids=before)]), [request]
        )
        runner.post_decode(
            SimpleNamespace(next_token_ids=torch.tensor([token])), None, None, [request]
        )

    generated = [77, 78, 79]
    agent_rows = list(data.talker_model_inputs["agent_rows"])
    frames_before = len(data.talker_model_inputs["frames"])
    assert len(agent_rows) == len(generated)

    forward_batch = SimpleNamespace(
        replace_embeds=None, input_ids=torch.zeros(prompt + len(generated))
    )
    runner.before_prefill(
        forward_batch,
        SimpleNamespace(reqs=[SimpleNamespace(output_ids=generated)]),
        [request],
    )

    embeds = get_omni_prefill_inputs(forward_batch).input_embeds
    assert embeds.shape[0] == prompt + len(generated)
    assert torch.equal(embeds[:prompt], timeline.prefill_tokens.float())
    for index, token in enumerate(generated):
        row = embeds[prompt + index].long()
        assert row[0].item() == token
        assert torch.equal(
            row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET], agent_rows[index]
        )
        assert torch.equal(row[USER_STREAM_OFFSET:], timeline.user_rows[prompt + index])

    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor([80])), None, None, [request]
    )
    resumed = model.depformer.calls[-1]
    assert (resumed.forced == -1).all()
    frames = data.talker_model_inputs["frames"]
    assert len(frames) == frames_before + 1
    assert torch.equal(frames[-1], output_frame(agent_rows[-1], resumed.codes))
