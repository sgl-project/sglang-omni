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
from sglang_omni.models.personaplex.request_builders import (
    apply_lm_result,
    build_lm_request,
)
from sglang_omni.models.personaplex.sampling import sample_token
from sglang_omni.models.personaplex.timeline import output_frame
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

VOICE_FRAMES = 4


class FakeDepformer:
    """Returns base + 100 * row + step codes, keeping forced ones; records calls."""

    spec = SimpleNamespace(steps=8)

    def __init__(self):
        self.calls = []

    def generate(self, text_token_B, transformer_out_BD, forced_BK, sample):
        base = 1000 * (1 + len(self.calls))
        rows = 100 * torch.arange(forced_BK.shape[0])[:, None]
        codes = torch.where(forced_BK >= 0, forced_BK, base + rows + torch.arange(8))
        self.calls.append(
            SimpleNamespace(
                text=text_token_B.clone(),
                hidden=transformer_out_BD.clone(),
                forced=forced_BK.clone(),
                codes=codes.clone(),
                sample=sample,
            )
        )
        return codes


class FakeModel:
    """Embeds a row as its own token ids, so fused inputs can be read back."""

    def __init__(self, max_batch: int = 2):
        self.fusion_buffer = torch.zeros(max_batch, NUM_STREAMS)
        self.hidden_out = torch.arange(max_batch * NUM_STREAMS, dtype=torch.float32)
        self.hidden_out = self.hidden_out.view(max_batch, NUM_STREAMS)
        self.depformer = FakeDepformer()

    def embed_rows(self, rows_NK: torch.Tensor) -> torch.Tensor:
        return rows_NK.to(torch.float32)


def make_runner(model: FakeModel) -> PersonaPlexModelRunner:
    runner = PersonaPlexModelRunner.__new__(PersonaPlexModelRunner)
    runner.model = model
    return runner


def make_request(
    num_frames: int,
    *,
    voice: bool = False,
    params: dict[str, int | float] | None = None,
    request_id: str = "r",
) -> SimpleNamespace:
    state = PersonaPlexState(
        text_prompt_ids=[11, 12, 13],
        user_codes=torch.arange(num_frames * 8).view(num_frames, 8) + 500,
    )
    if voice:
        state.voice_frames = VOICE_FRAMES
        state.voice_embeddings = torch.randn(VOICE_FRAMES - 1, NUM_STREAMS)
        state.voice_tail_codes = torch.arange(16).view(2, 8) + 300
    payload = StagePayload(
        request_id,
        request=OmniRequest(inputs={}, params=params or {}),
        data=state.to_dict(),
    )
    return SimpleNamespace(
        request_id=request_id, data=build_lm_request(payload, vocab_size=32000)
    )


def test_prefill_uses_stored_voice_rows_and_embeds_the_rest():
    runner = make_runner(FakeModel())
    with_voice, without_voice = make_request(3, voice=True), make_request(2)
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
    model = FakeModel()
    runner = make_runner(model)
    request = make_request(3, voice=True)
    data = request.data
    timeline = data.talker_model_inputs["timeline"]
    first_position = timeline.num_prompt_positions
    runner.before_prefill(
        SimpleNamespace(replace_embeds=None, input_ids=torch.zeros(first_position)),
        SimpleNamespace(reqs=[SimpleNamespace(output_ids=[])]),
        [request],
    )

    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor([77])), None, None, [request]
    )
    prefill_call = model.depformer.calls[0]
    assert prefill_call.text.tolist() == [77]
    assert torch.equal(prefill_call.hidden, model.hidden_out[0:1])
    assert torch.equal(prefill_call.forced[0], timeline.forced_agent_at_start)
    assert torch.equal(
        data.talker_model_inputs["frames"][0],
        output_frame(timeline.agent_row_before_start, prefill_call.codes[0]),
    )

    runner.before_decode(
        None, SimpleNamespace(reqs=[SimpleNamespace(output_ids=[77])]), [request]
    )
    row = model.fusion_buffer[0].long()
    assert row[0].item() == 77
    assert torch.equal(
        row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET], prefill_call.codes[0]
    )
    assert torch.equal(row[USER_STREAM_OFFSET:], timeline.user_rows[first_position])

    runner.post_decode(
        SimpleNamespace(next_token_ids=torch.tensor([78])), None, None, [request]
    )
    decode_call = model.depformer.calls[1]
    assert decode_call.text.tolist() == [78]
    assert (decode_call.forced == -1).all()
    frames = data.talker_model_inputs["frames"]
    assert torch.equal(
        frames[1], output_frame(prefill_call.codes[0], decode_call.codes[0])
    )
    assert len(data.talker_model_inputs["pending_frames"]) == 2

    runner.before_decode(
        None, SimpleNamespace(reqs=[SimpleNamespace(output_ids=[77, 78])]), [request]
    )
    row = model.fusion_buffer[0].long()
    assert row[0].item() == 78
    assert torch.equal(
        row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET], decode_call.codes[0]
    )
    assert torch.equal(row[USER_STREAM_OFFSET:], timeline.user_rows[first_position + 1])


def run_prefill(runner, requests, text_token_ids) -> None:
    prompt_positions = sum(
        request.data.talker_model_inputs["timeline"].num_prompt_positions
        for request in requests
    )
    runner.before_prefill(
        SimpleNamespace(replace_embeds=None, input_ids=torch.zeros(prompt_positions)),
        SimpleNamespace(reqs=[SimpleNamespace(output_ids=[]) for _ in requests]),
        requests,
    )
    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor(text_token_ids)),
        None,
        None,
        requests,
    )


def test_requests_with_any_sampling_share_one_depformer_pass() -> None:
    model = FakeModel(max_batch=5)
    runner = make_runner(model)
    requests = [
        make_request(2, request_id="plain"),
        make_request(2, request_id="greedy", params={"audio_temperature": 0.0}),
        make_request(2, request_id="seeded-0", params={"seed": 7}),
        make_request(2, request_id="top-5", params={"audio_top_k": 5}),
        make_request(2, request_id="seeded-1", params={"seed": 8}),
    ]
    timelines = [request.data.talker_model_inputs["timeline"] for request in requests]

    run_prefill(runner, requests, list(range(70, 75)))
    [prefill_call] = model.depformer.calls
    assert prefill_call.text.tolist() == list(range(70, 75))
    assert torch.equal(prefill_call.hidden, model.hidden_out)
    for index, (request, timeline) in enumerate(zip(requests, timelines, strict=True)):
        assert torch.equal(prefill_call.forced[index], timeline.forced_agent_at_start)
        agent_rows = request.data.talker_model_inputs["agent_rows"]
        assert torch.equal(agent_rows[0], prefill_call.codes[index])

    logits = torch.randn(5, 64)
    picks = prefill_call.sample(logits)
    assert picks[1] == logits[1].argmax()
    assert picks[3] in torch.topk(logits[3], 5).indices
    for row in (2, 4):
        sampling = requests[row].data.talker_model_inputs["sampling"]
        generator = torch.Generator().manual_seed(sampling.audio_seed)
        assert picks[row] == sample_token(
            logits[row : row + 1], sampling.audio, [generator]
        )

    runner.post_decode(
        SimpleNamespace(next_token_ids=torch.arange(80, 85)), None, None, requests
    )
    [decode_call] = model.depformer.calls[1:]
    assert (decode_call.forced == -1).all()
    for index, request in enumerate(requests):
        frames = request.data.talker_model_inputs["frames"]
        assert torch.equal(
            frames[1], output_frame(prefill_call.codes[index], decode_call.codes[index])
        )


class SamplingDepformer:
    """Samples every step from logits that depend on the step alone, so a
    request's codes depend only on its sampling settings and random stream."""

    spec = SimpleNamespace(steps=8)

    def __init__(self):
        self.logits = torch.randn(8, 64, generator=torch.Generator().manual_seed(0))
        self.batch_sizes = []

    def generate(self, text_token_B, transformer_out_BD, forced_BK, sample):
        self.batch_sizes.append(forced_BK.shape[0])
        codes = []
        for step in range(8):
            sampled = sample(self.logits[step].expand(forced_BK.shape[0], -1))
            forced = forced_BK[:, step]
            codes.append(torch.where(forced >= 0, forced, sampled))
        return torch.stack(codes, dim=1)


def test_seeded_codes_ignore_row_order_and_neighbours_leaving() -> None:
    def seeded_frames(schedule) -> list[torch.Tensor]:
        model = FakeModel(max_batch=4)
        model.depformer = SamplingDepformer()
        runner = make_runner(model)
        seeded = make_request(6, request_id="seeded", params={"seed": 7})
        neighbours = {
            "plain": make_request(1, request_id="plain"),
            "other-seed": make_request(6, request_id="other-seed", params={"seed": 8}),
            "top-3": make_request(6, request_id="top-3", params={"audio_top_k": 3}),
            "greedy": make_request(
                6, request_id="greedy", params={"audio_temperature": 0.0}
            ),
        }
        requests = {"seeded": seeded, **neighbours}
        for kind, names in schedule:
            batch = [requests[name] for name in names]
            if kind == "prefill":
                run_prefill(runner, batch, [70] * len(batch))
            else:
                runner.post_decode(
                    SimpleNamespace(next_token_ids=torch.full((len(batch),), 80)),
                    None,
                    None,
                    batch,
                )
        return seeded.data.talker_model_inputs["frames"]

    alone = seeded_frames([("prefill", ["seeded"])] + [("decode", ["seeded"])] * 3)
    # Note (edwardzh): plain finishes early, other-seed is aborted mid-stream, and
    # a retract reorders the rows, so seeded's row and pass neighbours change.
    batched = seeded_frames(
        [
            ("prefill", ["plain", "seeded", "other-seed"]),
            ("decode", ["other-seed", "plain", "seeded"]),
            ("prefill", ["top-3", "greedy"]),
            ("decode", ["seeded", "top-3", "greedy", "other-seed"]),
            ("decode", ["greedy", "top-3", "seeded"]),
        ]
    )
    assert len(batched) == len(alone) == 4
    for alone_frame, batched_frame in zip(alone, batched, strict=True):
        assert torch.equal(alone_frame, batched_frame)


def test_audio_sampler_keeps_one_generator_per_seeded_request():
    runner = make_runner(FakeModel())
    logits = torch.randn(1, 64)

    def draws(request):
        sampler = runner.audio_sampler([request])
        return [int(sampler(logits)) for _ in range(5)]

    params = {"seed": 7, "audio_temperature": 1.0, "audio_top_k": 0}
    first, second = make_request(1, params=params), make_request(1, params=params)
    assert draws(first) == draws(second)
    generator = first.data.talker_model_inputs["audio_generator"]
    runner.audio_sampler([first])
    assert first.data.talker_model_inputs["audio_generator"] is generator

    unseeded = make_request(1, params={"audio_temperature": 1.0})
    runner.audio_sampler([unseeded])
    assert "audio_generator" not in unseeded.data.talker_model_inputs


def test_resume_after_a_retract_replays_the_generated_positions() -> None:
    model = FakeModel()
    runner = make_runner(model)
    request = make_request(5)
    data = request.data
    timeline = data.talker_model_inputs["timeline"]
    prompt_positions = timeline.num_prompt_positions

    runner.before_prefill(
        SimpleNamespace(replace_embeds=None, input_ids=torch.zeros(prompt_positions)),
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

    fresh = make_request(2, request_id="fresh")
    fresh_timeline = fresh.data.talker_model_inputs["timeline"]
    replayed_positions = prompt_positions + len(generated)
    forward_batch = SimpleNamespace(
        replace_embeds=None,
        input_ids=torch.zeros(replayed_positions + fresh_timeline.num_prompt_positions),
    )
    runner.before_prefill(
        forward_batch,
        SimpleNamespace(
            reqs=[SimpleNamespace(output_ids=generated), SimpleNamespace(output_ids=[])]
        ),
        [request, fresh],
    )

    embeds = get_omni_prefill_inputs(forward_batch).input_embeds
    assert embeds.shape[0] == replayed_positions + fresh_timeline.num_prompt_positions
    assert torch.equal(embeds[:prompt_positions], timeline.prefill_tokens.float())
    for index, token in enumerate(generated):
        row = embeds[prompt_positions + index].long()
        assert row[0].item() == token
        assert torch.equal(
            row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET], agent_rows[index]
        )
        assert torch.equal(
            row[USER_STREAM_OFFSET:], timeline.user_rows[prompt_positions + index]
        )

    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor([80, 90])),
        None,
        None,
        [request, fresh],
    )
    resumed = model.depformer.calls[-1]
    assert resumed.text.tolist() == [80, 90]
    assert (resumed.forced[0] == -1).all()
    assert torch.equal(resumed.forced[1], fresh_timeline.forced_agent_at_start)
    frames = data.talker_model_inputs["frames"]
    assert len(frames) == frames_before + 1
    assert torch.equal(frames[-1], output_frame(agent_rows[-1], resumed.codes[0]))
    assert torch.equal(
        fresh.data.talker_model_inputs["frames"][0],
        output_frame(fresh_timeline.agent_row_before_start, resumed.codes[1]),
    )

    data.output_ids = generated + [80]
    state = PersonaPlexState.from_dict(apply_lm_result(data).data)
    assert state.text_ids == [3, 77, 78, 79]
    assert data.output_ids == [77, 78, 79, 80]
