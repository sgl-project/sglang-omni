# SPDX-License-Identifier: Apache-2.0
"""Prefill and decode hooks feed timeline rows to the model and frames to the codec."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import PendingStep
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.personaplex.architecture import (
    AGENT_STREAM_OFFSET,
    NUM_STREAMS,
    USER_STREAM_OFFSET,
)
from sglang_omni.models.personaplex.model_runner import (
    PersonaPlexFrameBatch,
    PersonaPlexModelRunner,
)
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.request_builders import (
    apply_lm_result,
    build_lm_request,
)
from sglang_omni.models.personaplex.sampling import sample_token
from sglang_omni.models.personaplex.timeline import output_frame
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerOutput

VOICE_FRAMES = 4
# Note (edwardzh): Leave generous host scheduling margin for the single-resolve check.
SLOW_FORWARD_CYCLES = 2_000_000_000


class FakeDepformer:
    """Returns base + 100 * row + step codes, keeping forced ones; records calls."""

    spec = SimpleNamespace(steps=8)

    def __init__(self):
        self.calls = []

    def generate(self, text_token_B, transformer_out_BD, forced_BK, sample):
        base = 1000 * (1 + len(self.calls))
        rows = 100 * torch.arange(forced_BK.shape[0], device=forced_BK.device)[:, None]
        codes = torch.where(
            forced_BK >= 0,
            forced_BK,
            base + rows + torch.arange(8, device=forced_BK.device),
        )
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
    runner.output_processor = SGLangOutputProcessor()
    runner.async_query_hit = runner.async_query_miss = 0
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


def scheduled(request, output_ids=(), prefix_len: int = 0) -> SimpleNamespace:
    """The scheduler's Req for request: prompt ids, generated ids, cached prefix."""
    return SimpleNamespace(
        origin_input_ids=list(request.data.req.origin_input_ids),
        output_ids=list(output_ids),
        prefix_len=prefix_len,
    )


def prefill(runner, request, req) -> torch.Tensor:
    extend_len = len(req.origin_input_ids) + len(req.output_ids)
    forward_batch = SimpleNamespace(
        replace_embeds=None,
        input_ids=torch.zeros(extend_len - req.prefix_len),
    )
    runner.before_prefill(forward_batch, SimpleNamespace(reqs=[req]), [request])
    return get_omni_prefill_inputs(forward_batch).input_embeds


def generate(runner, request, tokens: list[int]) -> None:
    """Prefill, then decode, sampling tokens in order."""
    prefill(runner, request, scheduled(request))
    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor(tokens[:1])), None, None, [request]
    )
    for step in range(1, len(tokens)):
        runner.before_decode(
            None,
            SimpleNamespace(reqs=[scheduled(request, tokens[:step])]),
            [request],
        )
        runner.post_decode(
            SimpleNamespace(next_token_ids=torch.tensor(tokens[step : step + 1])),
            None,
            None,
            [request],
        )


def test_prefill_uses_stored_voice_rows_and_embeds_the_rest():
    runner = make_runner(FakeModel())
    with_voice, without_voice = make_request(3, voice=True), make_request(2)
    voice_timeline = with_voice.data.talker_model_inputs["timeline"]
    plain_timeline = without_voice.data.talker_model_inputs["timeline"]
    total = voice_timeline.num_prompt_positions + plain_timeline.num_prompt_positions
    forward_batch = SimpleNamespace(replace_embeds=None, input_ids=torch.zeros(total))

    runner.before_prefill(
        forward_batch,
        SimpleNamespace(reqs=[scheduled(with_voice), scheduled(without_voice)]),
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
    prefill(runner, request, scheduled(request))

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
        None, SimpleNamespace(reqs=[scheduled(request, [77])]), [request]
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
        None, SimpleNamespace(reqs=[scheduled(request, [77, 78])]), [request]
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
        SimpleNamespace(reqs=[scheduled(request) for request in requests]),
        requests,
    )
    runner.post_prefill(
        SimpleNamespace(
            next_token_ids=torch.tensor(text_token_ids, device=runner.model_device)
        ),
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

    def generate(self, text_token_B, transformer_out_BD, forced_BK, sample):
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

    generate(runner, request, [77, 78, 79])
    previous_feedback = runner.decode_feedback(data).user_rows
    previous_copy = previous_feedback.clone()
    timeline.append_caller_frames(torch.arange(1024).view(128, 8))
    expanded_feedback = runner.decode_feedback(data).user_rows
    assert expanded_feedback.shape == timeline.user_rows.shape
    assert torch.equal(previous_feedback, previous_copy)

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
        SimpleNamespace(reqs=[scheduled(request, generated), scheduled(fresh)]),
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


def test_decode_uses_owned_device_feedback_before_host_commit() -> None:
    model = FakeModel()
    runner = make_runner(model)
    request = make_request(4)
    run_prefill(runner, [request], [77])
    sampled = torch.tensor([78])
    runner.post_decode(SimpleNamespace(next_token_ids=sampled), None, None, [request])
    sampled.fill_(999)

    runner.before_decode(None, None, [request], is_lookahead=True)
    timeline = request.data.talker_model_inputs["timeline"]
    row = model.fusion_buffer[0].long()
    assert row[0].item() == 78
    assert torch.equal(row[1:9], model.depformer.calls[-1].codes[0])
    assert torch.equal(row[9:], timeline.user_rows[timeline.input_position(2)])


class DeferredHostRows:
    def __init__(self) -> None:
        self.source: torch.Tensor | None = None
        self.ready = False

    def __getitem__(self, rows: slice) -> "DeferredHostRows":
        assert rows == slice(None, 1)
        return self

    def copy_(self, source: torch.Tensor, *, non_blocking: bool) -> None:
        assert non_blocking
        self.source = source

    def clone(self) -> torch.Tensor:
        assert self.ready, "Host staging was read before completion"
        assert self.source is not None
        return self.source.clone()

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.ready = True


def pending_frame(runner, request, token: int) -> PendingStep:
    host_rows = DeferredHostRows()
    runner.next_host_staging = lambda shape, dtype: host_rows
    result = SimpleNamespace(
        next_token_ids=torch.tensor([token]), can_run_cuda_graph=True
    )
    frames = runner.post_decode_launch(result, None, [request])
    batch = SimpleNamespace(reqs=[request.data.req])
    return PendingStep(
        event=host_rows,
        launch_buf=frames,
        forward_batch=None,
        schedule_batch=batch,
        batch_result=result,
        scheduler_output=SchedulerOutput(requests=[request], batch_data=batch),
    )


def test_resolve_waits_for_its_copy_and_does_not_wait_for_next_step() -> None:
    runner = make_runner(FakeModel())
    request = make_request(5)
    run_prefill(runner, [request], [77])
    first = pending_frame(runner, request, 78)
    second = pending_frame(runner, request, 79)
    assert len(request.data.talker_model_inputs["agent_rows"]) == 1
    output = runner.execute_resolve(first)
    assert output.host_token_ids.tolist() == [78]
    assert not second.event.query()
    frames = request.data.talker_model_inputs["frames"]
    first_frame = frames[-1].clone()
    first.event.source.fill_(999)
    assert torch.equal(frames[-1], first_frame)
    runner.execute_resolve(second)
    assert len(frames) == 3


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "request_orders",
    [
        [(2, 1, 0), (1, 0), (0,)],
        [(2, 1, 0), (0,)],
        [(0,), (2, 0, 1), (1, 2), (2,), (2,)],
    ],
    ids=["shrink-3-2-1", "shrink-3-1", "grow-shrink-reorder"],
)
def test_async_decode_with_changing_batches(
    device: str, request_orders: list[tuple[int, ...]]
) -> None:
    model = FakeModel(max_batch=3)
    model.fusion_buffer = model.fusion_buffer.to(device)
    model.hidden_out = model.hidden_out.to(device)
    runner = make_runner(model)
    runner.host_staging_buffers = (
        []
        if device == "cuda"
        else [torch.empty((3, 9), dtype=torch.long) for _ in range(2)]
    )
    runner.staging_slot = 0
    requests = [make_request(8, request_id=f"r{index}") for index in range(3)]
    run_prefill(runner, requests, [70, 71, 72])
    previous_tokens = [70, 71, 72]
    previous_codes = list(model.depformer.calls[0].codes.cpu())
    expected_frames = [
        [request.data.talker_model_inputs["frames"][0].clone()] for request in requests
    ]
    fused_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    pending: list[PendingStep] = []
    expected_tokens = [
        list(range(80 + 10 * step, 80 + 10 * step + len(order)))
        for step, order in enumerate(request_orders)
    ]
    sampled_tokens = [torch.tensor(tokens, device=device) for tokens in expected_tokens]

    for step, order in enumerate(request_orders):
        active_requests = [requests[index] for index in order]
        runner.before_decode(None, None, active_requests, is_lookahead=True)
        expected_inputs = []
        for index in order:
            timeline = requests[index].data.talker_model_inputs["timeline"]
            expected_inputs.append(
                torch.cat(
                    (
                        torch.tensor([previous_tokens[index]]),
                        previous_codes[index],
                        timeline.user_rows[
                            timeline.input_position(len(expected_frames[index]))
                        ],
                    )
                )
            )
        fused_rows.append(
            (
                model.fusion_buffer[: len(order)].clone(),
                torch.stack(expected_inputs).float(),
            )
        )
        tokens = expected_tokens[step]
        result = SimpleNamespace(
            next_token_ids=sampled_tokens[step], can_run_cuda_graph=True
        )
        frames = runner.post_decode_launch(result, None, active_requests)
        if device == "cuda":
            completed = torch.cuda.Event()
            completed.record()
        else:
            completed = SimpleNamespace(query=lambda: True)
        batch = SimpleNamespace(reqs=[request.data.req for request in active_requests])
        pending.append(
            PendingStep(
                event=completed,
                launch_buf=frames,
                forward_batch=None,
                schedule_batch=batch,
                batch_result=result,
                scheduler_output=SchedulerOutput(
                    requests=active_requests, batch_data=batch
                ),
            )
        )
        if step:
            output = runner.execute_resolve(pending[step - 1])
            assert output.host_token_ids.tolist() == expected_tokens[step - 1]
        else:
            pass
        for row, index in enumerate(order):
            codes = 1000 * (step + 2) + 100 * row + torch.arange(8)
            expected_frames[index].append(output_frame(previous_codes[index], codes))
            previous_tokens[index] = tokens[row]
            previous_codes[index] = codes

    output = runner.execute_resolve(pending[-1])
    assert output.host_token_ids.tolist() == expected_tokens[-1]
    for actual, expected in fused_rows:
        assert torch.equal(actual.cpu(), expected)
    for request, expected in zip(requests, expected_frames, strict=True):
        actual = request.data.talker_model_inputs["frames"]
        assert len(actual) == len(expected)
        for actual_frame, expected_frame in zip(actual, expected, strict=True):
            assert torch.equal(actual_frame, expected_frame)


def test_retract_discards_uncommitted_audio_draw_and_replays_it() -> None:
    def setup():
        model = FakeModel()
        model.depformer = SamplingDepformer()
        runner = make_runner(model)
        request = make_request(6, params={"seed": 7})
        run_prefill(runner, [request], [77])
        return runner, request

    reference, expected = setup()
    for token in (78, 79):
        reference.post_decode(
            SimpleNamespace(next_token_ids=torch.tensor([token])),
            None,
            None,
            [expected],
        )
    runner, request = setup()
    committed = pending_frame(runner, request, 78)
    discarded = pending_frame(runner, request, 79)
    runner.execute_resolve(committed)
    request.data.req.is_retracted = True
    runner.execute_resolve(discarded)
    assert len(request.data.talker_model_inputs["frames"]) == 2
    request.data.req.is_retracted = False
    runner.before_prefill(
        SimpleNamespace(
            replace_embeds=None,
            input_ids=torch.zeros(
                request.data.talker_model_inputs["timeline"].num_prompt_positions + 2
            ),
        ),
        SimpleNamespace(reqs=[scheduled(request, [77, 78])]),
        [request],
    )
    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor([79])), None, None, [request]
    )
    for actual, target in zip(
        request.data.talker_model_inputs["frames"],
        expected.data.talker_model_inputs["frames"],
        strict=True,
    ):
        assert torch.equal(actual, target)
    assert runner.decode_feedback(request.data).launched_frames == 3


def test_lookahead_requires_budget_and_history_free_sampling(
    monkeypatch,
) -> None:
    runner = make_runner(FakeModel())
    request = make_request(3)
    run_prefill(runner, [request], [77])
    request.data.req.omni_data = request.data
    batch = SimpleNamespace(reqs=[request.data.req])
    monkeypatch.setattr(
        PersonaPlexModelRunner,
        "model_device",
        property(lambda self: torch.device("cuda")),
    )
    rows = runner.decode_feedback(request.data)
    assert runner.lookahead_eligible(batch)
    rows.launched_frames = 3
    assert not runner.lookahead_eligible(batch)
    rows.launched_frames = 1
    request.data.req.sampling_params.repetition_penalty = 1.2
    assert not runner.lookahead_eligible(batch)
    request.data.req.sampling_params.repetition_penalty = 1.0
    request.data.return_logprob = True
    assert not runner.lookahead_eligible(batch)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_resolve_cuda_copy_while_next_forward_is_slow() -> None:
    runner = make_runner(FakeModel())
    request = make_request(4)
    run_prefill(runner, [request], [77])
    device_rows = torch.arange(9, device="cuda").view(1, 9)
    host_rows = torch.empty((1, 9), dtype=torch.long, pin_memory=True)
    host_rows.copy_(device_rows, non_blocking=True)
    completed = torch.cuda.Event()
    completed.record()
    completed.synchronize()
    frames = PersonaPlexFrameBatch(
        host_rows=host_rows, agent_rows=device_rows[:, 1:], audio_rng_states=[None]
    )
    batch = SimpleNamespace(reqs=[request.data.req])
    pending = PendingStep(
        event=completed,
        launch_buf=frames,
        forward_batch=None,
        schedule_batch=batch,
        batch_result=SimpleNamespace(
            next_token_ids=device_rows[:, 0], can_run_cuda_graph=True
        ),
        scheduler_output=SchedulerOutput(requests=[request], batch_data=batch),
    )
    torch.cuda._sleep(SLOW_FORWARD_CYCLES)  # noqa: leading-underscore
    next_completed = torch.cuda.Event()
    next_completed.record()
    try:
        output = runner.execute_resolve(pending)
        assert output.host_token_ids.tolist() == [0]
        assert not next_completed.query(), "Resolving N waited for N+1"
        assert request.data.talker_model_inputs["frames"][-1].tolist() == list(
            range(1, 9)
        )
    finally:
        next_completed.synchronize()


@pytest.mark.parametrize(
    "cached", ["none", "inside the prompt", "prompt", "all but one"]
)
def test_prefill_from_a_cached_prefix_embeds_only_the_missing_rows(cached: str):
    model = FakeModel()
    runner = make_runner(model)
    request = make_request(5)
    generated = [77, 78, 79]
    generate(runner, request, generated)
    prompt = request.data.talker_model_inputs["timeline"].num_prompt_positions
    agent_rows = request.data.talker_model_inputs["agent_rows"]
    full = prefill(runner, request, scheduled(request, generated))

    prefix_len = {
        "none": 0,
        "inside the prompt": prompt - 2,
        "prompt": prompt,
        "all but one": prompt + len(generated) - 1,
    }[cached]
    rows = prefill(runner, request, scheduled(request, generated, prefix_len))
    assert torch.equal(rows, full[prefix_len:])
    feedback = request.data.talker_model_inputs["decode_feedback"]
    assert torch.equal(feedback.agent_row, agent_rows[-1])
    assert (request.data.talker_model_inputs["prefill_forced"] == -1).all()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ],
        ),
    ],
)
def test_feedback_appends_only_valid_history_and_preserves_old_views(device):
    model = FakeModel()
    model.fusion_buffer = model.fusion_buffer.to(device)
    runner = make_runner(model)
    request = make_request(1)
    timeline = request.data.talker_model_inputs["timeline"]
    feedback = runner.decode_feedback(request.data)
    saved_views = []
    for added_frames in [1, 1, 64, 1, 200, 1]:
        old_rows = feedback.user_rows
        old_storage = feedback.user_rows_storage
        saved_views.append((old_rows, old_rows.clone()))
        timeline.append_caller_frames(
            torch.arange(added_frames * 8).view(added_frames, 8)
        )
        assert runner.decode_feedback(request.data) is feedback
        assert torch.equal(feedback.user_rows.cpu(), timeline.user_rows)
        if timeline.user_rows.shape[0] <= old_storage.shape[0]:
            assert feedback.user_rows_storage is old_storage
        else:
            assert feedback.user_rows_storage is not old_storage
        unchanged_rows = feedback.user_rows
        assert runner.decode_feedback(request.data).user_rows is unchanged_rows
    for old_rows, old_copy in saved_views:
        assert torch.equal(old_rows, old_copy)
