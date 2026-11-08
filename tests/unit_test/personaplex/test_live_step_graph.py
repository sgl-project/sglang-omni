# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.personaplex.architecture import NUM_AUDIO_STREAMS, NUM_STREAMS
from sglang_omni.models.personaplex.components.depformer import Depformer
from sglang_omni.models.personaplex.components.live_step_graph import (
    GREEDY_LIVE_STEP,
    LiveRowGraphRunner,
    LiveStepGraphRunner,
    LiveStepSampling,
    captured_buckets,
    graph_buckets,
    live_step,
)
from sglang_omni.models.personaplex.sampling import AudioSampling
from sglang_omni.models.personaplex.sglang_model import PersonaPlexForCausalLM
from sglang_omni.models.personaplex.timeline import output_frame

from .test_depformer import SPEC, reference_weights
from .test_model_runner import FakeModel, make_request, make_runner, prefill, scheduled

VOCAB = 32000
DEFAULT_LIVE_STEP = LiveStepSampling.of(AudioSampling(0.7, 25), AudioSampling(0.8, 250))


def eager_live_step(depformer, logits, hidden, previous, sampling) -> torch.Tensor:
    batch_size, steps = previous.shape
    return live_step(
        depformer,
        logits,
        hidden,
        previous,
        sampling,
        key_value_caches=depformer.create_key_value_caches(hidden),
        output_codes=torch.empty(
            batch_size, steps, dtype=torch.long, device=logits.device
        ),
        packed=torch.empty(
            batch_size, 1 + 2 * steps, dtype=torch.long, device=logits.device
        ),
    )


def scaled_depformer(
    device: str = "cpu", dtype: torch.dtype = torch.float32
) -> Depformer:
    torch.manual_seed(42)
    model = Depformer(SPEC)
    model.load_reference_weights(
        {
            name: (
                tensor / tensor.shape[-1] ** 0.5
                if tensor.ndim == 2 and "emb" not in name
                else tensor
            )
            for name, tensor in reference_weights(8).items()
        }
    )
    return model.to(device=device, dtype=dtype)


class EagerLiveSteps(LiveStepGraphRunner):
    """Eager stand-in for a replay on hosts without CUDA."""

    def graph_for(self, batch_size, sampling):
        return sampling

    def run(self, captured, logits_BV, previous_agent_rows):
        return eager_live_step(
            self.depformer,
            logits_BV,
            self.hidden_out[: logits_BV.shape[0]],
            torch.stack(previous_agent_rows),
            captured,
        )


class EagerLiveRows(LiveRowGraphRunner):
    """Eager stand-in for a row-embedding replay on hosts without CUDA."""

    def __init__(self, model) -> None:
        super().__init__(model, (2,))
        self.host_rows = torch.zeros(2, NUM_STREAMS, dtype=torch.long)

    def embed(self, batch_size):
        self.model.fusion_buffer[:batch_size] = self.model.embed_rows(
            self.host_rows[:batch_size].clone()
        )


def decode_steps(
    live: bool, num_steps: int = 3, host_built: bool = False
) -> list[list[torch.Tensor]]:
    model = FakeModel(max_batch=2)
    model.depformer = scaled_depformer()
    model.hidden_out = torch.zeros(2, SPEC.input_dim)
    runner = make_runner(model)
    if live:
        runner.live_step_graph = EagerLiveSteps(
            model.depformer, model.hidden_out, VOCAB, (1, 2)
        )
    else:
        pass
    if host_built:
        runner.live_row_graph = EagerLiveRows(model)
        runner.steady_units_as_decode = True
    else:
        pass
    inputs = []
    greedy = {"temperature": 0.0, "audio_temperature": 0.0}
    requests = [
        make_request(num_steps + 2, params=greedy, request_id=f"r{index}")
        for index in range(2)
    ]
    generator = torch.Generator().manual_seed(3)
    for request in requests:
        prefill(runner, request, scheduled(request))
    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor([7, 8])), None, None, requests
    )
    tokens = [[7], [8]]
    for _ in range(num_steps):
        runner.before_decode(
            None,
            SimpleNamespace(
                reqs=[
                    scheduled(request, output)
                    for request, output in zip(requests, tokens)
                ]
            ),
            requests,
        )
        inputs.append(model.fusion_buffer.clone())
        model.hidden_out.normal_(generator=generator)
        logits = torch.randn(2, VOCAB, generator=generator)
        result = SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=logits),
            next_token_ids=None,
        )
        if runner.sample_before_post_decode(None, None, requests):
            assert not live
            result.next_token_ids = logits.argmax(dim=-1)
        else:
            assert live
        runner.post_decode(result, None, None, requests)
        for output, token in zip(tokens, result.next_token_ids.tolist(), strict=True):
            output.append(token)
    return [
        [
            *request.data.talker_model_inputs["frames"],
            *request.data.talker_model_inputs["agent_rows"],
            torch.tensor(output),
        ]
        for request, output in zip(requests, tokens, strict=True)
    ] + [inputs]


def test_a_live_step_decodes_what_sampling_then_spelling_does():
    for live_rows, eager_rows in zip(
        decode_steps(True), decode_steps(False), strict=True
    ):
        assert len(live_rows) == len(eager_rows)
        for live_row, eager_row in zip(live_rows, eager_rows, strict=True):
            assert torch.equal(live_row, eager_row)


def test_host_built_rows_are_the_rows_built_on_the_device():
    for host_rows, device_rows in zip(
        decode_steps(True, host_built=True), decode_steps(True), strict=True
    ):
        assert len(host_rows) == len(device_rows)
        for host_row, device_row in zip(host_rows, device_rows, strict=True):
            assert torch.equal(host_row, device_row)


@pytest.mark.parametrize(
    "configured,max_batch_size,expected",
    [
        ((1, 2, 4, 8, 16, 32), 4, (1, 2, 4, 8, 16, 32)),
        ((1, 2, 4, 8, 16, 32), 65, (1, 2, 4, 8, 16, 32, 64, 65)),
        ((1, 2, 4, 8, 16, 32), 64, (1, 2, 4, 8, 16, 32, 64)),
        ((1, 4), 9, (1, 4, 8, 9)),
    ],
)
def test_graph_buckets_reach_the_lm_concurrency(configured, max_batch_size, expected):
    assert graph_buckets(configured, max_batch_size) == expected


def test_buckets_stop_at_the_first_that_holds_every_request_and_at_the_rows():
    assert captured_buckets((1, 2, 4, 8, 16, 32), 9, 9) == (1, 2, 4, 8, 9)
    assert captured_buckets((1, 2, 4, 8, 16, 32), 33, 33) == (1, 2, 4, 8, 16, 32, 33)
    assert captured_buckets((1, 2, 4, 8), 13, 64) == (1, 2, 4, 8, 13)


def test_only_unseeded_steps_of_one_captured_setting_take_the_graph():
    model = FakeModel(max_batch=2)
    model.depformer = scaled_depformer()
    model.hidden_out = torch.zeros(2, SPEC.input_dim)
    runner = make_runner(model)
    runner.live_step_graph = LiveStepGraphRunner(
        model.depformer, model.hidden_out, VOCAB, (1, 2)
    )
    runner.live_step_graph.graphs = {
        (2, GREEDY_LIVE_STEP): "greedy",
        (2, DEFAULT_LIVE_STEP): "default",
    }
    greedy = {"temperature": 0.0, "audio_temperature": 0.0}

    def chosen(*params):
        return runner.live_step_for(
            [
                make_request(2, params=p, request_id=f"r{i}")
                for i, p in enumerate(params)
            ]
        )

    assert chosen(greedy, greedy) == "greedy"
    assert chosen({}, {}) == "default"
    assert chosen(greedy, {}) is None
    assert chosen({"seed": 5}, {"seed": 6}) is None
    assert chosen({"top_p": 0.9}, {"top_p": 0.9}) is None
    assert chosen({}, {}, {}) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "sampling", [GREEDY_LIVE_STEP, DEFAULT_LIVE_STEP], ids=["greedy", "sampled"]
)
@torch.inference_mode()
def test_graph_replay_matches_the_eager_live_step(dtype, sampling):
    depformer = scaled_depformer("cuda", dtype)
    depformer.warmup_pointwise()
    hidden_out = torch.zeros(16, SPEC.input_dim, device="cuda", dtype=dtype)
    runner = LiveStepGraphRunner(depformer, hidden_out, VOCAB, (1, 2, 4, 8))
    buckets = graph_buckets(runner.batch_sizes, 16)
    steps = SPEC.steps
    cases = []
    # Note (Junnan Li): The eager references are taken before capture.
    for batch_size in (1, 3, 8, 11, 16):
        bucket_size = next(size for size in buckets if size >= batch_size)
        hidden = torch.randn(bucket_size, SPEC.input_dim, device="cuda", dtype=dtype)
        logits = torch.randn(batch_size, VOCAB, device="cuda")
        previous = torch.randint(0, 2048, (batch_size, steps), device="cuda")
        hidden_out[:bucket_size] = hidden
        state = torch.cuda.get_rng_state()
        expected = eager_live_step(
            depformer,
            torch.cat((logits, logits.new_zeros(bucket_size - batch_size, VOCAB))),
            hidden_out[:bucket_size],
            torch.cat((previous, previous.new_zeros(bucket_size - batch_size, steps))),
            sampling,
        )[:batch_size]
        cases.append(
            (hidden, logits, previous, state, expected, torch.cuda.get_rng_state())
        )
    runner.capture_graphs((sampling,), max_batch_size=16)
    assert runner.batch_sizes == buckets == (1, 2, 4, 8, 16)
    for hidden, logits, previous, state, expected, expected_state in cases:
        captured = runner.graph_for(logits.shape[0], sampling)
        assert captured is not None and captured.logits.shape[0] == hidden.shape[0]
        hidden_out[: hidden.shape[0]] = hidden
        torch.cuda.set_rng_state(state)
        actual = runner.run(captured, logits, list(previous.unbind(0)))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.equal(torch.cuda.get_rng_state(), expected_state)
        assert torch.equal(
            actual[:, 1 : 1 + steps], output_frame(previous, actual[:, 1 + steps :])
        )
        if sampling == GREEDY_LIVE_STEP:
            assert torch.equal(actual[:, 0], logits.argmax(dim=-1))
        else:
            top_k = logits.topk(25, dim=-1).indices
            assert (top_k == actual[:, :1]).any(dim=1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_eager_live_steps_between_replays_match_the_ones_before_capture():
    depformer = scaled_depformer("cuda", torch.float32)
    depformer.warmup_pointwise()
    hidden_out = torch.zeros(16, SPEC.input_dim, device="cuda")
    steps = SPEC.steps
    cases = []
    for batch_size in (2, 5, 16, 1, 9):
        hidden = torch.randn(batch_size, SPEC.input_dim, device="cuda")
        logits = torch.randn(batch_size, VOCAB, device="cuda")
        previous = torch.randint(0, 2048, (batch_size, steps), device="cuda")
        hidden_out[:batch_size] = hidden
        reference = eager_live_step(
            depformer, logits, hidden_out[:batch_size], previous, GREEDY_LIVE_STEP
        )
        cases.append((hidden, logits, previous, reference))
    runner = LiveStepGraphRunner(depformer, hidden_out, VOCAB, (1, 2, 4, 8))
    runner.capture_graphs((GREEDY_LIVE_STEP,), max_batch_size=16)
    for _ in range(2):
        for hidden, logits, previous, reference in cases:
            hidden_out[: hidden.shape[0]] = hidden
            eager = eager_live_step(
                depformer,
                logits,
                hidden_out[: hidden.shape[0]],
                previous,
                GREEDY_LIVE_STEP,
            )
            replayed = runner.run(
                runner.graph_for(hidden.shape[0], GREEDY_LIVE_STEP),
                logits,
                list(previous.unbind(0)),
            )
            torch.testing.assert_close(eager, reference, atol=0, rtol=0)
            torch.testing.assert_close(replayed, reference, atol=0, rtol=0)


class TinyEmbedder(torch.nn.Module):
    embed_rows = PersonaPlexForCausalLM.embed_rows

    def __init__(self, dim: int = 16, max_batch: int = 16) -> None:
        super().__init__()
        self.audio_emb = torch.nn.ModuleList(
            torch.nn.Embedding(2049, dim) for _ in range(NUM_AUDIO_STREAMS)
        )
        self.text_emb = torch.nn.Embedding(VOCAB + 1, dim)
        self.register_buffer("fusion_buffer", torch.zeros(max_batch, dim))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_row_embedding_replay_matches_the_eager_embedding():
    """Replays run from the scheduler, outside inference mode."""
    torch.manual_seed(5)
    model = TinyEmbedder().cuda().requires_grad_(False)
    runner = LiveRowGraphRunner(model, (1, 2, 4, 8))
    runner.capture_graphs(max_batch_size=13)
    assert runner.batch_sizes == (1, 2, 4, 8, 13)
    for batch_size in (1, 3, 8, 13):
        rows = torch.cat(
            (
                torch.randint(0, VOCAB, (batch_size, 1)),
                torch.randint(0, 2048, (batch_size, NUM_STREAMS - 1)),
            ),
            dim=1,
        )
        runner.staging(batch_size)[:] = rows.numpy()
        expected = model.embed_rows(rows.cuda())
        runner.embed(batch_size)
        torch.testing.assert_close(
            model.fusion_buffer[:batch_size], expected, atol=0, rtol=0
        )
