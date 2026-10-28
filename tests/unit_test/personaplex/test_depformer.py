# SPDX-License-Identifier: Apache-2.0
"""Depformer weight slicing and teacher forcing, on a scaled-down spec."""

import pytest
import torch

from sglang_omni.models.personaplex.architecture import AUDIO_CARD, DepformerSpec
from sglang_omni.models.personaplex.components.depformer import Depformer
from sglang_omni.models.personaplex.components.depformer_cuda_graph import (
    DepformerCudaGraphRunner,
)
from sglang_omni.models.personaplex.sampling import (
    AudioSampling,
    SamplingRun,
    batched_seeded_exponential_noise,
    batched_seeded_exponential_noises,
    sample_runs,
    sample_token,
)

SPEC = DepformerSpec(
    dim=32, num_heads=4, num_layers=2, ffn_hidden=24, steps=8, input_dim=16
)


def reference_weights(checkpoint_steps: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    dim, ffn = SPEC.dim, SPEC.ffn_hidden
    weights = {"depformer_text_emb.weight": torch.randn(32001, dim)}
    for step in range(checkpoint_steps):
        weights[f"depformer_in.{step}.weight"] = torch.randn(dim, SPEC.input_dim)
        weights[f"linears.{step}.weight"] = torch.randn(AUDIO_CARD, dim)
    for step in range(checkpoint_steps - 1):
        weights[f"depformer_emb.{step}.weight"] = torch.randn(AUDIO_CARD + 1, dim)
    for layer in range(SPEC.num_layers):
        prefix = f"depformer.layers.{layer}"
        weights[f"{prefix}.self_attn.in_proj_weight"] = torch.randn(
            checkpoint_steps * 3 * dim, dim
        )
        weights[f"{prefix}.self_attn.out_proj.weight"] = torch.randn(
            checkpoint_steps * dim, dim
        )
        weights[f"{prefix}.norm1.alpha"] = torch.randn(1, 1, dim)
        weights[f"{prefix}.norm2.alpha"] = torch.randn(1, 1, dim)
        for step in range(checkpoint_steps):
            weights[f"{prefix}.gating.{step}.linear_in.weight"] = torch.randn(
                2 * ffn, dim
            )
            weights[f"{prefix}.gating.{step}.linear_out.weight"] = torch.randn(dim, ffn)
    return weights


def first_steps(
    weights: dict[str, torch.Tensor], steps: int
) -> dict[str, torch.Tensor]:
    """The 8-step checkpoint a 16-step one embeds: per-step blocks sliced, the rest shared."""
    dim = SPEC.dim
    sliced = {}
    for name, value in weights.items():
        if ".self_attn.in_proj_weight" in name:
            sliced[name] = value.view(-1, 3 * dim, dim)[:steps].reshape(-1, dim)
        elif ".self_attn.out_proj.weight" in name:
            sliced[name] = value.view(-1, dim, dim)[:steps].reshape(-1, dim)
        elif (
            name.startswith(("depformer_in.", "linears.", "depformer_emb."))
            or ".gating." in name
        ):
            index = int(
                name.split(".")[1]
                if not ".gating." in name
                else name.split(".gating.")[1].split(".")[0]
            )
            limit = steps - 1 if name.startswith("depformer_emb.") else steps
            if index < limit:
                sliced[name] = value
        else:
            sliced[name] = value
    return sliced


def test_sixteen_step_checkpoint_loads_its_first_eight_steps():
    weights = reference_weights(16)
    sixteen = Depformer(SPEC)
    sixteen.load_reference_weights(weights)
    eight = Depformer(SPEC)
    eight.load_reference_weights(first_steps(weights, 8))
    for name, value in eight.state_dict().items():
        torch.testing.assert_close(sixteen.state_dict()[name], value, atol=0, rtol=0)
    layer = sixteen.layers[0]
    torch.testing.assert_close(
        layer.gate_in_weight[3],
        weights["depformer.layers.0.gating.3.linear_in.weight"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        layer.in_proj_weight[5],
        weights["depformer.layers.0.self_attn.in_proj_weight"].view(
            16, 3 * SPEC.dim, SPEC.dim
        )[5],
        atol=0,
        rtol=0,
    )


def test_forced_codes_are_kept_and_condition_later_steps():
    model = Depformer(SPEC)
    model.load_reference_weights(reference_weights(8))
    greedy = lambda logits: sample_token(logits, AudioSampling(0.0, 0), [None, None])
    text = torch.tensor([3, 3])
    hidden = torch.randn(2, SPEC.input_dim)
    free = torch.full((2, 8), -1, dtype=torch.long)
    forced = free.clone()
    forced[1, 1:] = torch.arange(1, 8)
    codes = model.generate(text, hidden, forced, greedy)
    unforced = model.generate(text, hidden, free, greedy)
    assert codes[1, 1:].tolist() == list(range(1, 8))
    assert codes[0].tolist() == unforced[0].tolist()
    assert codes[1, 0].item() == unforced[1, 0].item()


@torch.inference_mode()
def test_generated_codes_remain_owned_across_calls() -> None:
    model = Depformer(SPEC)
    model.load_reference_weights(reference_weights(SPEC.steps))
    text_tokens = torch.tensor([3, 4])
    hidden_states = torch.randn(2, SPEC.input_dim)
    forced_codes = torch.arange(2 * SPEC.steps).view(2, SPEC.steps)

    def greedy(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    first_codes = model.generate(text_tokens, hidden_states, forced_codes, greedy)
    history_row = first_codes[1]
    next_forced_codes = forced_codes + 2 * SPEC.steps
    next_codes = model.generate(text_tokens, hidden_states, next_forced_codes, greedy)

    torch.testing.assert_close(next_codes, next_forced_codes, rtol=0, atol=0)
    torch.testing.assert_close(first_codes, forced_codes, rtol=0, atol=0)
    torch.testing.assert_close(history_row, forced_codes[1], rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("teacher_forced", [True, False])
@torch.inference_mode()
def test_fused_projections_match_separate_linears(
    batch_size: int, dtype: torch.dtype, teacher_forced: bool
) -> None:
    weights = {
        name: (
            tensor / tensor.shape[-1] ** 0.5
            if tensor.ndim == 2 and "emb" not in name
            else tensor
        )
        for name, tensor in reference_weights(16).items()
    }
    model = Depformer(SPEC).to(dtype=dtype)
    model.load_reference_weights(weights)
    projections = [
        torch.nn.Linear(SPEC.input_dim, SPEC.dim, bias=False, dtype=dtype)
        for _ in range(SPEC.steps)
    ]
    for step, projection in enumerate(projections):
        projection.weight.copy_(weights[f"depformer_in.{step}.weight"])
    text_tokens = torch.arange(batch_size) + 3
    hidden_states = torch.randn(
        batch_size,
        SPEC.input_dim,
        dtype=dtype,
        generator=torch.Generator().manual_seed(7),
    )
    forced_codes = torch.arange(batch_size * SPEC.steps).view(batch_size, SPEC.steps)
    forced_codes = (
        forced_codes
        if teacher_forced
        else torch.where(forced_codes % 3 == 0, -1, forced_codes)
    )

    actual_logits: list[torch.Tensor] = []

    def record_logits(logits: torch.Tensor) -> torch.Tensor:
        actual_logits.append(logits)
        return logits.argmax(dim=-1)

    actual_codes = model.generate(
        text_tokens, hidden_states, forced_codes, record_logits
    )
    caches = [
        hidden_states.new_empty(
            2, batch_size, SPEC.num_heads, SPEC.steps, SPEC.head_dim
        )
        for _ in model.layers
    ]
    previous_tokens = text_tokens
    expected_logits: list[torch.Tensor] = []
    expected_codes: list[torch.Tensor] = []
    for step, projection in enumerate(projections):
        token_embeddings = (
            model.depformer_text_emb(previous_tokens)
            if step == 0
            else model.depformer_emb[step - 1](previous_tokens)
        )
        depth_hidden_states = projection(hidden_states) + token_embeddings
        for layer, cache in zip(model.layers, caches, strict=True):
            depth_hidden_states = layer.step(depth_hidden_states, step, cache)
        logits = model.linears[step](depth_hidden_states).float()
        expected_logits.append(logits)
        previous_tokens = torch.where(
            forced_codes[:, step] >= 0,
            forced_codes[:, step],
            logits.argmax(dim=-1),
        )
        expected_codes.append(previous_tokens)

    torch.testing.assert_close(
        torch.stack(actual_logits, dim=1),
        torch.stack(expected_logits, dim=1),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        actual_codes, torch.stack(expected_codes, dim=1), rtol=0, atol=0
    )


def test_greedy_sampling_is_argmax_and_top_k_stays_inside_k():
    logits = torch.randn(4, 50)
    assert (
        sample_token(logits, AudioSampling(0.0, 25), [None] * 4).tolist()
        == logits.argmax(-1).tolist()
    )
    generator = torch.Generator().manual_seed(7)
    picks = sample_token(logits, AudioSampling(0.8, 3), [generator] * 4)
    top3 = torch.topk(logits, 3).indices
    assert all(pick in top3[row].tolist() for row, pick in enumerate(picks.tolist()))
    again = sample_token(
        logits, AudioSampling(0.8, 3), [torch.Generator().manual_seed(7)] * 4
    )
    assert again.tolist() == picks.tolist()


def test_each_row_draws_from_its_own_generator():
    logits = torch.randn(4, 50)
    sampling = AudioSampling(1.0, 20)

    def alone(row: int, seed: int) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        return [
            int(sample_token(logits[row : row + 1], sampling, [generator]))
            for _ in range(20)
        ]

    generators = [
        torch.Generator().manual_seed(7),
        None,
        None,
        torch.Generator().manual_seed(8),
    ]
    batched = [sample_token(logits, sampling, generators) for _ in range(20)]
    assert [int(picks[0]) for picks in batched] == alone(0, 7)
    assert [int(picks[3]) for picks in batched] == alone(3, 8)


@torch.inference_mode()
def test_cuda_graph_runner_falls_back_on_cpu() -> None:
    model = Depformer(SPEC)
    model.load_reference_weights(reference_weights(8))
    runner = DepformerCudaGraphRunner(model, (1, 4))
    hidden_states = torch.randn(3, SPEC.input_dim)
    text_tokens = torch.arange(3)
    forced_codes = torch.full((3, SPEC.steps), -1, dtype=torch.long)
    sampling = AudioSampling(0.8, 25)
    runs = [
        SamplingRun(row_slice=slice(0, 3), sampling=sampling, generators=[None] * 3)
    ]
    torch.manual_seed(71)
    expected = model.generate(
        text_tokens,
        hidden_states,
        forced_codes,
        lambda logits: sample_runs(logits, runs),
    )
    torch.manual_seed(71)
    actual = runner.generate(text_tokens, hidden_states, forced_codes, runs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not runner.graphs


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_graph_batch_codes_and_rng_match_eager(
    batch_size: int, dtype: torch.dtype
) -> None:
    weights = {
        name: (
            tensor / tensor.shape[-1] ** 0.5
            if tensor.ndim == 2 and "emb" not in name
            else tensor
        )
        for name, tensor in reference_weights(8).items()
    }
    model = Depformer(SPEC).to(device="cuda", dtype=dtype)
    model.load_reference_weights(weights)
    runner = DepformerCudaGraphRunner(model, (1, 2, 4, 8, 16, 32))
    runner.capture_graphs(AudioSampling(0.8, 25), max_batch_size=batch_size)
    hidden_states = torch.randn(batch_size, SPEC.input_dim, device="cuda", dtype=dtype)
    text_tokens = torch.arange(batch_size, device="cuda")
    forced_codes = torch.full(
        (batch_size, SPEC.steps), -1, dtype=torch.long, device="cuda"
    )
    eager_generators = [
        torch.Generator(device="cuda").manual_seed(7 + row) if row % 4 < 2 else None
        for row in range(batch_size)
    ]
    graph_generators = [
        torch.Generator(device="cuda").manual_seed(7 + row) if row % 4 < 2 else None
        for row in range(batch_size)
    ]
    old_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frame in range(4):
        # note (Kokoro2336): Reordering and shrinking exercise request lifecycle and padded replay.
        rows = list(range(batch_size))[:: -1 if frame % 2 else 1]
        rows = rows[: max(1, batch_size - frame)]
        sampling = AudioSampling(0.8, 25)
        graph_runs = [
            SamplingRun(
                row_slice=slice(0, len(rows)),
                sampling=sampling,
                generators=[graph_generators[row] for row in rows],
            )
        ]
        bucket_size = next(size for size in runner.batch_sizes if size >= len(rows))
        assert ((bucket_size, sampling),) in runner.graphs
        hidden_states.normal_()
        forced_codes[:, frame] = frame + 13
        padded_text_tokens = torch.zeros(
            bucket_size, dtype=text_tokens.dtype, device="cuda"
        )
        padded_text_tokens[: len(rows)].copy_(text_tokens[rows])
        padded_hidden_states = torch.zeros(
            bucket_size, SPEC.input_dim, dtype=dtype, device="cuda"
        )
        padded_hidden_states[: len(rows)].copy_(hidden_states[rows])
        padded_forced_codes = torch.full(
            (bucket_size, SPEC.steps), -1, dtype=torch.long, device="cuda"
        )
        padded_forced_codes[: len(rows)].copy_(forced_codes[rows])

        def sample_padded(logits: torch.Tensor) -> torch.Tensor:
            actual = sample_token(
                logits[: len(rows)],
                sampling,
                [eager_generators[row] for row in rows],
            )
            if len(rows) < bucket_size:
                return torch.cat((actual, logits[len(rows) :].argmax(dim=-1)))
            else:
                return actual

        random_state = torch.cuda.get_rng_state()
        expected = model.generate(
            padded_text_tokens,
            padded_hidden_states,
            padded_forced_codes,
            sample_padded,
        )[: len(rows)]
        expected_random_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(random_state)
        actual = runner.generate(
            text_tokens[rows], hidden_states[rows], forced_codes[rows], graph_runs
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.equal(torch.cuda.get_rng_state(), expected_random_state)
        for eager_generator, graph_generator in zip(
            eager_generators, graph_generators, strict=True
        ):
            if eager_generator is None:
                pass
            else:
                assert torch.equal(
                    eager_generator.get_state(), graph_generator.get_state()
                )
        old_outputs.append((actual, actual.clone()))
    assert runner.graphs and all(graph is not None for graph in runner.graphs.values())
    for output, snapshot in old_outputs:
        torch.testing.assert_close(output, snapshot, atol=0, rtol=0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fallback", ["disabled", "oversized"])
@torch.inference_mode()
def test_graph_supported_fallbacks(fallback: str) -> None:
    model = Depformer(SPEC).cuda()
    model.load_reference_weights(reference_weights(8))
    runner = DepformerCudaGraphRunner(model, () if fallback == "disabled" else (1,))
    batch_size = 2 if fallback == "oversized" else 1
    sampling = AudioSampling(0.8, 25)
    if fallback == "oversized":
        runner.capture_graphs(sampling, max_batch_size=1)
    else:
        pass
    hidden_states = torch.randn(batch_size, SPEC.input_dim, device="cuda")
    text_tokens = torch.arange(batch_size, device="cuda")
    forced_codes = torch.full(
        (batch_size, SPEC.steps), -1, dtype=torch.long, device="cuda"
    )
    runs = [
        SamplingRun(
            row_slice=slice(0, batch_size),
            sampling=sampling,
            generators=[None] * batch_size,
        )
    ]
    torch.cuda.manual_seed(81)
    expected = model.generate(
        text_tokens,
        hidden_states,
        forced_codes,
        lambda logits: sample_runs(logits, runs),
    )
    torch.cuda.manual_seed(81)
    actual = runner.generate(text_tokens, hidden_states, forced_codes, runs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_batched_seeded_noise_accepts_every_codebook_position() -> None:
    """The second codebook must not become an invalid specialized scalar."""
    logits = torch.zeros(2, AUDIO_CARD, device="cuda", dtype=torch.bfloat16)
    seeds = torch.tensor([7, 8], device="cuda")
    frame_positions = torch.tensor([11, 12], device="cuda")

    noises = batched_seeded_exponential_noises(
        logits, seeds, frame_positions, codebook_count=SPEC.steps
    )
    individual_noises = torch.stack(
        [
            batched_seeded_exponential_noise(
                logits, seeds, frame_positions, codebook_position
            )
            for codebook_position in range(SPEC.steps)
        ]
    )

    assert torch.equal(noises, individual_noises)
    assert all(torch.isfinite(noise).all() and (noise > 0).all() for noise in noises)
    assert all(
        not torch.equal(previous, current)
        for previous, current in zip(noises[:-1], noises[1:], strict=True)
    )
