# SPDX-License-Identifier: Apache-2.0
"""Depformer weight routing, sampling, and CUDA graph parity."""

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

import sglang_omni.models.personaplex.components.depformer as depformer_module
from sglang_omni.models.personaplex.architecture import (
    AUDIO_CARD,
    DEPFORMER,
    DepformerSpec,
)
from sglang_omni.models.personaplex.components.depformer import (
    Depformer,
    DepformerLayer,
    rms_norm_f32,
    silu_gate,
)
from sglang_omni.models.personaplex.components.depformer_cuda_graph import (
    DepformerCudaGraphRunner,
)
from sglang_omni.models.personaplex.sampling import (
    AudioSampling,
    SamplingRun,
    sample_runs,
    sample_token,
)

CUDA_ONLY = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.cuda is not None),
    reason="NVIDIA CUDA is unavailable",
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
@pytest.mark.parametrize(
    "model_spec,batch_size",
    [
        pytest.param(SPEC, batch_size, id=f"scaled-batch-{batch_size}")
        for batch_size in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32)
    ]
    + [
        pytest.param(
            replace(DEPFORMER, num_layers=1, input_dim=SPEC.input_dim),
            batch_size,
            id=f"production-batch-{batch_size}",
            marks=CUDA_ONLY,
        )
        for batch_size in (1, 8)
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_graph_batch_codes_and_rng_match_eager(
    model_spec: DepformerSpec, batch_size: int, dtype: torch.dtype
) -> None:
    """Graph replay preserves eager codes and RNG at scaled and production dimensions."""
    torch.manual_seed(42)
    model = Depformer(model_spec).to(device="cuda", dtype=dtype)
    runner = DepformerCudaGraphRunner(model, (1, 2, 4, 8, 16, 32))
    if model_spec == SPEC:
        weights = {
            name: (
                tensor / tensor.shape[-1] ** 0.5
                if tensor.ndim == 2 and "emb" not in name
                else tensor
            )
            for name, tensor in reference_weights(8).items()
        }
        model.load_reference_weights(weights)
        runner.capture_graphs(AudioSampling(0.8, 25), max_batch_size=batch_size)
    else:
        for parameter in model.parameters():
            nn.init.normal_(parameter, std=0.05)
        with patch.multiple(
            depformer_module,
            rms_norm_f32=Mock(side_effect=AssertionError),
            silu_gate=Mock(side_effect=AssertionError),
        ):
            model.warmup_pointwise()
            runner.capture_graphs(AudioSampling(0.8, 25), max_batch_size=batch_size)
    hidden_states = torch.randn(
        batch_size, model_spec.input_dim, device="cuda", dtype=dtype
    )
    text_tokens = torch.arange(batch_size, device="cuda")
    forced_codes = torch.full(
        (batch_size, model_spec.steps), -1, dtype=torch.long, device="cuda"
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
            bucket_size, model_spec.input_dim, dtype=dtype, device="cuda"
        )
        padded_hidden_states[: len(rows)].copy_(hidden_states[rows])
        padded_forced_codes = torch.full(
            (bucket_size, model_spec.steps), -1, dtype=torch.long, device="cuda"
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
        with patch.multiple(
            depformer_module,
            fused_rms_norm_f32=rms_norm_f32,
            fused_silu_gate=silu_gate,
        ):
            expected = model.generate(
                padded_text_tokens,
                padded_hidden_states,
                padded_forced_codes,
                sample_padded,
            )[: len(rows)]
        expected_random_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(random_state)
        with patch.object(model, "generate", side_effect=AssertionError):
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
@CUDA_ONLY
@torch.inference_mode()
def test_startup_warmup_allows_graph_capture_with_mixed_norm_dtypes() -> None:
    model = Depformer(
        replace(DEPFORMER, num_layers=1, steps=1, input_dim=SPEC.input_dim)
    ).to(device="cuda", dtype=torch.bfloat16)
    model.requires_grad_(False)
    layer = model.layers[0]
    layer.norm1_alpha = nn.Parameter(layer.norm1_alpha.float(), requires_grad=False)
    layer.norm2_alpha = nn.Parameter(
        layer.norm2_alpha.to(torch.float16), requires_grad=False
    )
    model.warmup_pointwise()
    torch.cuda.synchronize()
    for batch_size in (3, 7, 15, 33):
        hidden_states = torch.randn(
            batch_size, DEPFORMER.dim, device="cuda", dtype=torch.bfloat16
        )
        gate_up = torch.randn(
            batch_size, 2 * DEPFORMER.ffn_hidden, device="cuda", dtype=torch.bfloat16
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for alpha in (layer.norm1_alpha, layer.norm2_alpha):
                depformer_module.fused_rms_norm_f32(
                    hidden_states, alpha, model.spec.rms_norm_eps
                )
            depformer_module.fused_silu_gate(gate_up)
        graph.replay()
        torch.cuda.synchronize()


@pytest.mark.accelerator
@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch_size,alpha_dtype",
    [
        (1, torch.float32),
        (2, torch.bfloat16),
        (3, torch.float16),
        (4, torch.float32),
        (8, torch.float32),
        (16, torch.float32),
    ],
)
@torch.inference_mode()
def test_fused_pointwise_preserves_rounding_stream_and_graph(
    dtype: torch.dtype, alpha_dtype: torch.dtype, batch_size: int
) -> None:
    torch.manual_seed(42)
    execution_stream = torch.cuda.Stream()
    execution_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(execution_stream):
        initial_hidden_states = torch.randn(
            batch_size, DEPFORMER.dim, device="cuda", dtype=dtype
        )
        alpha = torch.linspace(-2, 2, DEPFORMER.dim, device="cuda").to(alpha_dtype)
        gate_up = torch.randn(
            batch_size, 2 * DEPFORMER.ffn_hidden, device="cuda", dtype=dtype
        )
        if dtype == torch.float32:
            gate_up[0, 0] = torch.finfo(dtype).tiny
            gate_up[0, DEPFORMER.ffn_hidden] = 0.5
        else:
            pass
        activated_states = depformer_module.fused_silu_gate(gate_up)
        expected_activated_states = silu_gate(gate_up)
        torch.testing.assert_close(
            activated_states, expected_activated_states, rtol=0, atol=0
        )
        hidden_scales = (
            (0, 1e-40, 1e-6, 1, 100) if dtype == torch.float32 else (0, 1e-6, 1, 100)
        )
        for hidden_scale in hidden_scales:
            hidden_states = initial_hidden_states * hidden_scale
            normalized_states = depformer_module.fused_rms_norm_f32(
                hidden_states, alpha, DEPFORMER.rms_norm_eps
            )
            torch.testing.assert_close(
                normalized_states,
                rms_norm_f32(hidden_states, alpha, DEPFORMER.rms_norm_eps),
                rtol=0,
                atol=0,
                msg=f"RMSNorm scale={hidden_scale}",
            )
        execution_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=execution_stream):
            replayed_norm = depformer_module.fused_rms_norm_f32(
                hidden_states, alpha, DEPFORMER.rms_norm_eps
            )
            replayed_gate = depformer_module.fused_silu_gate(gate_up)
        hidden_states.mul_(0.5).add_(0.25)
        alpha.neg_()
        gate_up.neg_()
        graph.replay()
    execution_stream.synchronize()
    expected_norm = rms_norm_f32(hidden_states, alpha, DEPFORMER.rms_norm_eps)
    torch.testing.assert_close(replayed_norm, expected_norm, rtol=0, atol=0)
    torch.testing.assert_close(replayed_gate, silu_gate(gate_up), rtol=0, atol=0)


@pytest.mark.accelerator
@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch_size", [1, 3])
@torch.inference_mode()
def test_cuda_depformer_matches_eager_logits_and_codes(
    dtype: torch.dtype, batch_size: int
) -> None:
    torch.manual_seed(42)
    model_spec = replace(DEPFORMER, num_layers=1, input_dim=SPEC.input_dim)
    model = Depformer(model_spec).to(device="cuda", dtype=dtype)
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        nn.init.normal_(parameter, std=0.05)
    text_tokens = torch.arange(batch_size, device="cuda") + 3
    hidden_states = torch.randn(
        batch_size, model_spec.input_dim, device="cuda", dtype=dtype
    )
    forced_codes = torch.full(
        (batch_size, model_spec.steps), -1, device="cuda", dtype=torch.long
    )
    captured_logits: list[torch.Tensor] = []

    def sample(logits: torch.Tensor) -> torch.Tensor:
        captured_logits.append(logits)
        return logits.argmax(-1)

    model.warmup_pointwise()
    with patch.multiple(
        depformer_module,
        rms_norm_f32=Mock(side_effect=AssertionError),
        silu_gate=Mock(side_effect=AssertionError),
    ):
        fused_codes = model.generate(text_tokens, hidden_states, forced_codes, sample)
        fused_logits = torch.stack(captured_logits)
    captured_logits.clear()
    with patch.multiple(
        depformer_module,
        fused_rms_norm_f32=rms_norm_f32,
        fused_silu_gate=silu_gate,
    ):
        eager_codes = model.generate(text_tokens, hidden_states, forced_codes, sample)
    torch.testing.assert_close(fused_codes, eager_codes, rtol=0, atol=0)
    torch.testing.assert_close(
        fused_logits,
        torch.stack(captured_logits),
        rtol=2 * torch.finfo(dtype).eps,
        atol=1e-6,
    )


@pytest.mark.accelerator
@CUDA_ONLY
@pytest.mark.parametrize(
    "layer_spec",
    [
        replace(DEPFORMER, dim=SPEC.dim),
        replace(DEPFORMER, ffn_hidden=SPEC.ffn_hidden),
    ],
)
@torch.inference_mode()
def test_cuda_layer_keeps_eager_outputs_for_unsupported_inputs(
    layer_spec: DepformerSpec,
) -> None:
    layer = DepformerLayer(replace(layer_spec, steps=1)).cuda()
    for parameter in layer.parameters():
        nn.init.normal_(parameter, std=0.05)
    hidden_states = torch.randn(1, layer_spec.dim, device="cuda")
    cache = hidden_states.new_empty(2, 1, layer_spec.num_heads, 1, layer_spec.head_dim)
    expected = layer.step(hidden_states, 0, cache.clone())
    with patch.multiple(
        depformer_module,
        fused_rms_norm_f32=Mock(side_effect=AssertionError),
        fused_silu_gate=Mock(side_effect=AssertionError),
    ):
        actual = layer.step(hidden_states, 0, cache)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
