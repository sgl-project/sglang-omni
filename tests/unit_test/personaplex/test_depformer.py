# SPDX-License-Identifier: Apache-2.0
"""Depformer weight slicing and teacher forcing, on a scaled-down spec."""

import pytest
import torch

from sglang_omni.models.personaplex.architecture import AUDIO_CARD, DepformerSpec
from sglang_omni.models.personaplex.components.depformer import Depformer
from sglang_omni.models.personaplex.components.depformer_cuda_graph import (
    DepformerCudaGraphRunner,
)
from sglang_omni.models.personaplex.sampling import AudioSampling, sample_token

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
    greedy = lambda logits: sample_token(logits, AudioSampling(0.0, 0))
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


def test_greedy_sampling_is_argmax_and_top_k_stays_inside_k():
    logits = torch.randn(4, 50)
    assert (
        sample_token(logits, AudioSampling(0.0, 25)).tolist()
        == logits.argmax(-1).tolist()
    )
    generator = torch.Generator().manual_seed(7)
    picks = sample_token(logits, AudioSampling(0.8, 3), generator)
    top3 = torch.topk(logits, 3).indices
    assert all(pick in top3[row].tolist() for row, pick in enumerate(picks.tolist()))
    again = sample_token(
        logits, AudioSampling(0.8, 3), torch.Generator().manual_seed(7)
    )
    assert again.tolist() == picks.tolist()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "sampling", [AudioSampling(0.0, 0), AudioSampling(0.8, 7), AudioSampling(0.8, 0)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cuda_graph_codes_and_request_rng_match_eager(
    sampling: AudioSampling, dtype: torch.dtype
) -> None:
    model = Depformer(SPEC).to(device="cuda", dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.normal_(std=0.05)
    runner = DepformerCudaGraphRunner(model, (1, 4))
    eager_generators = [
        torch.Generator(device="cuda").manual_seed(seed) for seed in (7, 9)
    ]
    graph_generators = [
        torch.Generator(device="cuda").manual_seed(seed) for seed in (7, 9)
    ]
    retained_codes: list[tuple[torch.Tensor, torch.Tensor]] = []

    for request_index, batch_size in ((0, 1), (1, 4), (0, 3), (1, 1), (0, 1)):
        text_tokens = torch.arange(batch_size, device="cuda") + 3
        hidden_states = torch.randn(
            batch_size, SPEC.input_dim, device="cuda", dtype=dtype
        )
        forced_codes = torch.full((batch_size, SPEC.steps), -1, device="cuda")
        forced_codes[0, 0] = 17
        forced_codes[-1, 2:4] = 23
        expected_codes = model.generate(
            text_tokens,
            hidden_states,
            forced_codes,
            lambda logits: sample_token(
                logits, sampling, eager_generators[request_index]
            ),
        )
        actual_codes = runner.generate(
            text_tokens,
            hidden_states,
            forced_codes,
            sampling,
            graph_generators[request_index],
        )
        torch.testing.assert_close(actual_codes, expected_codes, atol=0, rtol=0)
        assert torch.equal(
            eager_generators[request_index].get_state(),
            graph_generators[request_index].get_state(),
        )
        retained_codes.append((actual_codes, actual_codes.clone()))

    assert runner.graphs and all(
        captured is not None for captured in runner.graphs.values()
    )
    for actual_codes, saved_codes in retained_codes:
        torch.testing.assert_close(actual_codes, saved_codes, atol=0, rtol=0)


@pytest.mark.parametrize("batch_sizes", [(), (1,), (1, 4)])
@torch.inference_mode()
def test_cuda_graph_runner_falls_back_on_cpu(batch_sizes: tuple[int, ...]) -> None:
    model = Depformer(SPEC).eval()
    model.load_reference_weights(reference_weights(SPEC.steps))
    runner = DepformerCudaGraphRunner(model, batch_sizes)
    sampling = AudioSampling(0.8, 7)
    text_tokens = torch.tensor([3, 4])
    hidden_states = torch.randn(2, SPEC.input_dim)
    forced_codes = torch.full((2, SPEC.steps), -1)
    expected_generator = torch.Generator().manual_seed(7)
    actual_generator = torch.Generator().manual_seed(7)
    expected_codes = model.generate(
        text_tokens,
        hidden_states,
        forced_codes,
        lambda logits: sample_token(logits, sampling, expected_generator),
    )
    actual_codes = runner.generate(
        text_tokens, hidden_states, forced_codes, sampling, actual_generator
    )
    torch.testing.assert_close(actual_codes, expected_codes, atol=0, rtol=0)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())
