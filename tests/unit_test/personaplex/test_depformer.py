# SPDX-License-Identifier: Apache-2.0
"""Depformer weight slicing, teacher forcing, fused inference and eager fallbacks."""

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
from sglang_omni.models.personaplex.sampling import AudioSampling, sample_token

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


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
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


@pytest.mark.accelerator
@CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 7, 8, 15, 16, 32])
@torch.inference_mode()
def test_fused_pointwise_preserves_rounding_stream_and_graph(
    dtype: torch.dtype, batch_size: int
) -> None:
    torch.manual_seed(42)
    execution_stream = torch.cuda.Stream()
    execution_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(execution_stream):
        initial_hidden_states = torch.randn(
            batch_size, DEPFORMER.dim, device="cuda", dtype=dtype
        )
        alpha = torch.linspace(-2, 2, DEPFORMER.dim, device="cuda", dtype=torch.float32)
        gate_up = torch.randn(
            batch_size, 2 * DEPFORMER.ffn_hidden, device="cuda", dtype=dtype
        )
        activated_states = depformer_module.fused_silu_gate(gate_up)
        torch.testing.assert_close(activated_states, silu_gate(gate_up), rtol=0, atol=0)
        if dtype != torch.float32:
            assert not torch.equal(
                silu_gate(gate_up), silu_gate(gate_up.float()).to(dtype)
            )
        else:
            pass
        for hidden_scale in (0, 1e-6, 1, 100):
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
@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
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
    with (
        torch.no_grad(),
        patch.multiple(
            depformer_module,
            rms_norm_f32=Mock(side_effect=AssertionError),
            silu_gate=Mock(side_effect=AssertionError),
        ),
    ):
        fused_codes = model.generate(text_tokens, hidden_states, forced_codes, sample)
        fused_logits = torch.stack(captured_logits)
    captured_logits.clear()
    with torch.enable_grad():
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
    "layer_spec,batch_size,strides",
    [
        (replace(DEPFORMER, dim=SPEC.dim), 1, (1, 1, 1)),
        (replace(DEPFORMER, ffn_hidden=SPEC.ffn_hidden), 1, (1, 1, 1)),
        (DEPFORMER, 0, (1, 1, 1)),
        (DEPFORMER, 2, (2, 1, 1)),
        (DEPFORMER, 2, (1, 2, 1)),
        (DEPFORMER, 2, (1, 1, 2)),
    ],
)
def test_cuda_layer_keeps_eager_outputs_for_unsupported_inputs(
    layer_spec: DepformerSpec, batch_size: int, strides: tuple[int, int, int]
) -> None:
    layer = DepformerLayer(replace(layer_spec, steps=1)).cuda().requires_grad_(False)
    for parameter in layer.parameters():
        nn.init.normal_(parameter, std=0.05)
    for alpha, stride in (
        (layer.norm1_alpha, strides[1]),
        (layer.norm2_alpha, strides[2]),
    ):
        alpha.set_(alpha.repeat_interleave(stride)[::stride])
    hidden_states = torch.randn(batch_size, layer_spec.dim * strides[0], device="cuda")
    hidden_states = hidden_states[:, :: strides[0]]
    cache = hidden_states.new_empty(
        2, batch_size, layer_spec.num_heads, 1, layer_spec.head_dim
    )
    with torch.enable_grad():
        expected = layer.step(hidden_states, 0, cache)
    with (
        torch.inference_mode(),
        patch.multiple(
            depformer_module,
            fused_rms_norm_f32=Mock(side_effect=AssertionError),
            fused_silu_gate=Mock(side_effect=AssertionError),
        ),
    ):
        actual = layer.step(hidden_states, 0, cache)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.accelerator, CUDA_ONLY])]
)
def test_layer_keeps_eager_gradients(device: str) -> None:
    layer_spec = replace(DEPFORMER, steps=1) if device == "cuda" else SPEC
    layer = DepformerLayer(layer_spec).to(device)
    for parameter in layer.parameters():
        nn.init.normal_(parameter, std=0.05)
    hidden_states = torch.randn(1, layer_spec.dim, device=device, requires_grad=True)
    cache = hidden_states.new_empty(
        2, 1, layer_spec.num_heads, layer_spec.steps, layer_spec.head_dim
    )
    with patch.multiple(
        depformer_module,
        fused_rms_norm_f32=Mock(side_effect=AssertionError),
        fused_silu_gate=Mock(side_effect=AssertionError),
        create=True,
    ):
        layer.step(hidden_states, 0, cache).sum().backward()
    for gradient in (
        hidden_states.grad,
        layer.norm1_alpha.grad,
        layer.gate_in_weight.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
