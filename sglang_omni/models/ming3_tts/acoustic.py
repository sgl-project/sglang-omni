# SPDX-License-Identifier: Apache-2.0
"""One Ming 3 TTS acoustic step: flow sampling, latent feedback, and stop head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.kernels.ops.layernorm.rmsnorm_hf import (
    is_supported_rmsnorm_hf_hidden_size,
    rmsnorm_hf,
)
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from sglang_omni.models.ming3_tts.kernels import (
    HAS_TRITON,
    add_rmsnorm,
    assemble_flow_states,
    cfg_solver_step,
    rotary_attention,
)
from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.dit import DiT
from sglang_omni.modules.ming_audio.flow.flow_loss import FlowLoss, expand_batch_param
from sglang_omni.modules.ming_audio.flow.modules import DiTBlock, FeedForward, RMSNorm
from sglang_omni.platforms import current_platform

FLOW_STEPS = 10
# The flow solver's default sway sampling coefficient.
SWAY_SAMPLING_COEFFICIENT = -1.0


@dataclass(kw_only=True)
class AcousticInputs:
    """Batch-major step inputs; solver_noise is [flow steps - 1, batch, 1, latent]."""

    hidden_states: torch.Tensor
    latent_history: torch.Tensor
    initial_noise: torch.Tensor
    solver_noise: torch.Tensor
    temperatures: torch.Tensor


@dataclass(kw_only=True)
class AcousticStep:
    latents: torch.Tensor
    feedback_embeddings: torch.Tensor
    stop_probabilities: torch.Tensor


@dataclass(kw_only=True)
class RotaryCache:
    """fp32 [length, dim] rows: the cosines, then the sines, of each token's pairs."""

    cos_sin_cache: torch.Tensor


@dataclass(kw_only=True)
class PackedProjection:
    """One block's query, key, and value projections as a single matrix."""

    weight: torch.Tensor
    bias: torch.Tensor


@dataclass(kw_only=True)
class FlowConstants:
    """Step inputs that depend only on the batch size, so one capture computes them once."""

    timestep_embeddings: torch.Tensor
    step_sizes: torch.Tensor
    noise_step_scales: torch.Tensor
    flow_rotary: RotaryCache
    aggregator_rotary: RotaryCache
    aggregator_class_rows: torch.Tensor
    flow_projections: list[PackedProjection]
    aggregator_projections: list[PackedProjection]


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def is_tanh_gelu_mlp(mlp: FeedForward) -> bool:
    project_in, _dropout, project_out = mlp.ff
    projection, activation = project_in
    return (
        type(projection) is nn.Linear
        and type(activation) is nn.GELU
        and activation.approximate == "tanh"
        and type(project_out) is nn.Linear
    )


def is_plain_rotary_transformer(module: DiT | Aggregator) -> bool:
    """Whether every block is full-head rotary attention, RMSNorm and a tanh-GELU MLP."""
    width = module.final_layer.norm_final.weight.shape[0]
    return (
        type(module.rotary_embed) is RotaryEmbedding
        and type(module.final_layer.norm_final) is RMSNorm
        and is_power_of_two(width)
        and is_supported_rmsnorm_hf_hidden_size(width)
        and all(
            block.attn.attn_backend == "torch"
            and block.attn.to_qkv is None
            and block.attn.q_norm is None
            and block.attn.k_norm is None
            and block.attn.pe_attn_head is None
            and block.attn.inner_dim == width
            and is_power_of_two(width // block.attn.heads)
            and width // block.attn.heads >= 2
            and type(block.norm1) is RMSNorm
            and type(block.norm2) is RMSNorm
            and is_tanh_gelu_mlp(block.mlp)
            for block in module.blocks
        )
    )


def pack_projections(module: DiT | Aggregator) -> list[PackedProjection]:
    """Move each block's q, k, v parameters into one matrix they then view.

    In-place weight loads keep reaching the packed copy because the
    projection parameters become slices of it.
    """
    packed: list[PackedProjection] = []
    for block in module.blocks:
        projections = (block.attn.to_q, block.attn.to_k, block.attn.to_v)
        weight = torch.cat([projection.weight.data for projection in projections])
        bias = torch.cat([projection.bias.data for projection in projections])
        for projection, weight_rows, bias_rows in zip(
            projections, weight.chunk(3), bias.chunk(3), strict=True
        ):
            projection.weight.data = weight_rows
            projection.bias.data = bias_rows
        packed.append(PackedProjection(weight=weight, bias=bias))
    return packed


def build_rotary_cache(
    rotary_embedding: RotaryEmbedding, sequence_length: int
) -> RotaryCache:
    """The rotary angles of rotate_half's interleaved pairs, laid out for the kernel."""
    frequencies, _scale = rotary_embedding.forward_from_seq_len(sequence_length)
    phase = frequencies[0, :, 0::2]
    return RotaryCache(
        cos_sin_cache=torch.cat([phase.cos(), phase.sin()], dim=-1).contiguous()
    )


def normalize(norm: RMSNorm, states: torch.Tensor) -> torch.Tensor:
    """RMSNorm that casts before its weight multiply, as Ming's RMSNorm does here."""
    return rmsnorm_hf(states.reshape(-1, states.shape[-1]), norm.weight, norm.eps).view(
        states.shape
    )


def run_block(
    block: DiTBlock,
    projection: PackedProjection,
    states: torch.Tensor,
    normed: torch.Tensor,
    rotary: RotaryCache,
    next_norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One block from its input and that input's norm1; returns the output and its next norm."""
    attention = block.attn
    attended = rotary_attention(
        F.linear(normed, projection.weight, projection.bias),
        rotary.cos_sin_cache,
        attention.heads,
    )
    states, normed = add_rmsnorm(
        states, attention.to_out[0](attended), block.norm2.weight, block.norm2.eps
    )
    (project_in, _activation), _dropout, project_out = block.mlp.ff
    # note (rsi): the up projection's bias and tanh GELU run in the GEMM epilogue.
    hidden = (
        torch._addmm_activation(  # noqa: leading-underscore - PyTorch operator name
            project_in.bias,
            normed.view(-1, normed.shape[-1]),
            project_in.weight.t(),
            use_gelu=True,
        )
    )
    update = F.linear(hidden, project_out.weight, project_out.bias).view(states.shape)
    return add_rmsnorm(states, update, next_norm.weight, next_norm.eps)


def run_blocks(
    module: DiT | Aggregator,
    projections: list[PackedProjection],
    states: torch.Tensor,
    rotary: RotaryCache,
) -> torch.Tensor:
    """All blocks of a module; returns the final layer's normed input."""
    blocks = module.blocks
    normed = normalize(blocks[0].norm1, states)
    next_norms = [block.norm1 for block in blocks[1:]] + [module.final_layer.norm_final]
    for block, projection, next_norm in zip(
        blocks, projections, next_norms, strict=True
    ):
        states, normed = run_block(block, projection, states, normed, rotary, next_norm)
    return normed


def build_flow_constants(
    flowloss: FlowLoss,
    aggregator: Aggregator,
    flow_timesteps: torch.Tensor,
    batch_size: int,
    flow_projections: list[PackedProjection],
    aggregator_projections: list[PackedProjection],
) -> FlowConstants:
    dit = flowloss.cfm.model
    device = flow_timesteps.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        timesteps = flow_timesteps + SWAY_SAMPLING_COEFFICIENT * (
            torch.cos(torch.pi / 2 * flow_timesteps) - 1 + flow_timesteps
        )
        step_sizes = timesteps[1:] - timesteps[:-1]
        return FlowConstants(
            timestep_embeddings=torch.stack(
                [
                    dit.t_embedder(timesteps[step].repeat(2 * batch_size)).unsqueeze(1)
                    for step in range(FLOW_STEPS)
                ]
            ),
            step_sizes=step_sizes,
            noise_step_scales=abs(step_sizes) ** 0.5,
            flow_rotary=build_rotary_cache(dit.rotary_embed, 3),
            aggregator_rotary=build_rotary_cache(aggregator.rotary_embed, 2),
            aggregator_class_rows=aggregator.word_embedder(
                torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            ),
            flow_projections=flow_projections,
            aggregator_projections=aggregator_projections,
        )


def can_fuse_flow_head(flowloss: FlowLoss, aggregator: Aggregator) -> bool:
    """Whether the fused loop reproduces both modules; otherwise they run themselves."""
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and is_power_of_two(flowloss.cfm.model.x_embedder.in_features)
        and is_plain_rotary_transformer(flowloss.cfm.model)
        and is_plain_rotary_transformer(aggregator)
    )


def compute_acoustic_step(
    flowloss: FlowLoss,
    aggregator: Aggregator,
    stop_head: nn.Linear,
    flow_timesteps: torch.Tensor,
    inputs: AcousticInputs,
    sampling_scales: torch.Tensor,
    constants: FlowConstants | None,
) -> AcousticStep:
    """Sample one latent per row; sampling_scales holds [guidance, noise] per row."""
    hidden_states = inputs.hidden_states
    conditioning = hidden_states.unsqueeze(1)
    with torch.autocast(device_type=hidden_states.device.type, dtype=torch.bfloat16):
        if constants is None:
            latents = flowloss.sample(
                z=conditioning,
                latent_history=inputs.latent_history,
                noise=inputs.initial_noise,
                timesteps=flow_timesteps,
                sde_random=inputs.solver_noise,
                cfg=sampling_scales[:, 0],
                sigma=sampling_scales[:, 1],
                temperature=inputs.temperatures,
            )
            feedback_embeddings = aggregator(latents)
        else:
            latents = sample_flow(
                flowloss.cfm.model, inputs, sampling_scales, constants
            )
            states = torch.cat(
                [constants.aggregator_class_rows, aggregator.x_embedder(latents)], dim=1
            )
            feedback_embeddings = aggregator.final_layer.linear(
                run_blocks(
                    aggregator,
                    constants.aggregator_projections,
                    states,
                    constants.aggregator_rotary,
                )
            )[:, :1, :]
        stop_logits = stop_head(conditioning)
    return AcousticStep(
        latents=latents,
        feedback_embeddings=feedback_embeddings.reshape(hidden_states.shape),
        stop_probabilities=stop_logits.softmax(dim=-1)[:, 0, 1],
    )


def sample_flow(
    dit: DiT,
    inputs: AcousticInputs,
    sampling_scales: torch.Tensor,
    constants: FlowConstants,
) -> torch.Tensor:
    """FlowLoss.sample's guided solve, with batch-only work hoisted out of the loop."""
    noise = inputs.initial_noise
    batch_size = noise.shape[0]
    guidance, sigma, temperature = (
        expand_batch_param(
            scale, batch_size=batch_size, device=noise.device, dtype=noise.dtype
        )
        for scale in (sampling_scales[:, 0], sampling_scales[:, 1], inputs.temperatures)
    )
    noise_scale = (sigma * (temperature**0.5)).contiguous()
    guidance = guidance.contiguous()
    conditioning = inputs.hidden_states.unsqueeze(1)
    condition_embedding = dit.c_embedder(
        torch.cat([conditioning, torch.zeros_like(conditioning)], dim=0), dit.training
    )
    # note (rsi): the history token is the same in every solver step and both
    # guidance halves share the history and sample rows, so only the sample
    # row is embedded per step.
    token_rows = constants.timestep_embeddings + condition_embedding
    history_rows = dit.x_embedder(inputs.latent_history)
    sample = noise.transpose(1, 2)
    for step in range(FLOW_STEPS):
        states = assemble_flow_states(
            token_rows[step], history_rows, dit.x_embedder(sample)
        )
        normed = run_blocks(
            dit, constants.flow_projections, states, constants.flow_rotary
        )
        noise_step = min(step, FLOW_STEPS - 2)
        sample = cfg_solver_step(
            dit.final_layer.linear(normed[:, -1:, :]),
            sample,
            guidance,
            noise_scale,
            constants.step_sizes[step : step + 1],
            constants.noise_step_scales[noise_step : noise_step + 1],
            inputs.solver_noise[noise_step],
            add_noise=step + 1 < FLOW_STEPS,
        )
    return sample
