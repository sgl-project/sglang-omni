# SPDX-License-Identifier: Apache-2.0
"""Build and execute batched flow requests while preserving request order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from sglang_omni.models.firered_tts3.flow_execution import euler_flow_loop
from sglang_omni.models.firered_tts3.flow_graph import FlowGraphRunner
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.request_builders import (
    FireRedTTS3SGLangRequestData,
    FlowState,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel


@dataclass(kw_only=True)
class FlowRow:
    """One request's inputs to a batched Euler integration."""

    request_data: FireRedTTS3SGLangRequestData
    history_latents: torch.Tensor
    initial_noise: torch.Tensor
    conditioning: torch.Tensor


class TimestepScheduleProvider(Protocol):
    def __call__(self, n_timesteps: int, device: torch.device) -> torch.Tensor: ...


def build_flow_row(
    model: FireRedTTS3SGLangModel,
    request_data: FireRedTTS3SGLangRequestData,
    hidden_states_float32: torch.Tensor,
) -> FlowRow:
    """Append conditioning and draw noise from this request generator."""
    flow_state = request_data.flow_state
    assert isinstance(flow_state, FlowState)
    assert request_data.probe is not None
    with torch.inference_mode(), request_data.probe.segment("cond"):
        if flow_state.patch_count == 0 and flow_state.prompt_patch_count > 0:
            conditioning_rows = hidden_states_float32[-flow_state.prompt_patch_count :]
        else:
            conditioning_rows = hidden_states_float32[-1:]
        flow_state.backbone_conditioning = torch.cat(
            [flow_state.backbone_conditioning, conditioning_rows.unsqueeze(0)], dim=1
        )
        conditioning_window = flow_state.backbone_conditioning[
            :, -(model.history_patches + 1) :
        ]
        backbone_conditioning = model.dit_head(conditioning_window)
        backbone_conditioning = backbone_conditioning.repeat_interleave(
            model.patch_size, dim=1
        )
        if flow_state.speaker_conditioning is not None:
            speaker_conditioning = flow_state.speaker_conditioning.unsqueeze(1).expand(
                -1, backbone_conditioning.shape[1], -1
            )
            conditioning = torch.cat(
                [backbone_conditioning, speaker_conditioning], dim=-1
            )
        else:
            conditioning = backbone_conditioning
        history_latents = flow_state.latent_history[:, -model.history_length :]
        initial_noise = torch.empty(
            1,
            model.patch_size,
            model.redae_dim,
            device=conditioning.device,
            dtype=torch.float32,
        )
        initial_noise.normal_(generator=flow_state.generator)
    return FlowRow(
        request_data=request_data,
        history_latents=history_latents,
        initial_noise=initial_noise,
        conditioning=conditioning,
    )


def execute_flow_rows(
    model: FireRedTTS3SGLangModel,
    flow_graphs: FlowGraphRunner | None,
    flow_rows: list[FlowRow],
    timestep_schedule: TimestepScheduleProvider,
) -> None:
    """Group rows by Euler step count and write feedback in request order."""
    groups: dict[int, list[FlowRow]] = {}
    for flow_row in flow_rows:
        groups.setdefault(int(flow_row.request_data.state.n_timesteps), []).append(
            flow_row
        )
    feedback_dtype = next(model.parameters()).dtype
    for n_timesteps, flow_group in groups.items():
        with torch.inference_mode():
            history_latents = torch.cat(
                [flow_row.history_latents for flow_row in flow_group]
            )
            initial_noise = torch.cat(
                [flow_row.initial_noise for flow_row in flow_group]
            )
            conditioning = torch.cat([flow_row.conditioning for flow_row in flow_group])
            guidance_scales = torch.tensor(
                [
                    float(flow_row.request_data.state.inference_cfg)
                    for flow_row in flow_group
                ],
                dtype=torch.float32,
                device=conditioning.device,
            ).view(-1, 1, 1)
            profiling_probes = [flow_row.request_data.probe for flow_row in flow_group]
            with ProfilingProbe.shared_segment("dit_flow", profiling_probes):
                generated_patches = (
                    flow_graphs.run(
                        history_latents,
                        initial_noise,
                        conditioning,
                        guidance_scales,
                        n_timesteps,
                    )
                    if flow_graphs is not None
                    else None
                )
                if generated_patches is None:
                    generated_patches = euler_flow_loop(
                        model.dit,
                        torch.cat([history_latents, initial_noise], dim=1),
                        conditioning,
                        timestep_schedule(n_timesteps, conditioning.device),
                        guidance_scales,
                        n_timesteps,
                        model.patch_size,
                    )
                else:
                    pass
            with ProfilingProbe.shared_segment("patch_encoder", profiling_probes):
                feedback_embeddings = model.patch_encoder(generated_patches)
            for request_row_index, flow_row in enumerate(flow_group):
                request_data = flow_row.request_data
                flow_state = request_data.flow_state
                assert isinstance(flow_state, FlowState)
                generated_patch = generated_patches[
                    request_row_index : request_row_index + 1
                ]
                flow_state.latent_history = torch.cat(
                    [flow_state.latent_history, generated_patch], dim=1
                )
                flow_state.patch_count += 1
                request_data.latent_patches.append(generated_patch)
                request_data.pending_feedback_queue.append(
                    feedback_embeddings[0, request_row_index].to(dtype=feedback_dtype)
                )
