# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 prompt embeddings and retained-history replay."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.firered_tts3.embeddings import (
    LATENT_IN_PAD_ID as LATENT_IN_PAD_ID,
)
from sglang_omni.models.firered_tts3.embeddings import (
    LATENT_OUT_PAD_ID as LATENT_OUT_PAD_ID,
)
from sglang_omni.models.firered_tts3.embeddings import (
    build_base_prefill_embeds,
    build_chatml_prefill_embeds,
)
from sglang_omni.models.firered_tts3.embeddings import (
    build_dual_stream_replay_rows as build_dual_stream_replay_rows,
)
from sglang_omni.models.firered_tts3.embeddings import (
    build_replay_rows,
    text_output_token_count,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    TEXT_MODE_AUDIO,
    TEXT_MODE_TEXT,
    FireRedFlowResume,
    FireRedTTS3SGLangRequestData,
    FlowState,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass(kw_only=True)
class PreparedPrefill:
    """Temporary conditioning and flow state for one materialized request."""

    flow_state: FlowState
    prompt_latents: torch.Tensor | None
    speaker_embedding: torch.Tensor | None


class RequestStateHook(Protocol):
    def __call__(self, request_data: FireRedTTS3SGLangRequestData) -> None: ...


def materialize_prefill_batch(
    model: FireRedTTS3SGLangModel,
    forward_batch: ForwardBatch,
    requests: list[SchedulerRequest],
    active_requests: dict[str, FireRedTTS3SGLangRequestData],
    suspend_request_data: RequestStateHook,
    clear_request_data: RequestStateHook,
) -> float | None:
    """Prepare embeddings and roll back request state if preparation fails."""
    device = forward_batch.input_ids.device
    embedding_dtype = next(model.parameters()).dtype
    model.text_logits_requested = any(
        request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests
    )
    embedding_rows: list[torch.Tensor] = []
    materialized_requests: list[tuple[str, FireRedTTS3SGLangRequestData]] = []
    try:
        for request in requests:
            request_data = request.data
            if request_data.flow_state is not None and (
                not isinstance(request_data.flow_state, FireRedFlowResume)
            ):
                suspend_request_data(request_data)
            else:
                pass
            active_requests.pop(request.request_id, None)
            prepared = prepare_prefill_flow(
                model, request_data, request.request_id, device
            )
            request_data.flow_state = prepared.flow_state
            request_data.probe = ProfilingProbe.create()
            active_requests[request.request_id] = request_data
            materialized_requests.append((request.request_id, request_data))
            embedding_rows.append(
                materialize_prefill_embeddings(
                    model, request_data, prepared, device, embedding_dtype
                )
            )
        forward_batch.input_embeds = torch.cat(embedding_rows, dim=0)
        profiling_enabled = any(
            request_data.probe.enabled
            for request_id, request_data in materialized_requests
        )
        if profiling_enabled:
            return time.perf_counter()
        else:
            return None
    except BaseException:
        for request_id, request_data in materialized_requests:
            active_requests.pop(request_id, None)
            clear_request_data(request_data)
        raise


def audio_hidden_row_offset(request_data: FireRedTTS3SGLangRequestData) -> int:
    """Count hidden rows preceding the first patch-driving audio-start feed."""
    if request_data.text_ar_mode == TEXT_MODE_AUDIO and (
        request_data.text_token_ids or request_data.eot_sampled
    ):
        return len(request_data.text_token_ids) + int(request_data.eot_sampled) + 1
    else:
        pass
    return 0


def dual_stream_conditioning_rows(
    request_data: FireRedTTS3SGLangRequestData, hidden_states_float32: torch.Tensor
) -> torch.Tensor:
    """Reconstruct flow-driving hidden rows in their generation order."""
    assert request_data.interleave_ratio is not None
    assert isinstance(request_data.flow_state, FlowState)
    text_steps, patch_steps = request_data.interleave_ratio
    group_steps = text_steps + patch_steps
    hidden_row_indices: list[int] = []
    remaining_patch_count = request_data.flow_state.patch_count
    group_index = 0
    while remaining_patch_count > 0:
        group_start = request_data.prefill_token_count + group_index * group_steps
        hidden_row_indices.append(group_start + text_steps - 1)
        remaining_patch_count -= 1
        for patch_index in range(patch_steps - 1):
            if remaining_patch_count == 0:
                break
            else:
                pass
            hidden_row_indices.append(group_start + text_steps + patch_index)
            remaining_patch_count -= 1
        group_index += 1
    return hidden_states_float32[hidden_row_indices].unsqueeze(0)


def restore_prefill_conditioning(
    request_data: FireRedTTS3SGLangRequestData, hidden_states_float32: torch.Tensor
) -> None:
    """Append retained hidden rows to the flow history after re-prefill."""
    flow_state = request_data.flow_state
    assert isinstance(flow_state, FlowState)
    if flow_state.patch_count <= 0:
        return
    elif request_data.dual_stream:
        conditioning_parts = [
            flow_state.backbone_conditioning,
            dual_stream_conditioning_rows(request_data, hidden_states_float32),
        ]
    else:
        prompt_hidden_start = (
            request_data.prefill_token_count - flow_state.prompt_patch_count
        )
        audio_hidden_start = request_data.prefill_token_count + audio_hidden_row_offset(
            request_data
        )
        conditioning_parts = [
            flow_state.backbone_conditioning,
            hidden_states_float32[
                prompt_hidden_start : request_data.prefill_token_count
            ].unsqueeze(0),
            hidden_states_float32[audio_hidden_start:-1].unsqueeze(0),
        ]
    flow_state.backbone_conditioning = torch.cat(conditioning_parts, dim=1)


def prepare_prefill_flow(
    model: FireRedTTS3SGLangModel,
    request_data: FireRedTTS3SGLangRequestData,
    request_id: str,
    device: torch.device,
) -> PreparedPrefill:
    """Reconstruct request conditioning and generator state before prefill."""
    state = request_data.state
    resume = request_data.flow_state
    if state.variant == "speaker":
        if state.prompt_latents is not None or state.speaker_embedding is not None:
            raise RuntimeError(
                "FireRedTTS3 speaker-variant requests carry no reference-conditioning tensors (prompt_latents / speaker_embedding)"
            )
        else:
            pass
    elif state.variant != "instruct" and (
        state.prompt_latents is None or state.speaker_embedding is None
    ):
        raise RuntimeError(
            "FireRedTTS3 request is missing its reference-conditioning tensors (prompt_latents / speaker_embedding)"
        )
    else:
        pass
    prompt_latents = (
        state.prompt_latents.to(device=device, dtype=torch.float32)
        if state.prompt_latents is not None
        else None
    )
    speaker_embedding = (
        state.speaker_embedding.to(device=device, dtype=torch.float32)
        if state.speaker_embedding is not None
        else None
    )
    generator = torch.Generator(device=device)
    if isinstance(resume, FireRedFlowResume) and resume.rng_state is not None:
        generator.set_state(resume.rng_state)
    elif state.seed is not None:
        generator.manual_seed(int(state.seed))
    else:
        generator.manual_seed(derive_sampling_seed("fireredtts3-flow", request_id))
    if prompt_latents is not None:
        latent_history = F.pad(prompt_latents, (0, 0, model.history_length, 0))
    else:
        latent_history = torch.zeros(
            1, model.history_length, model.redae_dim, device=device
        )
    if request_data.latent_patches:
        latent_history = torch.cat(
            [latent_history, *request_data.latent_patches], dim=1
        )
    else:
        pass
    return PreparedPrefill(
        flow_state=FlowState(
            generator=generator,
            prompt_patch_count=request_data.prompt_patch_count,
            speaker_conditioning=(
                model.spk_proj_dit(speaker_embedding)
                if model.spk_proj_dit is not None
                else None
            ),
            backbone_conditioning=torch.zeros(
                1, model.history_patches, model.llm_hidden_size, device=device
            ),
            latent_history=latent_history,
            patch_count=len(request_data.latent_patches),
        ),
        prompt_latents=prompt_latents,
        speaker_embedding=speaker_embedding,
    )


def materialize_prefill_embeddings(
    model: FireRedTTS3SGLangModel,
    request_data: FireRedTTS3SGLangRequestData,
    prepared: PreparedPrefill,
    device: torch.device,
    embedding_dtype: torch.dtype,
) -> torch.Tensor:
    """Build prompt and retained-history embeddings in engine token order."""
    state = request_data.state
    assert request_data.probe is not None
    with torch.inference_mode(), request_data.probe.segment("prefill_embeds"):
        if state.variant == "base":
            assert prepared.prompt_latents is not None
            assert prepared.speaker_embedding is not None
            prompt_embeddings = build_base_prefill_embeds(
                model,
                state,
                prepared.prompt_latents,
                prepared.speaker_embedding,
                device,
            )
        else:
            prompt_embeddings = build_chatml_prefill_embeds(model, state, device)
        if prompt_embeddings.shape[1] != request_data.prefill_token_count:
            raise RuntimeError(
                f"FireRedTTS3 prefill row is {prompt_embeddings.shape[1]} positions, expected {request_data.prefill_token_count}"
            )
        else:
            pass
        request_embeddings = [prompt_embeddings[0].to(embedding_dtype)]
        replay_embeddings = build_replay_rows(
            model, request_data, device, embedding_dtype
        )
        if replay_embeddings is not None:
            request_embeddings.append(replay_embeddings)
        else:
            pass
    prefix_token_count = request_data.req.prefix_len
    request_token_count = request_data.req.extend_len
    generated_token_count = request_token_count - request_data.prefill_token_count
    assert prefix_token_count == 0, "FireRedTTS3 radix prefix reuse is disabled"
    if request_data.dual_stream:
        patch_feeds = sum(
            1
            for token_id in request_data.req.output_ids
            if int(token_id) == CONTROL_TOKEN_ID
        )
        excess = patch_feeds - len(request_data.latent_patches)
        assert (
            generated_token_count == len(request_data.req.output_ids)
            and excess >= 0
            and (excess == 0 or request_data.stop_pending)
        ), "FireRedTTS3 dual-stream re-prefill history is inconsistent"
    else:
        excess = (
            len(request_data.req.output_ids)
            - text_output_token_count(request_data)
            - len(request_data.latent_patches)
        )
        assert generated_token_count == len(request_data.req.output_ids) and (
            excess >= 0 and (excess == 0 or request_data.stop_pending)
        ), "FireRedTTS3 re-prefill history must match generated tokens"
    return torch.cat(request_embeddings, dim=0)
