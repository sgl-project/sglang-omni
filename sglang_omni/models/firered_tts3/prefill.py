# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 prompt embeddings and retained-history replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

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
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    TEXT_MODE_AUDIO,
    FireRedFlowResume,
    FireRedTTS3SGLangRequestData,
    FlowState,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.sampling.seed import derive_sampling_seed


@dataclass(kw_only=True)
class PreparedPrefill:
    """Temporary conditioning and flow state for one materialized request."""

    flow_state: FlowState
    prompt_latents: torch.Tensor | None
    speaker_embedding: torch.Tensor | None


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
