# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 prompt embeddings and retained-history replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    DS_STAGE_PATCH,
    TEXT_MODE_AUDIO,
    TEXT_MODE_AUDIO_ZERO,
    TEXT_MODE_SOSP,
    FireRedFlowResume,
    FireRedTTS3SGLangRequestData,
    FlowState,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.sampling.seed import derive_sampling_seed

LATENT_IN_PAD_ID = 151655
LATENT_OUT_PAD_ID = 151656


@dataclass(kw_only=True)
class PreparedPrefill:
    """Temporary conditioning and flow state for one materialized request."""

    flow_state: FlowState
    prompt_latents: torch.Tensor | None
    speaker_embedding: torch.Tensor | None


def build_base_prefill_embeds(
    model: FireRedTTS3SGLangModel,
    state: FireRedTTS3State,
    prompt_latents: torch.Tensor,
    speaker_embedding: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Assemble speaker, text, and prompt-patch embeddings in model order."""
    assert model.spk_proj_llm is not None
    text_ids = torch.tensor(state.input_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )
    return torch.cat(
        [
            model.spk_proj_llm(speaker_embedding).unsqueeze(1),
            model.get_input_embeddings()(text_ids).to(torch.float32),
            model.patch_encoder(prompt_latents),
        ],
        dim=1,
    )


def build_chatml_prefill_embeds(
    model: FireRedTTS3SGLangModel, state: FireRedTTS3State, device: torch.device
) -> torch.Tensor:
    """Scatter edit-source and prompt latents into their typed token slots."""
    text_ids = torch.tensor(state.input_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )
    embeddings = model.get_input_embeddings()(text_ids)
    for latents, pad_token_id, latent_kind in (
        (state.audio_in_latents, LATENT_IN_PAD_ID, "audio input"),
        (state.prompt_latents, LATENT_OUT_PAD_ID, "ICL audio"),
    ):
        if latents is None:
            continue
        else:
            pass
        patch_embeddings = model.patch_encoder(
            latents.to(device=device, dtype=torch.float32)
        )
        mask = text_ids == pad_token_id
        slot_count = int(mask.sum())
        if slot_count != int(patch_embeddings.shape[1]):
            raise RuntimeError(
                f"FireRedTTS3 {latent_kind} slot mismatch: the prompt carries {slot_count} placeholder tokens but the audio encodes to {int(patch_embeddings.shape[1])} patches"
            )
        else:
            pass
        embeddings = embeddings.masked_scatter(
            mask.unsqueeze(-1), patch_embeddings.reshape(-1).to(embeddings.dtype)
        )
    return embeddings.to(torch.float32)


def build_replay_rows(
    model: FireRedTTS3SGLangModel,
    request_data: FireRedTTS3SGLangRequestData,
    device: torch.device,
    embedding_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Replay retained text and patches, including feeds pending deferred stop."""
    if request_data.dual_stream:
        return build_dual_stream_replay_rows(
            model, request_data, device, embedding_dtype
        )
    else:
        pass
    text_token_count = text_output_token_count(request_data)
    embedding_rows = []
    if text_token_count:
        token_ids = torch.tensor(
            request_data.req.output_ids[:text_token_count],
            dtype=torch.long,
            device=device,
        )
        embedding_rows.append(
            model.get_input_embeddings()(token_ids).to(embedding_dtype)
        )
    else:
        pass
    if request_data.latent_patches:
        embedding_rows.append(
            model.patch_encoder(torch.cat(request_data.latent_patches, dim=1))[0].to(
                embedding_dtype
            )
        )
    else:
        pass
    excess = (
        len(request_data.req.output_ids)
        - text_token_count
        - len(request_data.latent_patches)
    )
    if request_data.stop_pending and excess > 0:
        token_ids = torch.full(
            (excess,), CONTROL_TOKEN_ID, dtype=torch.long, device=device
        )
        embedding_rows.append(
            model.get_input_embeddings()(token_ids).to(embedding_dtype)
        )
    else:
        pass
    if not embedding_rows:
        return None
    else:
        pass
    return torch.cat(embedding_rows, dim=0)


def build_dual_stream_replay_rows(
    model: FireRedTTS3SGLangModel,
    request_data: FireRedTTS3SGLangRequestData,
    device: torch.device,
    embedding_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Replay interleaved feeds and requeue an interrupted patch embedding."""
    output_ids = [int(token_id) for token_id in request_data.req.output_ids]
    if not output_ids:
        return None
    else:
        pass
    embed_tokens = model.get_input_embeddings()
    embedding_rows: list[torch.Tensor] = []
    text_token_buffer: list[int] = []
    patch_index = 0

    def flush_text() -> None:
        if text_token_buffer:
            token_ids = torch.tensor(text_token_buffer, dtype=torch.long, device=device)
            embedding_rows.append(embed_tokens(token_ids).to(embedding_dtype))
            text_token_buffer.clear()
        else:
            pass

    for token_id in output_ids:
        if token_id == CONTROL_TOKEN_ID:
            flush_text()
            if request_data.stop_pending and patch_index == len(
                request_data.latent_patches
            ):
                token_ids = torch.tensor(
                    [CONTROL_TOKEN_ID], dtype=torch.long, device=device
                )
                embedding_rows.append(embed_tokens(token_ids).to(embedding_dtype))
            else:
                assert patch_index < len(
                    request_data.latent_patches
                ), "FireRedTTS3 dual-stream replay found more patch feeds than generated patches"
                embedding_rows.append(
                    model.patch_encoder(request_data.latent_patches[patch_index])[0].to(
                        embedding_dtype
                    )
                )
                patch_index += 1
        else:
            text_token_buffer.append(token_id)
    flush_text()
    pending_patch_count = len(request_data.latent_patches) - patch_index
    assert pending_patch_count in (
        0,
        1,
    ), f"FireRedTTS3 dual-stream replay must leave at most one fed-back patch pending, got {pending_patch_count}"
    if pending_patch_count:
        assert (
            request_data.dual_stream_stage == DS_STAGE_PATCH
        ), f"FireRedTTS3 dual-stream replay expects the machine at a patch feed when a patch is pending, got {request_data.dual_stream_stage!r}"
        # note (kermit): A generated patch can outlive its interrupted input feed.
        request_data.pending_feedback_queue.append(
            model.patch_encoder(request_data.latent_patches[-1])[0, 0].to(
                dtype=embedding_dtype
            )
        )
    else:
        pass
    return torch.cat(embedding_rows, dim=0)


def text_output_token_count(request_data: FireRedTTS3SGLangRequestData) -> int:
    """Count retained text and transition tokens before audio patch feeds."""
    if not request_data.text_token_ids and (not request_data.eot_sampled):
        return 0
    else:
        pass
    text_token_count = len(request_data.text_token_ids) + int(request_data.eot_sampled)
    if request_data.text_ar_mode in (TEXT_MODE_AUDIO_ZERO, TEXT_MODE_AUDIO):
        text_token_count += 2
    elif request_data.text_ar_mode == TEXT_MODE_SOSP:
        text_token_count += 1
    else:
        pass
    return text_token_count


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
