# SPDX-License-Identifier: Apache-2.0
"""Materialize prompt, replay, and decode embeddings for FireRedTTS3."""

from __future__ import annotations

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    DS_STAGE_PATCH,
    TEXT_MODE_AUDIO,
    TEXT_MODE_AUDIO_ZERO,
    TEXT_MODE_SOSP,
    FireRedTTS3SGLangRequestData,
    scheduler_request_data,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.scheduling.types import SchedulerRequest

LATENT_IN_PAD_ID = 151655
LATENT_OUT_PAD_ID = 151656


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


def materialize_decode_embeddings(
    model: FireRedTTS3SGLangModel,
    forward_batch: ForwardBatch,
    requests: list[SchedulerRequest],
) -> None:
    """Stage feedback and text embeddings in upcoming batch row order."""
    embedding_rows = []
    for request_row_index, request in enumerate(requests):
        request_data = scheduler_request_data(request)
        feed_is_patch = (
            not request_data.dual_stream
            and request_data.text_ar_mode == TEXT_MODE_AUDIO
            or (
                request_data.dual_stream
                and request_data.dual_stream_stage == DS_STAGE_PATCH
            )
        )
        if feed_is_patch:
            feedback_queue = request_data.pending_feedback_queue
            if feedback_queue:
                embedding_rows.append(feedback_queue.popleft())
            elif request_data.stop_pending:
                # note (kermit): Deferred stop discards this feed; decode still owns KV cleanup.
                embedding_rows.append(
                    torch.zeros(
                        model.llm_hidden_size,
                        device=forward_batch.input_ids.device,
                        dtype=next(model.parameters()).dtype,
                    )
                )
            else:
                raise RuntimeError("FireRedTTS3 decode is missing its latent feedback")
        else:
            token_id = int(forward_batch.input_ids[request_row_index])
            embedding_rows.append(
                model.get_input_embeddings()(
                    torch.tensor(
                        [token_id],
                        dtype=torch.long,
                        device=forward_batch.input_ids.device,
                    )
                )[0]
            )
    feedback_buffer = model.graph_feedback_buffer
    if feedback_buffer is not None:
        # note (kermit): Decode graphs read this persistent buffer rather than input_embeds.
        torch.stack(embedding_rows, out=feedback_buffer[: len(embedding_rows)])
    else:
        forward_batch.input_embeds = torch.stack(embedding_rows).to(
            device=forward_batch.input_ids.device,
            dtype=next(model.parameters()).dtype,
        )
