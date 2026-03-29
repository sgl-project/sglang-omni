# SPDX-License-Identifier: Apache-2.0
"""Merge and decode helpers for MiniCPM-V/MiniCPM-o pipelines.

MiniCPM-V/MiniCPM-o merging:
- Image embeddings via SigLIP + Perceiver Resampler
- Audio embeddings via Whisper encoder (MiniCPM-o only)
- Uses tgt_sizes for image position encoding
- LLM handles embedding injection via masked_scatter
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from sglang_omni.models.minicpm_v.io import LLMOutput, OmniEvent, PipelineState
from sglang_omni.models.minicpm_v.pipeline.next_stage import AUDIO_STAGE, IMAGE_STAGE
from sglang_omni.proto import StagePayload


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    try:
        return torch.as_tensor(value, dtype=dtype)
    except Exception:
        return None


def _non_empty(tensor: torch.Tensor | None) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.numel() > 0


def merge_for_llm(payloads: dict[str, StagePayload]) -> StagePayload:
    """Aggregate preprocessing + encoder outputs into LLM inputs.

    This function combines:
    - Preprocessing output (input_ids, attention_mask)
    - Image encoder output (image_embeds, tgt_sizes)

    Into a unified state ready for the LLM stage.
    """
    base = payloads.get("preprocessing") or next(iter(payloads.values()))
    state = PipelineState.from_dict(base.data)
    encoder_outs: dict[str, Any] = {}
    if state.encoder_outs:
        encoder_outs.update(state.encoder_outs)

    for stage_name, payload in payloads.items():
        stage_state = PipelineState.from_dict(payload.data)
        if stage_name in stage_state.encoder_outs:
            encoder_outs[stage_name] = stage_state.encoder_outs[stage_name]
            continue
        if stage_name in stage_state.engine_outputs:
            encoder_outs[stage_name] = stage_state.engine_outputs[stage_name]

    llm_inputs = build_llm_inputs(state, encoder_outs)

    state.encoder_outs = encoder_outs
    state.llm_inputs = llm_inputs
    state.encoder_inputs = {}
    _prune_preprocessing_for_llm(state, encoder_outs)
    base.data = state.to_dict()
    return base


def build_llm_inputs(
    state: PipelineState,
    encoder_outs: dict[str, Any],
) -> dict[str, Any]:
    """Build LLM inputs from pipeline state and encoder outputs.

    MiniCPM-V LLM inputs:
    - image_embeds: [total_tokens, hidden_dim] - flattened image embeddings
    - tgt_sizes: [total_slices, 2] - patch grid sizes for 2D RoPE
    - slice_lengths: number of slices per image (for reconstruction)

    MiniCPM-o additional inputs:
    - audio_embeds: [total_audio_tokens, hidden_dim] - audio embeddings
    - audio_output_lengths: [num_audio_clips] - lengths of each audio segment
    """
    mm_inputs = state.mm_inputs
    mm_image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}

    # --- Image encoder outputs ---
    image_out = (
        encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )

    image_embeds = (
        _as_tensor(image_out.get("image_embeds"))
        if isinstance(image_out, dict)
        else None
    )
    tgt_sizes = _as_tensor(
        (
            image_out.get("tgt_sizes")
            if isinstance(image_out, dict) and image_out.get("tgt_sizes") is not None
            else mm_image.get("tgt_sizes")
        ),
        dtype=torch.long,
    )
    slice_lengths = (
        image_out.get("slice_lengths")
        if isinstance(image_out, dict)
        else mm_image.get("slice_lengths")
    )

    # --- Audio encoder outputs (MiniCPM-o only) ---
    audio_out = (
        encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    audio_embeds = (
        _as_tensor(audio_out.get("audio_embeds"))
        if isinstance(audio_out, dict)
        else None
    )
    audio_output_lengths = (
        audio_out.get("audio_output_lengths")
        if isinstance(audio_out, dict)
        else None
    )

    # --- Build model inputs ---
    llm_model_inputs: dict[str, Any] = {}
    if _non_empty(image_embeds):
        llm_model_inputs["image_embeds"] = image_embeds
    if _non_empty(tgt_sizes):
        llm_model_inputs["tgt_sizes"] = tgt_sizes
    if slice_lengths:
        llm_model_inputs["slice_lengths"] = slice_lengths

    # Audio inputs (MiniCPM-o)
    if _non_empty(audio_embeds):
        llm_model_inputs["audio_embeds"] = audio_embeds
    if audio_output_lengths is not None:
        llm_model_inputs["audio_output_lengths"] = audio_output_lengths

    return {"model_inputs": llm_model_inputs}


def _prune_preprocessing_for_llm(
    state: PipelineState,
    encoder_outs: dict[str, Any],
) -> None:
    """Prune preprocessing data to reduce payload size after merge."""
    mm_inputs = state.mm_inputs
    mm_image = mm_inputs.get("image", {}) if isinstance(mm_inputs, dict) else {}

    image_out = (
        encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )

    tgt_sizes = _as_tensor(
        (
            image_out.get("tgt_sizes")
            if isinstance(image_out, dict) and image_out.get("tgt_sizes") is not None
            else mm_image.get("tgt_sizes")
        ),
        dtype=torch.long,
    )

    # Retain minimal metadata for LLM stage
    pruned_mm_inputs: dict[str, Any] = {
        "image": {"tgt_sizes": tgt_sizes},
    }

    # Preserve audio metadata if present (MiniCPM-o)
    audio_out = (
        encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    if isinstance(audio_out, dict) and audio_out.get("audio_output_lengths") is not None:
        pruned_mm_inputs["audio"] = {
            "audio_output_lengths": audio_out.get("audio_output_lengths"),
        }

    state.mm_inputs = pruned_mm_inputs


def decode_events(
    *,
    llm_out: LLMOutput,
    state: PipelineState,
    tokenizer: Any,
    eos_token_id: int | None,
    step: int,
) -> Iterable[OmniEvent]:
    """Decode LLM output tokens into streaming events.

    This is a text-only decoder for MiniCPM-V Phase 1.
    """
    output_ids = llm_out.get("output_ids", [])
    if not isinstance(output_ids, list) or not output_ids:
        return []

    stream_state = state.stream_state
    if not stream_state:
        stream_state.update({"token_ids": [], "text": "", "emitted_text": ""})
    token_ids = stream_state.setdefault("token_ids", [])
    stream_state.setdefault("text", "")
    stream_state.setdefault("emitted_text", "")

    is_final = bool(llm_out.get("is_final"))

    if is_final:
        tokens = [
            int(t)
            for t in output_ids
            if eos_token_id is None or int(t) != int(eos_token_id)
        ]
        text = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
        stream_state["token_ids"] = tokens
        stream_state["text"] = text
        return [
            OmniEvent(
                type="text_final",
                modality="text",
                payload={"text": text},
                is_final=True,
            )
        ]

    token_id = int(output_ids[-1])
    if eos_token_id is not None and token_id == int(eos_token_id):
        text = str(stream_state.get("text", ""))
        return [
            OmniEvent(
                type="text_final",
                modality="text",
                payload={"text": text},
                is_final=True,
            )
        ]

    token_ids.append(token_id)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    stream_state["text"] = decoded

    # Skip incomplete multi-byte characters (replacement char).
    if "\ufffd" in decoded:
        return []

    emitted_text = str(stream_state.get("emitted_text", ""))
    delta = decoded[len(emitted_text) :]
    if not delta:
        return []
    stream_state["emitted_text"] = decoded
    return [
        OmniEvent(
            type="text_delta", modality="text", payload={"text": delta}, is_final=False
        )
    ]
