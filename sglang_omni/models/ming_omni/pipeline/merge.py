# SPDX-License-Identifier: Apache-2.0
"""Merge and decode helpers for Ming-Omni pipelines."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from sglang_omni.models.ming_omni.io import (
    MingOmniEvent,
    MingOmniPipelineState,
    ThinkerOutput,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import AUDIO_STAGE, IMAGE_STAGE
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


def merge_for_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    """Aggregate preprocessing + audio encoder outputs into thinker inputs.

    Ming-Omni only has audio encoder (no image encoder in audio-only scope).
    The encoder output contains audio_embeds and audio_embed_lengths which are
    merged with the placeholder location info from preprocessing.
    """
    base = payloads.get("preprocessing") or next(iter(payloads.values()))
    state = MingOmniPipelineState.from_dict(base.data)
    encoder_outs: dict[str, Any] = {}
    if state.encoder_outs:
        encoder_outs.update(state.encoder_outs)

    for stage_name, payload in payloads.items():
        stage_state = MingOmniPipelineState.from_dict(payload.data)
        if stage_name in stage_state.encoder_outs:
            encoder_outs[stage_name] = stage_state.encoder_outs[stage_name]
            continue
        if stage_name in stage_state.engine_outputs:
            encoder_outs[stage_name] = stage_state.engine_outputs[stage_name]

    thinker_inputs = build_thinker_inputs(state, encoder_outs)

    state.encoder_outs = encoder_outs
    state.thinker_inputs = thinker_inputs
    state.encoder_inputs = {}
    base.data = state.to_dict()
    return base


def build_thinker_inputs(
    state: MingOmniPipelineState,
    encoder_outs: dict[str, Any],
) -> dict[str, Any]:
    """Build model_inputs dict for the Ming thinker from encoder outputs.

    The SGLang runtime's _inject_multimodal_embeds() handles embedding
    injection automatically: it finds positions where input_ids == token_id
    and patches embeddings from req.omni_model_inputs into those positions.

    We pass each modality as a flat [T', hidden_size] tensor —
    the runtime handles the rest via token ID matching.
    """
    audio_out = (
        encoder_outs.get(AUDIO_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )
    image_out = (
        encoder_outs.get(IMAGE_STAGE, {}) if isinstance(encoder_outs, dict) else {}
    )

    audio_embeds = (
        _as_tensor(audio_out.get("audio_embeds"))
        if isinstance(audio_out, dict)
        else None
    )
    image_embeds = (
        _as_tensor(image_out.get("image_embeds"))
        if isinstance(image_out, dict)
        else None
    )
    video_embeds = (
        _as_tensor(image_out.get("video_embeds"))
        if isinstance(image_out, dict)
        else None
    )

    thinker_model_inputs: dict[str, Any] = {}

    if _non_empty(audio_embeds):
        # Flatten: [B, T', H] -> [T', H] (remove batch dim for SGLang injection)
        if audio_embeds.dim() == 3:
            audio_embeds = audio_embeds.squeeze(0)
        thinker_model_inputs["audio_embeds"] = audio_embeds

    if _non_empty(image_embeds):
        # Flatten: [B, T', H] -> [T', H] if needed
        if image_embeds.dim() == 3:
            image_embeds = image_embeds.squeeze(0)
        thinker_model_inputs["image_embeds"] = image_embeds

    if _non_empty(video_embeds):
        if video_embeds.dim() == 3:
            video_embeds = video_embeds.squeeze(0)
        thinker_model_inputs["video_embeds"] = video_embeds

    media_cache_keys: dict[str, str] = {}
    encoder_inputs = state.encoder_inputs or {}
    image_ck = (encoder_inputs.get(IMAGE_STAGE) or {}).get("cache_key")
    audio_ck = (encoder_inputs.get(AUDIO_STAGE) or {}).get("cache_key")
    if image_ck:
        # Image and video share the same encoder cache key, so prefix them
        # differently so the SGLang adapter's modality-keyed lookup
        # (media_cache_keys.get("image"|"video")) gives each its own hashed
        # pad value. Without the "video" entry, video placeholder tokens
        # keep their raw token id and alias in the radix prefix cache
        # across different videos with the same placeholder count.
        media_cache_keys["image"] = f"image:{image_ck}"
        media_cache_keys["video"] = f"video:{image_ck}"
    if audio_ck:
        media_cache_keys["audio"] = f"audio:{audio_ck}"

    if not thinker_model_inputs:
        if media_cache_keys:
            return {"media_cache_keys": media_cache_keys}
        return {}
    result: dict[str, Any] = {"model_inputs": thinker_model_inputs}
    if media_cache_keys:
        result["media_cache_keys"] = media_cache_keys
    return result


def decode_events(
    *,
    thinker_out: ThinkerOutput,
    state: MingOmniPipelineState,
    tokenizer: Any,
    eos_token_id: int | None,
    step: int,
) -> Iterable[MingOmniEvent]:
    """Convert thinker output tokens to text events with streaming support."""
    output_ids = thinker_out.get("output_ids", [])
    if not isinstance(output_ids, list) or not output_ids:
        return []

    stream_state = state.stream_state
    if not stream_state:
        stream_state.update({"token_ids": [], "text": "", "emitted_text": ""})
    token_ids = stream_state.setdefault("token_ids", [])
    stream_state.setdefault("text", "")
    stream_state.setdefault("emitted_text", "")

    is_final = bool(thinker_out.get("is_final"))

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
            MingOmniEvent(
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
            MingOmniEvent(
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
        MingOmniEvent(
            type="text_delta", modality="text", payload={"text": delta}, is_final=False
        )
    ]
