# SPDX-License-Identifier: Apache-2.0
"""TalkerInputBuilder — construct HF-parity talker prefill inputs."""
from __future__ import annotations

import torch


def segment_chat_template(
    input_ids: torch.Tensor,
    *,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
) -> list[dict]:
    """Parse input_ids into chat template segments by <|im_start|> boundaries.

    Returns list of {"role": str, "start": int, "end": int}.
    """
    role_map = {
        system_token_id: "system",
        user_token_id: "user",
        assistant_token_id: "assistant",
    }

    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    segments = []
    im_start_positions = [i for i, tok in enumerate(ids) if tok == im_start_token_id]

    for idx, pos in enumerate(im_start_positions):
        # Role token is the one after im_start
        role_token = ids[pos + 1] if pos + 1 < len(ids) else None
        role = role_map.get(role_token, "unknown")
        start = pos
        end = (
            im_start_positions[idx + 1]
            if idx + 1 < len(im_start_positions)
            else len(ids)
        )
        segments.append({"role": role, "start": start, "end": end})

    return segments


def build_user_part(
    *,
    thinker_embed: torch.Tensor,
    thinker_hidden: torch.Tensor,
    multimodal_mask: torch.Tensor,
    text_projection,
    hidden_projection,
) -> torch.Tensor:
    """Build user segment: text_projection for text, hidden_projection for multimodal."""
    out_size = text_projection(thinker_embed[:1]).shape[-1]
    result = torch.empty(
        (thinker_embed.shape[0], out_size),
        device=thinker_embed.device,
        dtype=thinker_embed.dtype,
    )
    if multimodal_mask.any():
        result[multimodal_mask] = hidden_projection(thinker_hidden[multimodal_mask])
    text_mask = ~multimodal_mask
    if text_mask.any():
        result[text_mask] = text_projection(thinker_embed[text_mask])
    return result


def build_assistant_part(
    *,
    assistant_embed: torch.Tensor,
    text_projection,
    codec_embed_fn,
    tts_bos_embed: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    speaker_id: int,
    codec_nothink_id: int,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
    codec_pad_id: int,
    codec_bos_id: int,
    tts_pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Build assistant segment matching HF's _get_talker_assistant_parts."""
    device = assistant_embed.device
    dtype = assistant_embed.dtype

    projected = text_projection(assistant_embed)  # [N, hidden]

    # Text side: [first 3] + [4x pad] + [bos] + [4th token]
    # The initial talker request can be built before the thinker has emitted
    # four assistant tokens, so keep the slot stable and fill it later through
    # future text-row queue updates.
    fourth_token = (
        projected[3:4]
        if projected.shape[0] > 3
        else torch.zeros((1, projected.shape[-1]), device=device, dtype=dtype)
    )
    text_hidden = torch.cat(
        [
            projected[:3],
            tts_pad_embed.expand(4, -1),
            tts_bos_embed,
            fourth_token,
        ],
        dim=0,
    )  # [9, hidden]

    # Codec side: [3x zeros] + [embed(6 special tokens)]
    codec_special_ids = torch.tensor(
        [
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            speaker_id,
            codec_pad_id,
            codec_bos_id,
        ],
        device=device,
        dtype=torch.long,
    )
    codec_embeds = codec_embed_fn(codec_special_ids)  # [6, hidden]
    codec_hidden = torch.cat(
        [
            torch.zeros((3, text_hidden.shape[-1]), device=device, dtype=dtype),
            codec_embeds,
        ],
        dim=0,
    )  # [9, hidden]

    input_embeds = text_hidden + codec_hidden

    input_ids = torch.full(
        (text_hidden.shape[0],),
        tts_pad_token_id,
        dtype=torch.long,
        device=device,
    )

    # HF's trailing_text_hidden is a FIFO stream of future assistant text rows:
    # assistant tokens after the first spoken token, then TTS EOS.
    if projected.shape[0] > 4:
        future_text_rows = torch.cat([projected[4:], tts_eos_embed], dim=0)
    else:
        future_text_rows = tts_eos_embed.clone()

    return {
        "input_embeds": input_embeds,
        "input_ids": input_ids,
        "future_text_rows": future_text_rows,
    }


def build_prefill_input(
    *,
    thinker_embed: torch.Tensor,
    thinker_hidden: torch.Tensor,
    thinker_input_ids: torch.Tensor,
    multimodal_mask: torch.Tensor,
    text_projection,
    hidden_projection,
    codec_embed_fn,
    tts_bos_embed: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
    speaker_id: int,
    codec_nothink_id: int,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
    codec_pad_id: int,
    codec_bos_id: int,
    tts_pad_token_id: int,
    include_assistant_eos: bool = True,
    im_end_token_id: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build full talker prefill input from thinker outputs.

    When *im_end_token_id* is provided, the ``<|im_end|>`` token is stripped
    from the assistant segment before projection.  HF's ``thinker_embed``
    never contains a hidden state for the EOS token (``generate()`` stops
    before emitting one), so including the raw embedding introduces an
    off-by-one that shifts the future text-row queue.
    """
    segments = segment_chat_template(
        thinker_input_ids,
        im_start_token_id=im_start_token_id,
        system_token_id=system_token_id,
        user_token_id=user_token_id,
        assistant_token_id=assistant_token_id,
    )

    all_embeds = []
    all_ids = []
    future_text_rows = None
    assistant_segment_indices = [
        idx for idx, seg in enumerate(segments) if seg["role"] == "assistant"
    ]
    last_assistant_idx = (
        assistant_segment_indices[-1] if assistant_segment_indices else None
    )

    for seg_idx, seg in enumerate(segments):
        if seg["role"] == "system":
            continue

        start, end = seg["start"], seg["end"]
        # HF includes im_start token in each segment's slice
        seg_embed = thinker_embed[start:end]
        seg_hidden = thinker_hidden[start:end]
        seg_mm_mask = multimodal_mask[start:end]

        if seg["role"] == "user":
            user_part = build_user_part(
                thinker_embed=seg_embed,
                thinker_hidden=seg_hidden,
                multimodal_mask=seg_mm_mask,
                text_projection=text_projection,
                hidden_projection=hidden_projection,
            )
            all_embeds.append(user_part)
            all_ids.append(thinker_input_ids[start:end].to(dtype=torch.long))

        elif seg["role"] == "assistant":
            if last_assistant_idx is not None and seg_idx != last_assistant_idx:
                continue
            # Strip <|im_end|> from the assistant segment to match HF,
            # whose thinker_embed never contains a hidden state for the
            # EOS token produced by generate().
            if (
                im_end_token_id is not None
                and seg_embed.shape[0] > 0
                and int(thinker_input_ids[end - 1].item()) == im_end_token_id
            ):
                seg_embed = seg_embed[:-1]
            assistant_result = build_assistant_part(
                assistant_embed=seg_embed,
                text_projection=text_projection,
                codec_embed_fn=codec_embed_fn,
                tts_bos_embed=tts_bos_embed,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                speaker_id=speaker_id,
                codec_nothink_id=codec_nothink_id,
                codec_think_bos_id=codec_think_bos_id,
                codec_think_eos_id=codec_think_eos_id,
                codec_pad_id=codec_pad_id,
                codec_bos_id=codec_bos_id,
                tts_pad_token_id=tts_pad_token_id,
            )
            all_embeds.append(assistant_result["input_embeds"])
            all_ids.append(
                assistant_result["input_ids"].to(
                    device=thinker_input_ids.device,
                    dtype=torch.long,
                )
            )
            future_text_rows = assistant_result["future_text_rows"]
            if (
                not include_assistant_eos
                and future_text_rows is not None
                and future_text_rows.shape[0] > 0
            ):
                future_text_rows = future_text_rows[:-1]

    return {
        "input_embeds": torch.cat(all_embeds, dim=0),
        "input_ids": torch.cat(all_ids, dim=0),
        "future_text_rows": future_text_rows,
    }
