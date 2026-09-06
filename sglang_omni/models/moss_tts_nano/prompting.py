# SPDX-License-Identifier: Apache-2.0
"""Prompt construction for MOSS-TTS-Nano checkpoints."""

from __future__ import annotations

from typing import Any, Sequence

import torch

USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
USER_TEMPLATE_AFTER_REFERENCE = (
    "\n- Instruction:\nNone\n"
    "- Tokens:\nNone\n"
    "- Quality:\nNone\n"
    "- Sound Event:\nNone\n"
    "- Ambient Sound:\nNone\n"
    "- Language:\nNone\n"
    "- Text:\n"
)
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"


def encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def build_text_rows(
    token_ids: Sequence[int],
    *,
    n_vq: int,
    audio_pad_token_id: int,
) -> torch.Tensor:
    rows = torch.full(
        (len(token_ids), n_vq + 1),
        int(audio_pad_token_id),
        dtype=torch.long,
    )
    if token_ids:
        rows[:, 0] = torch.tensor(list(token_ids), dtype=torch.long)
    return rows


def build_audio_rows(
    audio_codes: torch.Tensor,
    *,
    slot_token_id: int,
    n_vq: int,
) -> torch.Tensor:
    codes = torch.as_tensor(audio_codes, dtype=torch.long, device="cpu")
    if codes.ndim != 2 or int(codes.shape[1]) != int(n_vq):
        raise ValueError(
            f"MOSS-TTS-Nano reference codes must have shape [T, {n_vq}], "
            f"got {tuple(codes.shape)}"
        )
    rows = torch.empty((int(codes.shape[0]), n_vq + 1), dtype=torch.long)
    rows[:, 0] = int(slot_token_id)
    rows[:, 1:] = codes
    return rows


def build_prompt_rows(
    *,
    tokenizer: Any,
    config: Any,
    text: str,
    reference_codes: torch.Tensor | None,
) -> torch.Tensor:
    """Build the exact multi-channel prompt consumed by the Nano checkpoint."""

    if not str(text).strip():
        raise ValueError("MOSS-TTS-Nano text must not be empty")

    n_vq = int(config.n_vq)
    audio_pad = int(config.audio_pad_token_id)
    prefix = (
        [int(config.im_start_token_id)]
        + encode_text(tokenizer, USER_ROLE_PREFIX)
        + encode_text(tokenizer, USER_TEMPLATE_REFERENCE_PREFIX)
    )
    sections = [build_text_rows(prefix, n_vq=n_vq, audio_pad_token_id=audio_pad)]
    if reference_codes is None:
        sections.append(
            build_text_rows(
                encode_text(tokenizer, "None"),
                n_vq=n_vq,
                audio_pad_token_id=audio_pad,
            )
        )
    else:
        sections.append(
            build_text_rows(
                [int(config.audio_start_token_id)],
                n_vq=n_vq,
                audio_pad_token_id=audio_pad,
            )
        )
        sections.append(
            build_audio_rows(
                reference_codes,
                slot_token_id=int(config.audio_user_slot_token_id),
                n_vq=n_vq,
            )
        )
        sections.append(
            build_text_rows(
                [int(config.audio_end_token_id)],
                n_vq=n_vq,
                audio_pad_token_id=audio_pad,
            )
        )

    suffix = (
        encode_text(tokenizer, USER_TEMPLATE_AFTER_REFERENCE)
        + encode_text(tokenizer, text)
        + encode_text(tokenizer, USER_TEMPLATE_SUFFIX)
        + [int(config.im_end_token_id)]
        + encode_text(tokenizer, ASSISTANT_TURN_PREFIX)
        + [int(config.im_start_token_id)]
        + encode_text(tokenizer, ASSISTANT_ROLE_PREFIX)
        + [int(config.audio_start_token_id)]
    )
    sections.append(build_text_rows(suffix, n_vq=n_vq, audio_pad_token_id=audio_pad))
    return torch.cat(sections, dim=0)
