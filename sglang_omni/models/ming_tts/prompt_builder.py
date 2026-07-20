# SPDX-License-Identifier: Apache-2.0
"""Prompt construction for Ming-Omni-TTS."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.models.ming_tts.tokenizer import (
    AUDIO_PATCH_TOKEN,
    AUDIO_START_TOKEN,
    SPK_START_TOKEN,
    MingTTSTokenizerBundle,
)

DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n"
TEXT_INPUT_PREFIX = " Text input:\n"
HUMAN_ROLE_PROMPT = "<role>HUMAN</role>"
ASSISTANT_ROLE_PROMPT = "<role>ASSISTANT</role>"
BGM_TTA_MARKERS = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")


@dataclass(frozen=True)
class MingTTSPromptPlan:
    input_ids: list[int]
    effective_prompt: str
    prompt_tokens: int
    audio_token_position: int
    spk_token_positions: list[int]
    spk_injection_positions: list[int]
    prompt_latent_start_position: int | None
    prompt_latent_token_count: int


def build_ming_tts_prompt(
    state: MingTTSState,
    tokenizer: MingTTSTokenizerBundle,
    *,
    instruction_text: str | None = None,
    prompt_text: str | None = None,
    speaker_count: int = 0,
    speaker_labels: list[str] | None = None,
    prompt_latent_token_count: int | None = None,
) -> MingTTSPromptPlan:
    """Build the official Ming-Omni-TTS MoE prompt token sequence."""

    text = state.text
    effective_prompt = state.prompt or DEFAULT_TTS_PROMPT
    if instruction_text is None:
        instruction_text = state.instructions
    if prompt_text is None:
        prompt_text = state.ref_text
    if prompt_latent_token_count is None:
        prompt_latent_token_count = state.prompt_latent_token_count

    speaker_count = int(speaker_count)
    prompt_latent_token_count = int(prompt_latent_token_count)
    if speaker_count < 0:
        raise ValueError("speaker_count must be non-negative")
    if prompt_latent_token_count < 0:
        raise ValueError("prompt_latent_token_count must be non-negative")
    if speaker_labels is not None and len(speaker_labels) != speaker_count:
        raise ValueError(
            "speaker_labels length must match speaker_count: "
            f"{len(speaker_labels)} != {speaker_count}"
        )

    input_ids: list[int] = []
    spk_token_positions: list[int] = []

    input_ids.extend(tokenizer.encode_no_special(HUMAN_ROLE_PROMPT))
    input_ids.extend(tokenizer.encode_no_special(effective_prompt))

    for speaker_index in range(speaker_count):
        label = (
            speaker_labels[speaker_index]
            if speaker_labels is not None
            else f"speaker_{speaker_index + 1}"
        )
        input_ids.extend(tokenizer.encode_no_special(f"  {label}:"))
        spk_token_position = len(input_ids)
        input_ids.extend(tokenizer.encode_no_special(SPK_START_TOKEN))
        input_ids.extend(tokenizer.encode_no_special(AUDIO_PATCH_TOKEN))
        input_ids.extend(tokenizer.encode_no_special("</spk>\n"))
        spk_token_positions.append(spk_token_position)

    text_input_prefix_included = not all(marker in text for marker in BGM_TTA_MARKERS)
    if text_input_prefix_included:
        input_ids.extend(tokenizer.encode_no_special(TEXT_INPUT_PREFIX))

    if prompt_text:
        input_ids.extend(tokenizer.encode_no_special(prompt_text))
    input_ids.extend(tokenizer.encode_no_special(text))
    input_ids.extend(tokenizer.encode_no_special(ASSISTANT_ROLE_PROMPT))

    if instruction_text:
        input_ids.extend(tokenizer.encode_no_special(instruction_text))
        input_ids.append(tokenizer.special.eos)

    audio_token_position = len(input_ids)
    input_ids.extend(tokenizer.encode_no_special(AUDIO_START_TOKEN))
    if input_ids[audio_token_position] != tokenizer.special.audio_start:
        raise ValueError(
            "Ming-Omni-TTS prompt audio anchor is not "
            f"{AUDIO_START_TOKEN}: {input_ids[audio_token_position]}"
        )

    prompt_latent_start_position = None
    if prompt_latent_token_count:
        prompt_latent_start_position = audio_token_position + 1
        input_ids.extend([tokenizer.special.audio_patch] * prompt_latent_token_count)

    if input_ids[0] != tokenizer.special.role_start:
        raise ValueError("Ming-Omni-TTS MoE prompt must start with role tokens")

    return MingTTSPromptPlan(
        input_ids=input_ids,
        effective_prompt=effective_prompt,
        prompt_tokens=len(input_ids),
        audio_token_position=audio_token_position,
        spk_token_positions=spk_token_positions,
        spk_injection_positions=[position + 1 for position in spk_token_positions],
        prompt_latent_start_position=prompt_latent_start_position,
        prompt_latent_token_count=prompt_latent_token_count,
    )
