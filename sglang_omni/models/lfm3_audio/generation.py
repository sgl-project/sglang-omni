# SPDX-License-Identifier: Apache-2.0
"""Per-request text/speech state machine of LFM3 generation.

A request is in text, speech (after audio_start) or interleaved (after mixed_start) mode, as in the checkpoint's own generate. Interleaved blocks alternate 6 text slots and 12 speech frames until the text ends with text_end, after which the request speaks until the end-of-speech frame. An end-of-speech frame, any code equal to the EOS code, returns the request to text mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import torch

from sglang_omni.models.lfm3_audio.frame_sampling import FrameSampling
from sglang_omni.models.lfm3_audio.hf_config import Lfm3AudioTokens


class GenerationMode(IntEnum):
    """Also the mode codes of the device speech state (speech_pool.py)."""

    TEXT = 0
    AUDIO = 1
    INTERLEAVED = 2


@dataclass(kw_only=True)
class SpeechState:
    tokens: Lfm3AudioTokens
    sampling: FrameSampling
    mode: GenerationMode = GenerationMode.TEXT
    text_only: bool = False
    speaking: bool = False
    text_done: bool = False
    # Slots left in the current interleaved block, text or speech.
    slots_left: int
    finished: bool = False
    # The generated ids as the checkpoint's generate returns them: text tokens and one audio_out per emitted frame.
    text_out: list[int] = field(default_factory=list)
    frames: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def from_prompt(
        cls,
        tokens: Lfm3AudioTokens,
        last_prompt_token: int,
        *,
        text_only: bool,
        sampling: FrameSampling,
    ) -> SpeechState:
        state = cls(
            tokens=tokens,
            text_only=text_only,
            sampling=sampling,
            slots_left=tokens.interleaved_n_text,
        )
        if text_only:
            state.mode = GenerationMode.TEXT
        elif last_prompt_token in (
            tokens.audio_start_token_id,
            tokens.audio_out_token_id,
        ):
            state.mode = GenerationMode.AUDIO
        elif last_prompt_token == tokens.mixed_start_token_id:
            state.mode = GenerationMode.INTERLEAVED
        else:
            state.mode = GenerationMode.TEXT
        return state

    @property
    def interleaving(self) -> bool:
        return self.mode is GenerationMode.INTERLEAVED

    def needs_frame(self) -> bool:
        return not self.finished and (
            self.mode is GenerationMode.AUDIO or (self.interleaving and self.speaking)
        )

    def apply_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Record a sampled frame; returns the codes to feed back, all end-of-speech codes for an end-of-speech frame."""
        eos_code = self.tokens.audio_eos_code
        if bool((codes == eos_code).any()):
            self.mode = GenerationMode.TEXT
            return torch.full_like(codes, eos_code)
        else:
            pass
        self.sampling.append(codes)
        self.frames.append(codes)
        self.text_out.append(self.tokens.audio_out_token_id)
        return codes

    def apply_text(self, token_id: int, eos_token_ids: tuple[int, ...]) -> None:
        tokens = self.tokens
        self.text_out.append(token_id)
        interleaving = self.interleaving
        if token_id == tokens.text_end_token_id:
            if interleaving:
                self.text_done, self.speaking = True, True
            else:
                self.finished = True
        elif token_id in eos_token_ids:
            self.finished = True
        elif interleaving:
            # note (ruilin): mode markers only open a block; inside one they are ordinary text.
            pass
        elif token_id == tokens.audio_start_token_id:
            self.mode = GenerationMode.AUDIO
        elif token_id == tokens.mixed_start_token_id:
            self.mode = GenerationMode.INTERLEAVED
            self.slots_left = tokens.interleaved_n_text
            self.speaking = False
            self.text_done = False
        else:
            pass

    def advance_slot(self, was_interleaving: bool) -> None:
        """Count one interleaved slot of the step just taken (the mode it started in)."""
        if not was_interleaving or self.text_done:
            return
        else:
            pass
        self.slots_left -= 1
        if self.slots_left == 0:
            self.speaking = not self.speaking
            self.slots_left = (
                self.tokens.interleaved_n_audio
                if self.speaking
                else self.tokens.interleaved_n_text
            )
        else:
            pass
