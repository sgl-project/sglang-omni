# SPDX-License-Identifier: Apache-2.0
"""Prompt construction for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sglang_omni.models.moss_tts_realtime.payload_types import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_PAD_TOKEN,
    N_CODEBOOKS,
    PREFILL_TEXT_TOKENS,
)

_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are a highly expressive text-to-speech (TTS) engine developed by "
    "Mosi Intelligence. \n"
    "You possess natural language understanding, emotional modeling, and "
    "multi-style speech generation capabilities, allowing you to generate "
    "the corresponding speech based on the text given in the assistant."
    "<|im_end|>\n"
)
_VOICE_CONTEXT = (
    "<|im_start|>context\n"
    "The assistant section should be synthesized using the following voice "
    "timbre:{audio}<|im_end|>\n"
)
_ASSISTANT_PREFIX = "<|im_start|>assistant\n"


class MossTTSRealtimePromptProcessor:
    """Build the 17-channel prompt consumed by MOSS-TTS-Realtime."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.audio_pad_token_id = int(tokenizer.convert_tokens_to_ids("<|audio_pad|>"))

    def _text_rows(self, text: str) -> np.ndarray:
        token_ids = self.tokenizer(text)["input_ids"]
        rows = np.full(
            (len(token_ids), N_CODEBOOKS + 1),
            AUDIO_PAD_TOKEN,
            dtype=np.int64,
        )
        rows[:, 0] = token_ids
        return rows

    @staticmethod
    def _normalize_reference_codes(codes: torch.Tensor | np.ndarray) -> np.ndarray:
        values = np.asarray(
            codes.detach().cpu().numpy() if isinstance(codes, torch.Tensor) else codes
        )
        values = np.squeeze(values)
        if values.ndim != 2:
            raise ValueError(
                f"MOSS-TTS-Realtime reference codes must be rank 2, got {values.shape}"
            )
        if values.shape[0] == N_CODEBOOKS:
            values = values.T
        elif values.shape[1] != N_CODEBOOKS:
            raise ValueError(
                "MOSS-TTS-Realtime reference codes must contain 16 codebooks, "
                f"got {values.shape}"
            )
        return values.astype(np.int64, copy=False)

    def make_ensemble(
        self, reference_codes: torch.Tensor | np.ndarray | None
    ) -> np.ndarray:
        prompt = _SYSTEM_PROMPT
        normalized = None
        if reference_codes is not None:
            normalized = self._normalize_reference_codes(reference_codes)
            prompt += _VOICE_CONTEXT.format(
                audio="<|audio_pad|>" * int(normalized.shape[0])
            )
        prompt += _ASSISTANT_PREFIX
        rows = self._text_rows(prompt)
        if normalized is not None:
            indices = np.flatnonzero(rows[:, 0] == self.audio_pad_token_id)
            if int(indices.size) != int(normalized.shape[0]):
                raise ValueError(
                    "MOSS-TTS-Realtime voice prompt token count does not match "
                    "the encoded reference length"
                )
            rows[indices, 1:] = normalized
        return rows

    _TURN_BREAK = "<|im_end|>\n" + _ASSISTANT_PREFIX

    def build_history_turn_rows(
        self,
        text: str,
        audio_codes: torch.Tensor | np.ndarray,
        *,
        leading_break: bool,
    ) -> np.ndarray:
        """Render one completed assistant turn (text + generated codes).

        The layout mirrors the MOSS-TTS-Realtime finetuning packer exactly
        (text channel with ``<|text_pad|>`` padding, audio delayed by
        ``PREFILL_TEXT_TOKENS``, BOS before and EOS after the codes), so a
        multi-turn session prompt matches what the model saw in training.
        """
        prefill_temp = self._TURN_BREAK if leading_break else ""
        raw = np.asarray(
            audio_codes.detach().cpu().numpy()
            if isinstance(audio_codes, torch.Tensor)
            else audio_codes
        )
        # note (luojiaxuan): normalize to [T, C] here instead of reusing the
        # reference normalizer — its np.squeeze collapses both a zero-frame turn (the
        # delay pattern may defer a short turn's audio entirely) and a
        # single-frame turn ((1,16) or (16,1) -> (16,)), which are valid here.
        raw = np.squeeze(raw)
        if raw.size == 0:
            codes = np.zeros((0, N_CODEBOOKS), dtype=np.int64)
        elif raw.ndim == 1 and raw.size == N_CODEBOOKS:
            codes = raw.reshape(1, N_CODEBOOKS).astype(np.int64)
        elif raw.ndim == 2:
            if raw.shape[1] != N_CODEBOOKS and raw.shape[0] == N_CODEBOOKS:
                raw = raw.T
            if raw.shape[1] != N_CODEBOOKS:
                raise ValueError(
                    "MOSS-TTS-Realtime history codes must contain 16 codebooks, "
                    f"got {raw.shape}"
                )
            codes = raw.astype(np.int64, copy=False)
        else:
            raise ValueError(
                f"MOSS-TTS-Realtime history codes must be rank 2, got {raw.shape}"
            )
        text_tokens = self.tokenizer(text)["input_ids"]
        text_start_pos = len(self.tokenizer(prefill_temp)["input_ids"]) if prefill_temp else 0
        text_len = len(text_tokens)
        audio_len = int(codes.shape[0])

        if text_len >= PREFILL_TEXT_TOKENS:
            padded_text_len = audio_len + PREFILL_TEXT_TOKENS - text_len + 1
            ch1_ids = self.tokenizer(
                prefill_temp + text + "<|text_pad|>" * max(0, padded_text_len)
            )["input_ids"]
            rows = np.full(
                (len(ch1_ids), N_CODEBOOKS + 1), AUDIO_PAD_TOKEN, dtype=np.int64
            )
            rows[:, 0] = ch1_ids
            audio_start = text_start_pos + PREFILL_TEXT_TOKENS
            rows[audio_start : audio_start + audio_len, 1:] = codes
            rows[audio_start - 1, 1] = AUDIO_BOS_TOKEN
            rows[audio_start + audio_len, 1] = AUDIO_EOS_TOKEN
        else:
            padded_text_len = audio_len + 1
            ch1_ids = self.tokenizer(
                prefill_temp + text + "<|text_pad|>" * padded_text_len
            )["input_ids"]
            rows = np.full(
                (len(ch1_ids), N_CODEBOOKS + 1), AUDIO_PAD_TOKEN, dtype=np.int64
            )
            rows[:, 0] = ch1_ids
            rows[-(audio_len + 1) : -1, 1:] = codes
            rows[-(audio_len + 2), 1] = AUDIO_BOS_TOKEN
            rows[-1, 1] = AUDIO_EOS_TOKEN
        return rows

    def build_generation_prompt(
        self,
        text: str,
        reference_codes: torch.Tensor | np.ndarray | None,
        history: list[tuple[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, list[int], int]:
        text_ids = list(self.tokenizer(text)["input_ids"])
        if not text_ids:
            raise ValueError("MOSS-TTS-Realtime input text must not be empty")
        prefill_count = min(len(text_ids), PREFILL_TEXT_TOKENS)
        text_rows = np.full(
            (prefill_count, N_CODEBOOKS + 1),
            AUDIO_PAD_TOKEN,
            dtype=np.int64,
        )
        text_rows[:, 0] = text_ids[:prefill_count]
        text_rows[-1, 1] = AUDIO_BOS_TOKEN
        parts = [self.make_ensemble(reference_codes)]
        for turn_idx, (turn_text, turn_codes) in enumerate(history or []):
            parts.append(
                self.build_history_turn_rows(
                    turn_text, turn_codes, leading_break=turn_idx > 0
                )
            )
        if history:
            parts.append(self._text_rows(self._TURN_BREAK))
        parts.append(text_rows)
        rows = np.concatenate(parts)
        return torch.from_numpy(rows), text_ids, prefill_count


__all__ = ["MossTTSRealtimePromptProcessor"]
