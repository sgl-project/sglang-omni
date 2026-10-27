# SPDX-License-Identifier: Apache-2.0
"""CSM prompt assembly over the checkpoint Llama-3 tokenizer.

Context: 128000 + tok("[spk]text") + 128001 + 128002*F + 128003.
Target: 128000 + tok("[spk]text") + 128001. 128002 is a real token id.
"""

from __future__ import annotations

import os

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.csm_tts.utils import (
    AUDIO_EOS_TOKEN_ID,
    AUDIO_TOKEN_ID,
    BOS_TOKEN_ID,
    EOT_TOKEN_ID,
)


class CsmTokenizerAdapter:
    def __init__(self, checkpoint_dir: str) -> None:
        raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
        self._tok = PreTrainedTokenizerFast(tokenizer_object=raw)

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tok

    def build_prompt(
        self,
        *,
        text: str,
        speaker_id: int,
        context: list[dict] | None = None,
        context_frame_counts: list[int] | None = None,
    ) -> list[int]:
        """Assemble prompt ids. Audio placeholders stay 128002 / 128003."""
        context = context or []
        frame_counts = context_frame_counts or []
        if len(frame_counts) != len(context):
            raise ValueError(
                f"context_frame_counts has {len(frame_counts)} entries for "
                f"{len(context)} context segments"
            )

        prompt_token_ids: list[int] = []
        for segment, num_frames in zip(context, frame_counts):
            num_frames = int(num_frames)
            if num_frames <= 0:
                raise ValueError(
                    f"context segment needs >= 1 audio frame, got {num_frames}"
                )
            prompt_token_ids.append(BOS_TOKEN_ID)
            prompt_token_ids.extend(
                self.tokenize_segment_text(
                    segment.get("speaker_id", 0), segment.get("text", "")
                )
            )
            prompt_token_ids.append(EOT_TOKEN_ID)
            prompt_token_ids.extend([AUDIO_TOKEN_ID] * num_frames)
            prompt_token_ids.append(AUDIO_EOS_TOKEN_ID)

        prompt_token_ids.append(BOS_TOKEN_ID)
        prompt_token_ids.extend(self.tokenize_segment_text(speaker_id, text))
        prompt_token_ids.append(EOT_TOKEN_ID)
        return prompt_token_ids

    def tokenize_segment_text(self, speaker_id: int, text: str) -> list[int]:
        """Tokenize speaker-tagged text without BOS/EOT."""
        return self._tok.encode(f"[{int(speaker_id)}]{text}", add_special_tokens=False)


__all__ = [
    "AUDIO_EOS_TOKEN_ID",
    "AUDIO_TOKEN_ID",
    "BOS_TOKEN_ID",
    "EOT_TOKEN_ID",
    "CsmTokenizerAdapter",
]
