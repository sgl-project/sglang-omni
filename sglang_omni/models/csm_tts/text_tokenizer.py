# SPDX-License-Identifier: Apache-2.0
# Prompt assembly mirrors the sesame/csm-1b hub chat template and HuggingFace Transformers
# CSM (transformers/models/csm, Apache-2.0); the raw-tokenizer load follows sglang-omni Higgs.
"""CSM prompt assembly over the checkpoint's Llama-3 tokenizer.

Mirrors the hub chat template EXACTLY (parity is hostage to token-exact
prompt assembly):

- per context message:
  ``128000 ⊕ tok("[<spk>]<text>") ⊕ 128001 ⊕ 128002×F ⊕ 128003``
- final (target) message:
  ``128000 ⊕ tok("[<spk>]<text>") ⊕ 128001``

``add_special_tokens=False`` everywhere; the speaker tag is LITERAL text
``[0]``. Unlike higgs's ``-100`` sentinel, the ``<|AUDIO|>`` positions keep
their REAL id 128002 (in-vocab) — no masking needed; the prefill overlay
simply overwrites those embedding rows.

The tokenizer is loaded from the raw ``tokenizer.json`` (higgs trick — dodges
the transformers-v5 tokenizer-metadata incompatibility).


"""

from __future__ import annotations

import os

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.csm_tts.utils import (
    AUDIO_EOS_TOKEN_ID,
    AUDIO_TOKEN_ID,
    BOS,
    EOT,
)


class CsmPromptBuilder:
    """Token-exact CSM prompt builder."""

    def __init__(self, checkpoint_dir: str) -> None:
        """Load the Llama-3 tokenizer from ``<checkpoint_dir>/tokenizer.json``
        via ``tokenizers.Tokenizer`` wrapped in ``PreTrainedTokenizerFast``
        (higgs text_tokenizer pattern)."""
        # transformers-v5 tokenizer-metadata dodge: bypass from_pretrained and
        # load the raw tokenizer.json (higgs stages.py:198-200).
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
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Assemble the full prompt id sequence + per-segment placeholder spans.

        Args:
            text: target utterance text.
            speaker_id: speaker tag rendered as literal ``[<spk>]`` text.
            context: ordered context segments ``{speaker_id, text}``.
            context_frame_counts: F per context segment — MUST come from the
                actual Mimi encode on the raw-waveform path (deferred prompt
                assembly) or ``codes.shape[0]`` on the
                pre-encoded fast path.

        Returns:
            ``(prompt_ids, placeholder_spans)`` where ``placeholder_spans`` is
            one ``(start, length)`` per context segment covering its
            ``128002×F`` run (the ``128003`` position immediately after gets
            the all-zeros-frame embed at overlay time — HF
            ``_merge_input_ids_with_input_values`` parity).
        """
        context = context or []
        frame_counts = context_frame_counts or []
        if len(frame_counts) != len(context):
            raise ValueError(
                f"context_frame_counts has {len(frame_counts)} entries for "
                f"{len(context)} context segments"
            )

        prompt_ids: list[int] = []
        placeholder_spans: list[tuple[int, int]] = []
        for segment, num_frames in zip(context, frame_counts):
            num_frames = int(num_frames)
            if num_frames <= 0:
                # Hub template: every non-final message must carry audio.
                raise ValueError(
                    f"context segment needs >= 1 audio frame, got {num_frames}"
                )
            prompt_ids.append(BOS)
            prompt_ids.extend(
                self.tokenize_segment_text(
                    segment.get("speaker_id", 0), segment.get("text", "")
                )
            )
            prompt_ids.append(EOT)
            placeholder_spans.append((len(prompt_ids), num_frames))
            prompt_ids.extend([AUDIO_TOKEN_ID] * num_frames)
            prompt_ids.append(AUDIO_EOS_TOKEN_ID)

        prompt_ids.append(BOS)
        prompt_ids.extend(self.tokenize_segment_text(speaker_id, text))
        prompt_ids.append(EOT)
        return prompt_ids, placeholder_spans

    def tokenize_segment_text(self, speaker_id: int, text: str) -> list[int]:
        """``tok("[<spk>]<text>", add_special_tokens=False)`` — body tokens
        only (no BOS/EOT framing).

        Returns:
            list[int] token ids in the 128256 text vocab.
        """
        return self._tok.encode(f"[{int(speaker_id)}]{text}", add_special_tokens=False)


__all__ = [
    "AUDIO_EOS_TOKEN_ID",
    "AUDIO_TOKEN_ID",
    "BOS",
    "EOT",
    "CsmPromptBuilder",
]
