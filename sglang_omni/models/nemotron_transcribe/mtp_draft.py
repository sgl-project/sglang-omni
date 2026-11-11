# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's native MTP layer as an SGLang EAGLE-style draft for Nemotron Transcribe."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP

from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_language_model_weights,
)


class NemotronTranscribeMTPDraft(NemotronHForCausalLMMTP):
    """Loads llm.mtp.* from the consolidated NeMo checkpoint the target is served from.

    The draft model path is the target checkpoint, so the loader streams every tensor through here; only
    the MTP layer is kept, with its fused experts split the way the decoder's are. Embeddings and lm_head
    come from the target through set_embed_and_head after both models are loaded.
    """

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> None:
        del is_mtp
        super().load_weights(
            checkpoint_to_language_model_weights(weights, is_mtp_draft=True),
            is_mtp=True,
        )


EntryClass = NemotronTranscribeMTPDraft
