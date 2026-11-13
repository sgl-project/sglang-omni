# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's native MTP layer as an SGLang draft for Nemotron Transcribe."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP

from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_language_model_weights,
)


class NemotronTranscribeMTPDraft(NemotronHForCausalLMMTP):

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> None:
        del is_mtp
        super().load_weights(
            checkpoint_to_language_model_weights(weights, is_mtp_draft=True),
            is_mtp=True,
        )


EntryClass = NemotronTranscribeMTPDraft
