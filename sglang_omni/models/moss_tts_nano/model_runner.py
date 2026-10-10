# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Nano model runner for OmniScheduler."""

from __future__ import annotations

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class MossTTSNanoModelRunner(MossTTSLocalModelRunner):
    """Reuse the local-frame scheduler with Nano-safe radix token ids."""

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        config = self.model.config
        self.radix_hash_space = int(config.vocab_size)
        self.radix_hash_offset = (
            max(
                int(config.pad_token_id),
                int(config.im_start_token_id),
                int(config.im_end_token_id),
                int(config.audio_start_token_id),
                int(config.audio_end_token_id),
                int(config.audio_user_slot_token_id),
                int(config.audio_assistant_slot_token_id),
            )
            + 1
        )


EntryClass = MossTTSNanoModelRunner
