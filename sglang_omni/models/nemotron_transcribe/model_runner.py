# SPDX-License-Identifier: Apache-2.0
"""Omni model runner for Nemotron Transcribe: composes prefill embeddings a speculative draft can reuse."""

from __future__ import annotations

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.nemotron_transcribe.request_builders import (
    NemotronTranscribeRequestData,
)
from sglang_omni.scheduling.types import SchedulerRequest


class NemotronTranscribeModelRunner(ModelRunner[NemotronTranscribeRequestData]):
    """Attaches the audio-aware prompt embeddings to the prefill batch when a draft will extend over them.

    The EAGLE-style MTP draft rebuilds its KV over the prompt from the same embeddings the decoder saw, audio
    positions included. The upstream eager runner forwards a copy of the batch, so the sidecar is attached here,
    on the batch the runner retains, and the model forward reads it back as input_embeds.
    """

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        if (
            forward_batch is None
            or not self.retain_prefill_input_embeds
            or not forward_batch.contains_mm_inputs()
        ):
            return
        else:
            pass
        model = self.tp_worker.model_runner.model
        input_embeds = model.compose_input_embeds(
            forward_batch.input_ids, forward_batch, {}
        )
        attach_omni_prefill_inputs(
            forward_batch, OmniPrefillInputs(input_embeds=input_embeds)
        )
