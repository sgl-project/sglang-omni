# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker runner: condition-embeds prefill + windowed rep penalty."""

from __future__ import annotations

from collections import Counter

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.scheduling.types import SchedulerRequest

# note (MayDomine): the checkpoint penalizes only the most recent 16 codec tokens.
REP_PENALTY_WINDOW = 16


class MiniCPMOTalkerModelRunner(ModelRunner):
    """Prefill codec conditions and apply a frequency penalty over recent tokens."""

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Prepare request embeddings; schedule_batch follows the runner interface."""
        parts: list[torch.Tensor] = []
        for sched_req in requests:
            data = sched_req.data
            tensor = data.prefill_input_embeds
            if tensor is None:
                raise RuntimeError(
                    "MiniCPM-o talker prefill requires condition embeddings"
                )
            req = data.req
            prefix_len = len(req.prefix_indices)
            end = prefix_len + int(req.extend_range.length)
            prompt_len = int(tensor.shape[0])
            if prefix_len < prompt_len:
                parts.append(tensor[prefix_len : min(end, prompt_len)])
            if end > prompt_len:
                # note (MayDomine): retracted requests replay already-generated tokens.
                fill_ids = req.get_fill_ids()
                generated = torch.tensor(
                    fill_ids[max(prefix_len, prompt_len) : end],
                    dtype=torch.long,
                    device=self.model.emb_code.weight.device,
                )
                parts.append(self.model.emb_code(generated))
        input_embeds = torch.cat(parts, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model.emb_code.weight.dtype,
        )
        expected_rows = int(forward_batch.input_ids.shape[0])
        if input_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "Talker prefill embeds must align with forward input_ids: "
                f"got {input_embeds.shape[0]} rows for {expected_rows} input ids"
            )
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=input_embeds,
                input_embeds_are_projected=True,
            ),
        )

    def process_sampling_logits(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        else:
            vocabulary_size = logits.shape[1]
            token_coordinates: list[tuple[int, int]] = []
            penalty_frequencies: list[tuple[float, int]] = []
            for row_index, request in enumerate(requests):
                penalty = float(
                    request.data.talker_model_inputs.get("rep_penalty", 1.0)
                )
                if penalty == 1.0 or not request.data.req.output_ids:
                    continue
                else:
                    frequencies = Counter(
                        token_id
                        for token_id in map(
                            int, request.data.req.output_ids[-REP_PENALTY_WINDOW:]
                        )
                        if 0 <= token_id < vocabulary_size
                    )
                    for token_id, frequency in frequencies.items():
                        token_coordinates.append((row_index, token_id))
                        penalty_frequencies.append((penalty, frequency))

            if not token_coordinates:
                return
            else:
                # note (koppx): unique coordinates preserve counts without repeated writes.
                coordinates = torch.tensor(
                    token_coordinates, dtype=torch.long, device=logits.device
                )
                penalties = torch.tensor(
                    penalty_frequencies, dtype=torch.float32, device=logits.device
                )
                scaling_factors = penalties[:, 0].pow(penalties[:, 1])
                row_indices, token_indices = coordinates.unbind(dim=1)
                scores = logits[row_indices, token_indices].to(torch.float32)
                penalized_scores = torch.where(
                    scores < 0, scores * scaling_factors, scores / scaling_factors
                )
                logits[row_indices, token_indices] = penalized_scores.to(logits.dtype)
