# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker runner: condition-embeds prefill + windowed rep penalty."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)

# The checkpoint's CustomRepetitionPenaltyLogitsProcessorRepeat scores only
# the most recent window of generated codes.
REP_PENALTY_WINDOW = 16


class MiniCPMOTalkerModelRunner(ModelRunner):
    """Feedback-free codec runner.

    Prefill feeds the projected condition embeddings through the omni
    sidecar; decode is the standard sglang path (``get_input_embeddings`` is
    ``emb_code``). The only sampling divergence from the base runner is the
    repetition penalty: the checkpoint applies a frequency-based penalty
    (``penalty**count``) over a sliding window of the last 16 generated
    codes, while both the base implementation and sglang's native
    ``BatchedRepetitionPenalizer`` (applied inside ``model_runner.sample``)
    are presence-based over the whole output stream — far too aggressive for
    codec sequences that legitimately revisit tokens. The request builder
    therefore keeps ``sampling_params.repetition_penalty`` at 1.0 (both
    presence penalties stay inert) and passes the real penalty through
    ``data.talker_model_inputs["rep_penalty"]``, applied here. Batches stay
    on the sync decode path via the base ``lookahead_eligible`` gate because
    ``sampling_params.min_new_tokens`` is nonzero.
    """

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
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
                # Retract replay: re-embed already-generated codec tokens the
                # same way decode does.
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

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        vocab = logits.shape[1]
        device = logits.device
        penalized_rows: list[int] = []
        penalties: list[float] = []
        windows: list[list[int]] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            penalty = float(data.talker_model_inputs.get("rep_penalty", 1.0))
            if penalty == 1.0:
                continue
            window = [
                tok
                for tok in map(int, data.req.output_ids[-REP_PENALTY_WINDOW:])
                if 0 <= tok < vocab
            ]
            if not window:
                continue
            penalized_rows.append(row_idx)
            penalties.append(penalty)
            windows.append(window)
        if not penalized_rows:
            return
        # Vectorized window counting: pad the ragged windows into a dummy bin
        # at index `vocab` and scatter_add once per batch, instead of building
        # per-request Python count dicts on the decode hot path.
        num = len(windows)
        window_ids = torch.full((num, REP_PENALTY_WINDOW), vocab, dtype=torch.long)
        for i, window in enumerate(windows):
            window_ids[i, : len(window)] = torch.tensor(window, dtype=torch.long)
        window_ids = window_ids.to(device)
        counts = torch.zeros(num, vocab + 1, dtype=torch.float32, device=device)
        counts.scatter_add_(
            1, window_ids, torch.ones_like(window_ids, dtype=torch.float32)
        )
        counts = counts[:, :vocab]
        alphas = (
            torch.tensor(penalties, dtype=torch.float32, device=device).unsqueeze(1)
            ** counts
        )
        rows_t = torch.tensor(penalized_rows, dtype=torch.long, device=device)
        orig_dtype = logits.dtype
        scores = logits[rows_t].to(torch.float32)
        penalized = torch.where(scores < 0, scores * alphas, scores / alphas)
        scores = torch.where(counts > 0, penalized, scores)
        logits[rows_t] = scores.to(orig_dtype)

    def _process_sampling_logits(self, logits_output: Any, requests: list) -> None:
        self._apply_repetition_penalty(logits_output, requests)
