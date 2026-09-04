# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context

from .sglang_model import VOCAB_SIZE


class FunCosyVoice3ModelRunner(ModelRunner):
    """Runs Fun-CosyVoice3 AR steps and collects generated speech tokens."""

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_tokens(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_tokens(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> bool:
        """Sample the first speech token before collecting prefill output."""
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> bool:
        """Sample each speech token before collecting decode output."""
        del forward_batch, schedule_batch, requests
        return True

    def _collect_tokens(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if result.next_token_ids is None:
            return
        token_ids = result.next_token_ids
        if token_ids.ndim != 1:
            token_ids = token_ids.reshape(-1)
        # note: copy the whole batch to host once instead of calling
        # ``.item()`` per request — each ``.item()`` forces its own
        # host/GPU synchronization, which is a per-decode-step cost that
        # scales with batch size.
        token_ids_cpu = token_ids.tolist()
        for idx, sched_req in enumerate(requests):
            token_id = int(token_ids_cpu[idx])
            if token_id >= VOCAB_SIZE:
                continue
            sched_req.data.output_codes.append(
                torch.tensor([token_id], dtype=torch.long)
            )

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            prompt_embeds = data.prompt_input_embeds
            if prompt_embeds is None:
                raise RuntimeError(
                    "Fun-CosyVoice3 prefill requires prompt_input_embeds"
                )
            pieces.append(prompt_embeds[prefix_len : prefix_len + req_len])
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
