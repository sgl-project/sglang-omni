# SPDX-License-Identifier: Apache-2.0
"""CosyVoice3 model runner for the OmniScheduler AR stage.

CosyVoice3 is a plain single-token AR speech LM: the prompt is supplied as
already-projected hidden states (built in preprocessing), and every decoded
speech token is fed back through ``speech_embedding`` by SGLang's standard
decode path. The runner therefore only needs to install the per-request prompt
embeds onto ``forward_batch.input_embeds`` before the standard prefill forward
(mirrors ``moss_tts``/``qwen3_tts`` ``_build_prefill_input_embeds``); there is no
custom decode forward, feedback buffer, or codebook collection.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner


class CosyVoice3ModelRunner(ModelRunner):
    """Installs CosyVoice3 prompt embeds at prefill; standard AR decode otherwise."""

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        # The CosyVoice3 model reads ``forward_batch.input_embeds`` at prefill and
        # falls back to ``speech_embedding(input_ids)`` at decode, so we only stage
        # the prompt embeds here and let the standard forward path run.
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
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
            prompt_embeds = getattr(data, "prompt_input_embeds", None)
            if prompt_embeds is None:
                raise RuntimeError("CosyVoice3 prefill requires prompt_input_embeds")
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            pieces.append(prompt_embeds[prefix_len : prefix_len + req_len])
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )
