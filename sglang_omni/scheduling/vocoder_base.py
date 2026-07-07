# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import PipelineStateBase
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

__all__ = ["BatchVocoderBase"]


class BatchVocoderBase:
    def prepare_item(self, payload: StagePayload) -> tuple[PipelineStateBase, Any]:
        raise NotImplementedError

    async def decode_batch(
        self, items: list[tuple[PipelineStateBase, Any]]
    ) -> list[tuple[torch.Tensor, int]]:
        raise NotImplementedError

    def store_result(
        self,
        payload: StagePayload,
        state: PipelineStateBase,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        raise NotImplementedError

    def build_scheduler(
        self, *, max_batch_size: int = 8, max_batch_wait_ms: int = 2
    ) -> SimpleScheduler:
        async def _single(payload):
            state, codes = self.prepare_item(payload)
            results = await self.decode_batch([(state, codes)])
            if len(results) != 1:
                raise RuntimeError(
                    f"decode_batch returned {len(results)} results for 1 input"
                )
            wav, sr = results[0]
            return self.store_result(payload, state, wav, sr)

        async def _batch(payloads):
            items = [self.prepare_item(p) for p in payloads]
            results = await self.decode_batch(items)
            if len(results) != len(items):
                raise RuntimeError(
                    f"decode_batch returned {len(results)} results for {len(items)} inputs"
                )
            return [
                self.store_result(p, s, wav, sr)
                for p, (s, _), (wav, sr) in zip(payloads, items, results)
            ]

        return SimpleScheduler(
            _single,
            batch_compute_fn=_batch,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )
