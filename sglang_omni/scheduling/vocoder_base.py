# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import PipelineStateBase
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

__all__ = ["BatchVocoderBase", "group_by_padding_waste"]


def group_by_padding_waste(
    lengths: Sequence[int], max_padding_waste: float
) -> list[list[int]]:
    """Group indices by ascending length while each group's padded size stays
    within max_padding_waste times its summed real length."""
    groups: list[list[int]] = []
    group: list[int] = []
    total = 0
    for index in sorted(range(len(lengths)), key=lengths.__getitem__):
        length = lengths[index]
        if group and length * (len(group) + 1) > max_padding_waste * (total + length):
            groups.append(group)
            group, total = [], 0
        group.append(index)
        total += length
    if group:
        groups.append(group)
    return groups


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

    async def decode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self.prepare_item(payload) for payload in payloads]
        results = await self.decode_batch(items)
        if len(results) != len(items):
            raise RuntimeError(
                f"decode_batch returned {len(results)} results for {len(items)} inputs"
            )
        else:
            pass
        return [
            self.store_result(payload, state, wav, sample_rate)
            for payload, (state, _), (wav, sample_rate) in zip(
                payloads, items, results, strict=True
            )
        ]

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
            else:
                pass
            wav, sr = results[0]
            return self.store_result(payload, state, wav, sr)

        return SimpleScheduler(
            _single,
            batch_compute_fn=self.decode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )
