# SPDX-License-Identifier: Apache-2.0
"""YuE2 AR sampling on top of SGLang's forward: masks, windowed penalty and CFG."""

from collections import defaultdict
from dataclasses import dataclass

import torch
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.types import SchedulerRequest

from .generation import filter_scores
from .request import SamplingConfig
from .sglang_request_builder import Yue2ARRequestData


@dataclass(frozen=True, eq=False)
class SamplingRow:
    """One sampled row: its conditional batch row and, under CFG, its twin's row."""

    conditional: int
    unconditional: int | None
    data: Yue2ARRequestData


def sampling_rows(requests: list[SchedulerRequest]) -> list[SamplingRow]:
    rows: list[SamplingRow] = []
    index = 0
    while index < len(requests):
        data: Yue2ARRequestData = requests[index].data
        if data.is_cfg_uncond:
            raise RuntimeError(
                f"YuE2 CFG row {requests[index].request_id} is not preceded by its request"
            )
        elif data.cfg_uncond is None:
            rows.append(SamplingRow(index, None, data))
            index += 1
        elif index + 1 < len(requests) and requests[index + 1].data is data.cfg_uncond:
            rows.append(SamplingRow(index, index + 1, data))
            index += 2
        else:
            raise RuntimeError(
                f"YuE2 request {requests[index].request_id} lost its adjacent CFG row"
            )
    return rows


def sampling_group_key(data: Yue2ARRequestData) -> tuple[str, str, bool]:
    assert data.sampling is not None
    return (data.phase, data.sampling.model_dump_json(), data.legacy_off)


class Yue2ModelRunner(ModelRunner[Yue2ARRequestData]):
    """Sample every YuE2 row itself so CFG twins receive their request's token."""

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch
        if schedule_batch.is_prefill_only or not requests:
            return
        else:
            self.sample(result, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch, schedule_batch
        if requests:
            self.sample(result, requests)
        else:
            pass

    def sample(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> None:
        logits = result.logits_output.next_token_logits
        if logits.shape[0] != len(requests):
            raise RuntimeError(
                f"YuE2 expected one logits row per request, got {logits.shape[0]} "
                f"for {len(requests)}"
            )
        else:
            pass
        device = logits.device
        rows = sampling_rows(requests)
        guided = [
            position
            for position, row in enumerate(rows)
            if row.unconditional is not None
        ]
        combined = logits[[row.conditional for row in rows]]
        if guided:
            conditional = combined[guided]
            unconditional = logits[
                [rows[position].unconditional for position in guided]
            ]
            scales = torch.tensor(
                [rows[position].data.cfg_scale for position in guided],
                dtype=logits.dtype,
                device=device,
            )[:, None]
            combined[guided] = unconditional + scales * (conditional - unconditional)
        else:
            pass

        groups: defaultdict[tuple[str, str, bool], list[int]] = defaultdict(list)
        for position, row in enumerate(rows):
            groups[sampling_group_key(row.data)].append(position)
        sampled = torch.empty(len(rows), dtype=torch.long, device=device)
        for (phase, _, legacy_off), positions in groups.items():
            members = [rows[position].data for position in positions]
            group_logits = combined[positions]
            if legacy_off:
                group_logits = group_logits.to(torch.bfloat16)
            else:
                pass
            sampled[positions] = self.sample_group(
                group_logits, members, phase, members[0].sampling, legacy_off
            )

        next_token_ids = torch.empty(len(requests), dtype=torch.long, device=device)
        next_token_ids[[row.conditional for row in rows]] = sampled
        if guided:
            next_token_ids[[rows[position].unconditional for position in guided]] = (
                sampled[guided]
            )
        else:
            pass
        result.next_token_ids = next_token_ids

    @staticmethod
    def sample_group(
        logits: torch.Tensor,
        members: list[Yue2ARRequestData],
        phase: str,
        sampling: SamplingConfig,
        legacy_off: bool,
    ) -> torch.Tensor:
        device = logits.device
        window = sampling.penalty_window
        for data in members:
            if data.recent_tokens is None:
                data.recent_tokens = torch.full(
                    (window,), -1, dtype=torch.long, device=device
                )
            else:
                pass
        recent = torch.stack([data.recent_tokens for data in members])
        frequencies = torch.zeros_like(
            logits, dtype=logits.dtype if legacy_off else torch.float32
        )
        if sampling.repetition_penalty != 1.0:
            frequencies.scatter_add_(
                -1, recent.clamp_min(0), (recent >= 0).to(frequencies.dtype)
            )
        else:
            pass
        allow_end = torch.tensor(
            [data.generated_tokens >= sampling.min_tokens for data in members],
            device=device,
        )
        scores = filter_scores(
            logits, sampling, frequencies, allow_end, phase, legacy_off
        )
        if sampling.temperature == 0:
            tokens = scores.argmax(-1)
        else:
            tokens = multinomial_with_seed(
                scores,
                torch.tensor(
                    [data.sampling_seed for data in members],
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor(
                    [data.generated_tokens for data in members],
                    dtype=torch.int64,
                    device=device,
                ),
            ).view(-1)
        slots = torch.tensor(
            [data.generated_tokens % window for data in members], device=device
        )
        recent[torch.arange(len(members), device=device), slots] = tokens
        for index, data in enumerate(members):
            data.recent_tokens = recent[index]
            data.generated_tokens += 1
        return tokens
