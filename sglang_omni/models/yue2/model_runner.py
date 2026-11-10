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

from .generation import masked_scores
from .request import SamplingConfig
from .sglang_request_builder import Yue2ARRequestData


@dataclass(frozen=True, eq=False)
class SamplingRow:
    """One sampled row: its conditional batch row and, under CFG, its twin's row."""

    conditional: int
    unconditional: int | None
    data: Yue2ARRequestData


def sampling_rows(requests: list[SchedulerRequest]) -> list[SamplingRow]:
    # note (ruiling): SGLang reorders the batch when it retracts requests, so twins are matched by identity.
    positions = {id(request.data): index for index, request in enumerate(requests)}
    rows: list[SamplingRow] = []
    for index, request in enumerate(requests):
        data: Yue2ARRequestData = request.data
        if data.is_cfg_uncond:
            pass
        elif data.cfg_uncond is None:
            rows.append(SamplingRow(index, None, data))
        elif id(data.cfg_uncond) in positions:
            rows.append(SamplingRow(index, positions[id(data.cfg_uncond)], data))
        else:
            raise RuntimeError(
                f"YuE2 request {request.request_id} lost its CFG row in this batch"
            )
    if len(rows) + sum(row.unconditional is not None for row in rows) != len(requests):
        raise RuntimeError("YuE2 CFG row without its request in this batch")
    else:
        pass
    return rows


def to_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a small host tensor without blocking the stream on CUDA."""
    if device.type == "cuda":
        return values.pin_memory().to(device, non_blocking=True)
    else:
        return values.to(device)


def device_index(values: list[int], device: torch.device) -> torch.Tensor:
    return to_device(torch.tensor(values, dtype=torch.long), device)


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

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        # note (ruiling): async decode launches the next step from these tokens before the host reads them.
        if requests:
            self.sample(result, requests)
        else:
            pass
        return super().post_decode_launch(result, forward_batch, requests)

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
        if guided:
            conditional_rows = device_index([row.conditional for row in rows], device)
            guided_rows = device_index(guided, device)
            twin_rows = device_index(
                [rows[position].unconditional for position in guided], device
            )
            scales = to_device(
                torch.tensor(
                    [rows[position].data.cfg_scale for position in guided],
                    dtype=logits.dtype,
                ),
                device,
            )[:, None]
            combined = logits.index_select(0, conditional_rows)
            conditional = combined.index_select(0, guided_rows)
            unconditional = logits.index_select(0, twin_rows)
            combined.index_copy_(
                0, guided_rows, unconditional + scales * (conditional - unconditional)
            )
        else:
            combined = logits

        groups: defaultdict[tuple[str, str, bool], list[int]] = defaultdict(list)
        for position, row in enumerate(rows):
            groups[sampling_group_key(row.data)].append(position)
        if len(groups) == 1:
            (phase, _, legacy_off), _ = next(iter(groups.items()))
            sampled = self.sample_group(
                combined, [row.data for row in rows], phase, legacy_off
            )
        else:
            sampled = torch.empty(len(rows), dtype=torch.long, device=device)
            for (phase, _, legacy_off), positions in groups.items():
                group_rows = device_index(positions, device)
                sampled.index_copy_(
                    0,
                    group_rows,
                    self.sample_group(
                        combined.index_select(0, group_rows),
                        [rows[position].data for position in positions],
                        phase,
                        legacy_off,
                    ),
                )

        if guided:
            next_token_ids = torch.empty(len(requests), dtype=torch.long, device=device)
            next_token_ids.index_copy_(0, conditional_rows, sampled)
            next_token_ids.index_copy_(
                0, twin_rows, sampled.index_select(0, guided_rows)
            )
            result.next_token_ids = next_token_ids
        else:
            result.next_token_ids = sampled

    @staticmethod
    def sample_group(
        logits: torch.Tensor,
        members: list[Yue2ARRequestData],
        phase: str,
        legacy_off: bool,
    ) -> torch.Tensor:
        """Mask, penalize and sample rows that share one sampling configuration."""
        sampling: SamplingConfig = members[0].sampling
        device = logits.device
        window = sampling.penalty_window
        for data in members:
            if data.recent_tokens is None:
                data.recent_tokens = torch.full(
                    (window,), -1, dtype=torch.long, device=device
                )
            else:
                pass
        # note (ruiling): one copy per step; building each row tensor on the device would sync the stream per tensor.
        row_values = to_device(
            torch.tensor(
                [
                    [
                        data.sampling_seed,
                        data.generated_tokens,
                        data.generated_tokens % window,
                        int(data.generated_tokens >= sampling.min_tokens),
                    ]
                    for data in members
                ],
                dtype=torch.long,
            ),
            device,
        )
        seeds, positions, slots, allow_end = row_values.unbind(-1)
        recent = torch.stack([data.recent_tokens for data in members])
        scores = masked_scores(
            logits, sampling, recent, allow_end.bool(), phase, legacy_off
        )
        if sampling.temperature == 0:
            tokens = scores.argmax(-1)
        else:
            # note (ruiling): top-p over the top-k candidates equals top-p over the top-k-masked
            # vocabulary, and dividing by the temperature does not change which tokens are top-k.
            values, candidates = scores.topk(min(sampling.top_k, scores.shape[-1]))
            values = values / sampling.temperature
            if sampling.top_p < 1:
                probabilities = values.softmax(-1)
                removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
                removed[..., : 3 if legacy_off else 1] = False
                values = values.masked_fill(removed, -torch.inf)
            else:
                pass
            choice = multinomial_with_seed(values, seeds, positions).view(-1, 1)
            tokens = candidates.gather(-1, choice).view(-1)
        recent[torch.arange(len(members), device=device), slots] = tokens
        for index, data in enumerate(members):
            data.recent_tokens = recent[index]
            data.generated_tokens += 1
        return tokens
