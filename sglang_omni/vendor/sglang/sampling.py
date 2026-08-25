# SPDX-License-Identifier: Apache-2.0
"""Vendor wrapper for sglang.srt.sampling.sampling_batch_info.

Patches ``SamplingBatchInfo.from_schedule_batch`` to vectorize ``logit_bias``
construction. The upstream implementation builds the dense bias tensor with
a nested Python loop that writes one scalar GPU-tensor element at a time:

    logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
    for i, r in enumerate(reqs):
        if r.sampling_params.logit_bias is not None:
            for key, value in r.sampling_params.logit_bias.items():
                logit_bias[i, int(key)] = value

Each assignment is its own tiny CUDA kernel launch. Requests that apply
``suppress_tokens`` via ``logit_bias`` (e.g. Whisper ASR, which suppresses
the same ~90-token set on every request) turn this into dozens-to-hundreds
of launches per call, and this classmethod runs on every extend/prefill
batch-composition step. We instead collect all (row, col, value) triples on
the CPU and materialize the tensor with a single ``index_put_`` call.

Captured against sglang==0.5.16
(sglang/srt/sampling/sampling_batch_info.py, lines 82-217). This is a
full-method override (the hot loop is inline in a much larger classmethod,
not separable into a call-through wrapper) -- re-diff
``from_schedule_batch`` against upstream on version bumps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import sglang.srt.sampling.penaltylib as penaltylib
import torch
from sglang.srt.runtime_context import get_server_args
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def _build_logit_bias(reqs, vocab_size: int, device) -> Optional[torch.Tensor]:
    if not any(r.sampling_params.logit_bias is not None for r in reqs):
        return None

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for i, r in enumerate(reqs):
        bias = r.sampling_params.logit_bias
        if not bias:
            continue
        for key, value in bias.items():
            rows.append(i)
            cols.append(int(key))
            values.append(float(value))

    logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
    if rows:
        row_idx = torch.tensor(rows, dtype=torch.long, device=device)
        col_idx = torch.tensor(cols, dtype=torch.long, device=device)
        val = torch.tensor(values, dtype=logit_bias.dtype, device=device)
        logit_bias.index_put_((row_idx, col_idx), val)
    return logit_bias


@classmethod
def _from_schedule_batch(cls, batch: "ScheduleBatch", vocab_size: int):
    global_server_args = get_server_args()
    enable_deterministic = global_server_args.enable_deterministic_inference

    reqs = batch.reqs
    device = batch.device
    _pin = is_pin_memory_available(device)
    temperatures = (
        torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            pin_memory=_pin,
        )
        .to(device, non_blocking=True)
        .view(-1, 1)
    )
    top_ps = torch.tensor(
        [r.sampling_params.top_p for r in reqs],
        dtype=torch.float,
        pin_memory=_pin,
    ).to(device, non_blocking=True)
    top_ks = torch.tensor(
        [r.sampling_params.top_k for r in reqs],
        dtype=torch.int32,
        pin_memory=_pin,
    ).to(device, non_blocking=True)
    min_ps = torch.tensor(
        [r.sampling_params.min_p for r in reqs],
        dtype=torch.float,
        pin_memory=_pin,
    ).to(device, non_blocking=True)
    sampling_seed = (
        torch.tensor(
            [
                (
                    r.sampling_params.sampling_seed
                    if r.sampling_params.sampling_seed is not None
                    else 42
                )
                for r in reqs
            ],
            dtype=torch.int64,
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        if enable_deterministic
        else None
    )

    # omni patch: vectorized logit_bias construction, see _build_logit_bias above.
    logit_bias = _build_logit_bias(reqs, vocab_size, device)

    # Check if any request has custom logit processor
    has_custom_logit_processor = (
        global_server_args.enable_custom_logit_processor
        and any(r.custom_logit_processor for r in reqs)  # check the flag first.
    )  # then check the requests.
    return_sampling_masks = [r.return_sampling_mask for r in reqs]
    sampling_mask_max_top_k = max(
        (r.sampling_params.top_k for r in reqs if r.return_sampling_mask),
        default=0,
    )

    if has_custom_logit_processor:
        # Merge the same type of custom logit processors together
        processor_dict = {}
        for i, r in enumerate(reqs):
            if r.custom_logit_processor is None:
                continue
            processor_str = r.custom_logit_processor
            if processor_str not in processor_dict:
                processor_dict[processor_str] = []
            processor_dict[processor_str].append(i)

        merged_custom_logit_processor = {
            hash(processor_str): (
                # The deserialized custom logit processor object
                CustomLogitProcessor.from_str(processor_str),
                # The mask tensor for the requests that use this custom logit processor
                torch.zeros(len(reqs), dtype=torch.bool)
                .scatter_(0, torch.tensor(true_indices), True)
                .to(device, non_blocking=True),
            )
            for processor_str, true_indices in processor_dict.items()
        }
        custom_params = [r.sampling_params.custom_params for r in reqs]
    else:
        merged_custom_logit_processor = None
        custom_params = None

    # Each penalizers will do nothing if they evaluate themselves as not required by looking at
    # the sampling_params of the requests (See {_is_required()} of each penalizers). So this
    # should not add hefty computation overhead other than simple checks.
    #
    # While we can choose not to even create the class instances if they are not required, this
    # could add additional complexity to the {ScheduleBatch} class, especially we need to
    # handle {filter_batch()} and {merge_batch()} cases as well.
    penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
        vocab_size=vocab_size,
        batch=batch,
        penalizers={
            penaltylib.BatchedFrequencyPenalizer,
            penaltylib.BatchedMinNewTokensPenalizer,
            penaltylib.BatchedPresencePenalizer,
            penaltylib.BatchedRepetitionPenalizer,
        },
    )

    ret = cls(
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
        min_ps=min_ps,
        sampling_seed=sampling_seed,
        is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
        is_any_greedy=any(r.sampling_params.top_k <= 1 for r in reqs),
        need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
        need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
        need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
        vocab_size=vocab_size,
        penalizer_orchestrator=penalizer_orchestrator,
        has_custom_logit_processor=has_custom_logit_processor,
        custom_params=custom_params,
        custom_logit_processor=merged_custom_logit_processor,
        device=device,
        logit_bias=logit_bias,
        return_sampling_masks=return_sampling_masks,
        sampling_mask_max_top_k=sampling_mask_max_top_k,
    )
    ret.adjusted_from_schedule_batch(batch, vocab_size)
    return ret


# Patch the source class so any direct imports also see the change.
SamplingBatchInfo.from_schedule_batch = _from_schedule_batch

__all__ = ["SamplingBatchInfo"]
