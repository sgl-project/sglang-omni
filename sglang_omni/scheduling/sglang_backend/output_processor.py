# SPDX-License-Identifier: Apache-2.0
"""Converts SGLang GenerationBatchResult to per-request RequestOutputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch

from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
else:
    pass


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def __init__(
        self,
        capture_hidden: bool = False,
        should_emit_hidden: Callable[[SchedulerRequest], bool] | None = None,
    ) -> None:
        self.capture_hidden = capture_hidden
        self.should_emit_hidden = should_emit_hidden

    def process(
        self,
        model_output: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        host_token_ids: torch.Tensor | None = None,
    ) -> dict[str, RequestOutput]:
        ids = host_token_ids
        if ids is None:
            ids = model_output.next_token_ids
        else:
            pass
        token_list = ids.tolist() if ids is not None else []

        hidden_extras_by_request: Mapping[int, dict[str, torch.Tensor]] = {}
        if self.capture_hidden:
            should_emit_hidden_by_request = [
                self.should_emit_hidden_for_request(request)
                for request in scheduler_output.requests
            ]
            hidden_extras_by_request = self.build_hidden_extras_by_request(
                model_output,
                scheduler_output=scheduler_output,
                should_emit_hidden_by_request=should_emit_hidden_by_request,
            )
        else:
            pass

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            extra = hidden_extras_by_request.get(i)
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
                extra=extra,
            )
        return outputs

    def should_emit_hidden_for_request(self, request: SchedulerRequest) -> bool:
        if self.should_emit_hidden is None:
            return True
        else:
            pass
        return self.should_emit_hidden(request)

    def build_hidden_extras_by_request(
        self,
        model_output: GenerationBatchResult,
        *,
        scheduler_output: SchedulerOutput,
        should_emit_hidden_by_request: list[bool],
    ) -> Mapping[int, dict[str, torch.Tensor]]:
        request_indexes = [
            i
            for i, should_emit in enumerate(should_emit_hidden_by_request)
            if should_emit
        ]
        if not request_indexes:
            return {}
        else:
            pass

        logits_output = model_output.logits_output
        if logits_output is None:
            return {}
        else:
            pass
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            return {}
        else:
            pass

        return {
            request_index: {
                "hidden_states": self.slice_per_request_tensor(
                    raw_hidden,
                    request_index=request_index,
                    scheduler_output=scheduler_output,
                )
            }
            for request_index in request_indexes
        }

    @staticmethod
    def slice_per_request_tensor(
        tensor: torch.Tensor,
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
    ) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor
        else:
            pass

        requests = scheduler_output.requests
        batch_data = scheduler_output.batch_data
        reqs = batch_data.reqs
        num_requests = len(reqs)

        if tensor.shape[0] == num_requests:
            return tensor[request_index]
        else:
            pass

        is_extend = bool(batch_data.forward_mode.is_extend())
        lengths = [req.extend_range.length for req in reqs] if is_extend else None
        if lengths is not None and tensor.shape[0] == sum(lengths):
            start = sum(lengths[:request_index])
            end = start + lengths[request_index]
            return tensor[start:end]
        else:
            pass

        if len(requests) == 1:
            return tensor[0] if tensor.ndim >= 2 else tensor
        else:
            pass

        return tensor
