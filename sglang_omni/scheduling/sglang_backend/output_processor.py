# SPDX-License-Identifier: Apache-2.0
"""Converts SGLang GenerationBatchResult to per-request RequestOutputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from sglang_omni.scheduling.types import RequestOutput, SchedulerOutput


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def __init__(
        self,
        capture_hidden: bool = False,
        capture_hidden_layers: list[int] | None = None,
        model: Any = None,
        should_emit_hidden: Callable[[Any], bool] | None = None,
    ):
        self._capture_hidden = capture_hidden
        self._capture_hidden_layers = capture_hidden_layers
        self._model = model
        self._should_emit_hidden = should_emit_hidden

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        token_list = (
            model_output.next_token_ids.tolist()
            if model_output.next_token_ids is not None
            else []
        )

        hidden_extras_by_request: dict[int, dict[str, Any] | None] = {}
        if self._capture_hidden:
            should_emit_hidden_by_request = [
                self._should_emit_hidden_for_request(request)
                for request in scheduler_output.requests
            ]
            hidden_extras_by_request = self._build_hidden_extras_by_request(
                model_output,
                scheduler_output=scheduler_output,
                should_emit_hidden_by_request=should_emit_hidden_by_request,
            )

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

    def _should_emit_hidden_for_request(self, request: Any) -> bool:
        if self._should_emit_hidden is None:
            return True
        return self._should_emit_hidden(request)

    def _build_hidden_extras_by_request(
        self,
        model_output: Any,
        *,
        scheduler_output: SchedulerOutput,
        should_emit_hidden_by_request: list[bool],
    ) -> dict[int, dict[str, Any] | None]:
        request_indexes = [
            i
            for i, should_emit in enumerate(should_emit_hidden_by_request)
            if should_emit
        ]

        if self._model is not None and self._capture_hidden_layers:
            captured_aux_hidden_states = self._model._captured_aux_hidden_states
            if captured_aux_hidden_states is not None:
                self._model._captured_aux_hidden_states = None
                if not request_indexes:
                    return {}
                stream_hidden_states = self._extract_stream_hidden_states(model_output)
                return {
                    request_index: self._build_aux_hidden_extra(
                        captured_aux_hidden_states,
                        request_index=request_index,
                        scheduler_output=scheduler_output,
                        stream_hidden_states=stream_hidden_states,
                    )
                    for request_index in request_indexes
                }

        if not request_indexes:
            return {}

        logits_output = model_output.logits_output
        if logits_output is None:
            return {}
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            return {}

        if isinstance(raw_hidden, dict):
            return {
                request_index: self._build_dict_hidden_extra(
                    raw_hidden,
                    request_index=request_index,
                    scheduler_output=scheduler_output,
                )
                for request_index in request_indexes
            }
        elif isinstance(raw_hidden, torch.Tensor):
            return {
                request_index: {
                    "hidden_states": self._slice_per_request_tensor(
                        raw_hidden,
                        request_index=request_index,
                        scheduler_output=scheduler_output,
                    )
                }
                for request_index in request_indexes
            }
        return {}

    def _build_aux_hidden_extra(
        self,
        aux_hidden_states: Sequence[torch.Tensor],
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
        stream_hidden_states: torch.Tensor | None,
    ) -> dict[str, Any]:
        per_request_hidden = {}
        for layer_id, tensor in zip(
            self._capture_hidden_layers or [],
            aux_hidden_states,
        ):
            key = "embed" if layer_id == 0 else layer_id
            per_request_hidden[key] = self._slice_per_request_tensor(
                tensor,
                request_index=request_index,
                scheduler_output=scheduler_output,
            ).clone()

        extra: dict[str, Any] = {"hidden_states": per_request_hidden}
        if stream_hidden_states is not None:
            extra["stream_hidden_states"] = self._slice_per_request_tensor(
                stream_hidden_states,
                request_index=request_index,
                scheduler_output=scheduler_output,
            ).clone()
        return extra

    def _build_dict_hidden_extra(
        self,
        hidden_states: dict[Any, torch.Tensor],
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, Any]:
        return {
            "hidden_states": {
                key: self._slice_per_request_tensor(
                    tensor,
                    request_index=request_index,
                    scheduler_output=scheduler_output,
                )
                for key, tensor in hidden_states.items()
            }
        }

    def _extract_stream_hidden_states(self, model_output: Any) -> torch.Tensor | None:
        logits_output = model_output.logits_output
        if logits_output is None:
            return None
        raw_hidden = logits_output.hidden_states
        return raw_hidden if isinstance(raw_hidden, torch.Tensor) else None

    @staticmethod
    def _slice_per_request_tensor(
        tensor: torch.Tensor,
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
    ) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor

        requests = scheduler_output.requests
        if len(requests) == 1:
            return tensor[0] if tensor.ndim >= 2 else tensor

        batch_data = scheduler_output.batch_data
        reqs = batch_data.reqs
        num_requests = len(reqs)
        if tensor.shape[0] == num_requests:
            return tensor[request_index]

        lengths = [req.extend_input_len for req in reqs]
        total_tokens = sum(lengths)
        if tensor.shape[0] == total_tokens:
            start = sum(lengths[:request_index])
            end = start + lengths[request_index]
            return tensor[start:end]

        return tensor
