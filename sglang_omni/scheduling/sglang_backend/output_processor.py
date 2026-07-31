# SPDX-License-Identifier: Apache-2.0
"""Converts SGLang GenerationBatchResult to per-request RequestOutputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from sglang_omni.model_runner._hidden_capture import unpack_packed_hidden_capture
from sglang_omni.scheduling.types import RequestOutput, SchedulerOutput


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def __init__(
        self,
        capture_hidden: bool = False,
        capture_hidden_layers: list[int] | None = None,
        model: Any = None,
        should_emit_hidden: Callable[[Any], bool] | None = None,
        capture_hidden_width: int | None = None,
    ):
        if capture_hidden and not capture_hidden_layers:
            raise ValueError("capture_hidden requires capture_hidden_layers")
        if capture_hidden_layers and capture_hidden_width is None:
            raise ValueError(
                "capture_hidden_layers requires capture_hidden_width to unpack "
                "the packed hidden capture"
            )
        self._capture_hidden = capture_hidden
        self._capture_hidden_layers = capture_hidden_layers
        self._model = model
        self._capture_hidden_width = capture_hidden_width
        self._should_emit_hidden = should_emit_hidden

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
        host_token_ids: torch.Tensor | None = None,
    ) -> dict[str, RequestOutput]:
        ids = host_token_ids
        if ids is None:
            ids = model_output.next_token_ids
        token_list = ids.tolist() if ids is not None else []

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

        if not request_indexes:
            return {}
        captured_aux_hidden_states = self._take_captured_aux_hidden_states(model_output)
        if captured_aux_hidden_states is None:
            return {}
        return {
            request_index: self._build_aux_hidden_extra(
                captured_aux_hidden_states,
                request_index=request_index,
                scheduler_output=scheduler_output,
            )
            for request_index in request_indexes
        }

    def _take_captured_aux_hidden_states(
        self, model_output: Any
    ) -> Sequence[torch.Tensor] | None:
        captured = model_output._captured_aux_hidden_states
        if captured is not None:
            model_output._captured_aux_hidden_states = None
            return captured
        logits_output = model_output.logits_output
        if logits_output is None:
            return None
        return unpack_packed_hidden_capture(
            logits_output.hidden_states,
            capture_layer_count=len(self._capture_hidden_layers),
            hidden_size=self._capture_hidden_width,
        )

    def _build_aux_hidden_extra(
        self,
        aux_hidden_states: Sequence[torch.Tensor],
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
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

        return {"hidden_states": per_request_hidden}

    @staticmethod
    def _slice_per_request_tensor(
        tensor: torch.Tensor,
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
    ) -> torch.Tensor:
        # ``requests`` is frozen at launch, whereas the live ScheduleBatch may
        # be filtered before a lookahead step resolves. Capture tensors carry
        # one row per launch-time request (decode FULL slices the replayed
        # graph output to the batch, prefill LAST keeps one row per request),
        # so launch position is the only legal mapping — anything else is a
        # wrong-request hidden state, not a shape to accommodate.
        requests = scheduler_output.requests
        if tensor.shape[0] != len(requests):
            raise ValueError(
                f"per-request hidden tensor has {tensor.shape[0]} rows for "
                f"{len(requests)} launch-time requests"
            )
        return tensor[request_index]
