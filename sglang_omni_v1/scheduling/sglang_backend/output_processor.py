# SPDX-License-Identifier: Apache-2.0
"""Converts SGLang GenerationBatchResult to per-request RequestOutputs."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni_v1.scheduling.types import RequestOutput, SchedulerOutput


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def __init__(
        self,
        capture_hidden: bool = False,
        capture_hidden_layers: list[int] | None = None,
        model: Any = None,
    ):
        self._capture_hidden = capture_hidden
        self._capture_hidden_layers = capture_hidden_layers
        self._model = model

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

        hidden_states_dict = None
        stream_hidden_states = None
        if self._capture_hidden:
            hidden_states_dict = self._extract_hidden_states(model_output)
            stream_hidden_states = self._extract_stream_hidden_states(model_output)

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            extra = None
            if hidden_states_dict is not None:
                if "_single" in hidden_states_dict:
                    extra = {
                        "hidden_states": self._slice_per_request_tensor(
                            hidden_states_dict["_single"],
                            request_index=i,
                            scheduler_output=scheduler_output,
                        )
                    }
                else:
                    per_req = {}
                    for key, tensor in hidden_states_dict.items():
                        per_req[key] = self._slice_per_request_tensor(
                            tensor,
                            request_index=i,
                            scheduler_output=scheduler_output,
                        )
                    extra = {"hidden_states": per_req}
                    if stream_hidden_states is not None:
                        extra["stream_hidden_states"] = self._slice_per_request_tensor(
                            stream_hidden_states,
                            request_index=i,
                            scheduler_output=scheduler_output,
                        )
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
                extra=extra,
            )
        return outputs

    def _extract_hidden_states(
        self,
        model_output: Any,
    ) -> dict[str, torch.Tensor] | None:
        if self._model is not None and self._capture_hidden_layers:
            aux = self._model._captured_aux_hidden_states
            if aux is not None:
                self._model._captured_aux_hidden_states = None
                result = {}
                for layer_id, tensor in zip(self._capture_hidden_layers, aux):
                    key = "embed" if layer_id == 0 else layer_id
                    result[key] = tensor.clone()
                return result

        logits_output = model_output.logits_output
        if logits_output is None:
            return None
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            return None

        if isinstance(raw_hidden, dict):
            return raw_hidden
        elif isinstance(raw_hidden, torch.Tensor):
            return {"_single": raw_hidden}
        return None

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
