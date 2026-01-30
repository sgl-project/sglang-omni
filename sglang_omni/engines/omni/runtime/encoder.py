# SPDX-License-Identifier: Apache-2.0
"""Encoder model support - BatchPlanner, InputPreparer, OutputProcessor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..types import RequestOutput, SchedulerOutput, SchedulerRequest
from .interfaces import ResourceManager

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class EncoderRequestData:
    """Encoder-specific request data (stored in SchedulerRequest.data)."""

    input_ids: torch.Tensor | None = None
    input_dict: dict[str, Any] | None = None
    embeddings: torch.Tensor | None = None  # Filled after execution (text encoders)
    output_dict: dict[str, Any] | None = None  # Filled after execution (multimodal)
    cache_key: str | None = None


@dataclass
class EncoderBatchData:
    """Encoder-specific batch data (SchedulerOutput.batch_data)."""

    input_ids_list: list[torch.Tensor] | None = None
    seq_lens: list[int] | None = None
    input_dicts: list[dict[str, Any]] | None = None
    active_indices: list[int] | None = None
    skip_results: list[dict[str, Any] | None] | None = None


# -----------------------------------------------------------------------------
# BatchPlanner
# -----------------------------------------------------------------------------


class EncoderBatchPlanner:
    """Batch planner for encoder models."""

    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        del running
        selected: list[SchedulerRequest] = []
        for request in waiting:
            if len(selected) >= self.max_batch_size:
                break
            if not resource_manager.can_allocate(request):
                break
            resource_manager.allocate(request)
            selected.append(request)
        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> EncoderBatchData:
        if any(getattr(r.data, "input_dict", None) is not None for r in requests):
            input_dicts: list[dict[str, Any]] = []
            active_indices: list[int] = []
            skip_results: list[dict[str, Any] | None] = []
            for idx, request in enumerate(requests):
                data = request.data
                input_dict = getattr(data, "input_dict", None)
                if input_dict is None:
                    input_dict = {}
                if not isinstance(input_dict, dict):
                    input_dict = {}
                skip_result = (
                    input_dict.get("_result") if input_dict.get("_skip") else None
                )
                skip_results.append(
                    skip_result if isinstance(skip_result, dict) else None
                )
                if input_dict.get("_skip"):
                    input_dicts.append({})
                    continue
                active_indices.append(idx)
                input_dicts.append(input_dict)
            return EncoderBatchData(
                input_dicts=input_dicts,
                active_indices=active_indices,
                skip_results=skip_results,
            )

        return EncoderBatchData(
            input_ids_list=[r.data.input_ids for r in requests],
            seq_lens=[len(r.data.input_ids) for r in requests],
        )

    def update_request(
        self,
        request: SchedulerRequest,
        output: RequestOutput,
    ) -> None:
        if isinstance(output.data, dict) and hasattr(request.data, "output_dict"):
            request.data.output_dict = output.data
        elif hasattr(request.data, "embeddings"):
            request.data.embeddings = output.data

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        return True  # Encoder always done in one pass


# -----------------------------------------------------------------------------
# InputPreparer
# -----------------------------------------------------------------------------


class EncoderInputPreparer:
    """Converts EncoderBatchData to model inputs."""

    EXCLUDED_KEYS = {"cache_key", "_skip", "_result"}

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        device: torch.device,
    ) -> dict[str, Any]:
        batch_data: EncoderBatchData = scheduler_output.batch_data
        if batch_data.input_dicts is not None:
            active_indices = batch_data.active_indices or []
            if not active_indices:
                return {"_skip_all": True}
            active_inputs = [batch_data.input_dicts[i] for i in active_indices]
            first = active_inputs[0]
            batched: dict[str, Any] = {}
            cat_keys = {
                "pixel_values",
                "image_grid_thw",
                "input_features",
                "feature_attention_mask",
                "audio_feature_lengths",
            }
            for key, value in first.items():
                # Skip metadata keys that shouldn't be passed to the model
                if key in self.EXCLUDED_KEYS:
                    continue
                if isinstance(value, torch.Tensor):
                    tensors = [inp[key] for inp in active_inputs]
                    if value.dim() == 0:
                        batched[key] = torch.stack(tensors).to(device)
                    elif key in cat_keys:
                        batched[key] = torch.cat(tensors, dim=0).to(device)
                    else:
                        try:
                            batched[key] = torch.stack(tensors).to(device)
                        except Exception:
                            batched[key] = torch.cat(tensors, dim=0).to(device)
                else:
                    batched[key] = [inp.get(key) for inp in active_inputs]
            return batched

        max_len = max(batch_data.seq_lens or [0])
        batch_size = len(batch_data.input_ids_list or [])

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        for i, ids in enumerate(batch_data.input_ids_list):
            seq_len = len(ids)
            input_ids[i, :seq_len] = ids.to(device)
            attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class EncoderOutputProcessor:
    """Extracts embeddings from encoder output."""

    def __init__(self, pooling: str = "last"):
        """Initialize output processor.

        Args:
            pooling: Pooling strategy - "last", "mean", or "cls"
        """
        self.pooling = pooling

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        batch_data: EncoderBatchData = scheduler_output.batch_data
        if batch_data.input_dicts is not None:
            outputs: dict[str, RequestOutput] = {}
            active_indices = batch_data.active_indices or []
            skip_results = batch_data.skip_results or []
            input_dicts = batch_data.input_dicts or []

            if not active_indices:
                for i, request in enumerate(scheduler_output.requests):
                    out = skip_results[i] or {}
                    outputs[request.request_id] = RequestOutput(
                        request_id=request.request_id,
                        data=out,
                        finished=True,
                        finish_reason="stop",
                    )
                return outputs

            if not isinstance(model_output, dict):
                for i, request in enumerate(scheduler_output.requests):
                    out = skip_results[i] or {"result": model_output}
                    outputs[request.request_id] = RequestOutput(
                        request_id=request.request_id,
                        data=out,
                        finished=True,
                        finish_reason="stop",
                    )
                return outputs

            image_counts = model_output.get("image_token_counts")
            audio_counts = model_output.get("audio_output_lengths")

            image_sizes: list[int] = []
            audio_sizes: list[int] = []
            for req_idx in active_indices:
                inp = input_dicts[req_idx] if req_idx < len(input_dicts) else {}
                grid = inp.get("image_grid_thw")
                if isinstance(grid, torch.Tensor):
                    if grid.dim() >= 2:
                        image_sizes.append(int(grid.shape[0]))
                    elif grid.numel() > 0:
                        image_sizes.append(1)
                    else:
                        image_sizes.append(0)
                else:
                    image_sizes.append(0)
                lengths = inp.get("audio_feature_lengths")
                if isinstance(lengths, torch.Tensor):
                    audio_sizes.append(int(lengths.numel()))
                else:
                    audio_sizes.append(0)

            def _split_by_sizes(
                value: torch.Tensor, sizes: list[int]
            ) -> list[torch.Tensor]:
                splits: list[torch.Tensor] = []
                offset = 0
                for size in sizes:
                    end = offset + int(size)
                    splits.append(value[offset:end])
                    offset = end
                return splits

            def _split_by_counts(
                value: torch.Tensor, counts: torch.Tensor
            ) -> list[torch.Tensor]:
                splits: list[torch.Tensor] = []
                offset = 0
                for count in counts.tolist():
                    end = offset + int(count)
                    splits.append(value[offset:end])
                    offset = end
                return splits

            embed_splits: dict[str, list[torch.Tensor]] = {}
            if (
                isinstance(image_counts, torch.Tensor)
                and "image_embeds" in model_output
            ):
                embed_splits["image_embeds"] = _split_by_counts(
                    model_output["image_embeds"], image_counts
                )
            if (
                isinstance(audio_counts, torch.Tensor)
                and "audio_embeds" in model_output
            ):
                embed_splits["audio_embeds"] = _split_by_counts(
                    model_output["audio_embeds"], audio_counts
                )

            for out_idx, req_idx in enumerate(active_indices):
                request = scheduler_output.requests[req_idx]
                out: dict[str, Any] = {}
                for key, value in model_output.items():
                    if key == "image_grid_thw" and isinstance(value, torch.Tensor):
                        if image_sizes and sum(image_sizes) == value.shape[0]:
                            out[key] = _split_by_sizes(value, image_sizes)[out_idx]
                        else:
                            out[key] = (
                                value
                                if len(active_indices) == 1
                                else value[out_idx : out_idx + 1]
                            )
                        continue
                    if key == "audio_feature_lengths" and isinstance(
                        value, torch.Tensor
                    ):
                        if audio_sizes and sum(audio_sizes) == value.numel():
                            out[key] = _split_by_sizes(value, audio_sizes)[out_idx]
                        else:
                            out[key] = (
                                value
                                if len(active_indices) == 1
                                else value[out_idx : out_idx + 1]
                            )
                        continue
                    if key in {
                        "image_token_counts",
                        "audio_output_lengths",
                    } and isinstance(value, torch.Tensor):
                        if value.dim() == 1 and value.shape[0] == len(active_indices):
                            out[key] = value[out_idx : out_idx + 1]
                        else:
                            out[key] = value
                        continue
                    if key in embed_splits:
                        out[key] = embed_splits[key][out_idx]
                        continue
                    if isinstance(value, torch.Tensor) and value.shape[0] == len(
                        active_indices
                    ):
                        out[key] = value[out_idx]
                    else:
                        out[key] = value
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=out,
                    finished=True,
                    finish_reason="stop",
                )

            for idx, request in enumerate(scheduler_output.requests):
                if idx in active_indices:
                    continue
                out = skip_results[idx] or {}
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=out,
                    finished=True,
                    finish_reason="stop",
                )
            return outputs

        # Handle different output formats
        hidden_states = getattr(model_output, "last_hidden_state", None)
        if hidden_states is None:
            if not isinstance(model_output, torch.Tensor):
                raise ValueError(f"Unexpected model output type: {type(model_output)}")
            hidden_states = model_output

        outputs = {}
        for i, request in enumerate(scheduler_output.requests):
            seq_len = batch_data.seq_lens[i]

            if self.pooling == "last":
                emb = hidden_states[i, seq_len - 1]
            elif self.pooling == "mean":
                emb = hidden_states[i, :seq_len].mean(dim=0)
            elif self.pooling == "cls":
                emb = hidden_states[i, 0]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=emb,
                finished=True,
                finish_reason="stop",
            )

        return outputs
