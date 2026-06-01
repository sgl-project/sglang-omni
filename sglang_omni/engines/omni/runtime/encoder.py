# SPDX-License-Identifier: Apache-2.0
"""Encoder model support - BatchPlanner, InputPreparer, OutputProcessor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ..types import RequestOutput, SchedulerOutput, SchedulerRequest
from .interfaces import ResourceManager

_AUDIO_FEATURES_NDIM = 3
_AUDIO_MASK_NDIM = 2


def _right_pad_last_dim(
    tensor: torch.Tensor, target_len: int, pad_value: float | int = 0.0
) -> torch.Tensor:
    """Right-pad only the last dimension of tensor to target_len.

    Note (Chenyang):
    Returns the tensor unchanged if its last dimension already meets or
    exceeds target_len. pad_value is cast by F.pad to the tensor's dtype,
    so both float and int are accepted for use with feature/mask tensors.
    """
    current_len = tensor.shape[-1]
    if current_len >= target_len:
        return tensor
    return F.pad(tensor, (0, target_len - current_len), value=pad_value)


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


class EncoderBatchPlanner:
    """Batch planner for encoder models."""

    def __init__(
        self,
        max_batch_size: int = 32,
        *,
        request_cost_fn: Callable[[SchedulerRequest], int] | None = None,
        max_batch_cost: int | None = None,
    ):
        self.max_batch_size = max_batch_size
        self.request_cost_fn = request_cost_fn
        self.max_batch_cost = max_batch_cost

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        del running
        selected: list[SchedulerRequest] = []
        selected_cost = 0
        for request in waiting:
            if len(selected) >= self.max_batch_size:
                break
            request_cost = self._request_cost(request)
            if (
                self.max_batch_cost is not None
                and selected
                and selected_cost + request_cost > self.max_batch_cost
            ):
                break
            if not resource_manager.can_allocate(request):
                break
            resource_manager.allocate(request)
            selected.append(request)
            selected_cost += request_cost
        return selected

    def _request_cost(self, request: SchedulerRequest) -> int:
        if self.request_cost_fn is None:
            return 0
        return max(0, int(self.request_cost_fn(request)))

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


class EncoderInputPreparer:
    """Converts EncoderBatchData to model inputs."""

    EXCLUDED_KEYS = {"cache_key", "_skip", "_result"}
    # Note (Ratish): Qwen3-Omni pads within each request but can still produce
    # different time lengths across requests. Keys listed here are right-padded
    # on their last (time) dimension before being concatenated on the batch
    # dimension. Each entry maps to (expected_ndim, pad_value_for_last_dim) so
    # the dispatch key set and per-key invariants stay in a single source of
    # truth.
    TIME_PAD_SPECS: dict[str, tuple[int, float]] = {
        "input_features": (_AUDIO_FEATURES_NDIM, 0.0),
        "feature_attention_mask": (_AUDIO_MASK_NDIM, 0.0),
    }

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def _pad_and_cat_tensors(
        self,
        *,
        key: str,
        tensors: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        assert (
            tensors
        ), f"_pad_and_cat_tensors called with empty tensor list for key={key}"
        expected_dim, pad_value = self.TIME_PAD_SPECS[key]
        if any(tensor.dim() != expected_dim for tensor in tensors):
            raise ValueError(
                f"Unsupported tensor layout for time padding: key={key}, "
                f"tensor_shapes={[tuple(tensor.shape) for tensor in tensors]}"
            )

        # Note (Chenyang, Ratish):
        # Only the last (time) dim is padded; all non-time dims must match
        # across requests. Catching the mismatch here surfaces a clear error
        # instead of a silent shape bug downstream.
        ref_shape = tensors[0].shape[:-1]
        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.shape[:-1] != ref_shape:
                raise ValueError(
                    f"Non-time dimensions mismatch for key={key}: "
                    f"tensor[0].shape={tuple(tensors[0].shape)}, "
                    f"tensor[{i}].shape={tuple(tensor.shape)}"
                )

        max_time_dim = max(tensor.shape[-1] for tensor in tensors)
        padded = [
            _right_pad_last_dim(tensor, max_time_dim, pad_value=pad_value)
            for tensor in tensors
        ]
        return torch.cat(padded, dim=0).to(device)

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

            # Note (Ratish, Chenyang):
            # Keys routed through plain torch.cat on dim=0. Entries in
            # TIME_PAD_SPECS are intercepted earlier and handled via the
            # right-pad-and-cat path, so they do not appear here.

            cat_keys = {
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "audio_feature_lengths",
                "audio_feats",
                "audio_feats_lengths",
                "audio_placeholder_loc_lens",
            }
            for key, value in first.items():
                # Skip metadata keys that shouldn't be passed to the model
                if key in self.EXCLUDED_KEYS:
                    continue
                if isinstance(value, torch.Tensor):
                    tensors = [inp[key] for inp in active_inputs]
                    if value.dim() == 0:
                        batched[key] = torch.stack(tensors).to(device)
                    elif key in self.TIME_PAD_SPECS:
                        batched[key] = self._pad_and_cat_tensors(
                            key=key,
                            tensors=tensors,
                            device=device,
                        )
                    elif key in cat_keys:
                        batched[key] = torch.cat(tensors, dim=0).to(device)
                    else:
                        batched[key] = torch.stack(tensors).to(device)
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


@dataclass
class ModalityConfig:
    """Configuration for a single modality (image/audio/video)."""

    input_grid_key: str
    input_size_key: str
    embed_key: str
    count_key: str
    size_extractor: Callable[[dict[str, Any]], int]


class ModalitySplitter:
    """Utility for splitting tensors by sizes or counts."""

    @staticmethod
    def split_by_sizes(value: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
        """Split tensor by predetermined sizes."""
        splits: list[torch.Tensor] = []
        offset = 0
        for size in sizes:
            end = offset + int(size)
            splits.append(value[offset:end])
            offset = end
        return splits

    @staticmethod
    def split_by_counts(
        value: torch.Tensor, counts: torch.Tensor
    ) -> list[torch.Tensor]:
        """Split tensor by counts tensor."""
        splits: list[torch.Tensor] = []
        offset = 0
        for count in counts.tolist():
            end = offset + int(count)
            splits.append(value[offset:end])
            offset = end
        return splits


class EncoderOutputProcessor:
    """Extracts embeddings from encoder output."""

    def __init__(self, pooling: str = "last"):
        """Initialize output processor.

        Args:
            pooling: Pooling strategy - "last", "mean", or "cls"
        """
        self.pooling = pooling
        self.splitter = ModalitySplitter()
        self.modality_configs = self._init_modality_configs()

    def _init_modality_configs(self) -> dict[str, ModalityConfig]:
        """Initialize modality configurations."""

        def _extract_image_size(inp: dict[str, Any]) -> int:
            grid = inp.get("image_grid_thw")
            if isinstance(grid, torch.Tensor):
                if grid.dim() >= 2:
                    return int(grid.shape[0])
                elif grid.numel() > 0:
                    return 1
            return 0

        def _extract_audio_size(inp: dict[str, Any]) -> int:
            lengths = inp.get("audio_feature_lengths")
            if isinstance(lengths, torch.Tensor):
                return int(lengths.numel())
            return 0

        def _extract_video_size(inp: dict[str, Any]) -> int:
            video_grid = inp.get("video_grid_thw")
            if isinstance(video_grid, torch.Tensor):
                if video_grid.dim() >= 2:
                    return int(video_grid.shape[0])
                elif video_grid.numel() > 0:
                    return 1
            return 0

        return {
            "image": ModalityConfig(
                input_grid_key="image_grid_thw",
                input_size_key="image_grid_thw",
                embed_key="image_embeds",
                count_key="image_token_counts",
                size_extractor=_extract_image_size,
            ),
            "audio": ModalityConfig(
                input_grid_key="audio_feature_lengths",
                input_size_key="audio_feature_lengths",
                embed_key="audio_embeds",
                count_key="audio_output_lengths",
                size_extractor=_extract_audio_size,
            ),
            "video": ModalityConfig(
                input_grid_key="video_grid_thw",
                input_size_key="video_grid_thw",
                embed_key="video_embeds",
                count_key="video_token_counts",
                size_extractor=_extract_video_size,
            ),
        }

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        """Process model output and return per-request outputs.

        Routes to either multimodal or text embedding processing based on batch_data type.
        """
        batch_data: EncoderBatchData = scheduler_output.batch_data

        if batch_data.input_dicts is not None:
            return self._process_multimodal(model_output, scheduler_output)
        else:
            return self._process_text_embedding(model_output, scheduler_output)

    def _process_multimodal(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        """Process multimodal encoder output (image/audio/video)."""
        batch_data: EncoderBatchData = scheduler_output.batch_data
        active_indices = batch_data.active_indices or []
        skip_results = batch_data.skip_results or []
        input_dicts = batch_data.input_dicts or []

        # Handle all-skip case: directly return skip results
        if not active_indices:
            outputs: dict[str, RequestOutput] = {}
            for i, request in enumerate(scheduler_output.requests):
                out = skip_results[i] or {}
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=out,
                    finished=True,
                    finish_reason="stop",
                )
            return outputs

        # Handle non-dict output (fallback case)
        if not isinstance(model_output, dict):
            outputs: dict[str, RequestOutput] = {}
            for i, request in enumerate(scheduler_output.requests):
                out = skip_results[i] or {"result": model_output}
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=out,
                    finished=True,
                    finish_reason="stop",
                )
            return outputs

        # Extract modality sizes and split embeddings
        modality_sizes = self._extract_modality_sizes(input_dicts, active_indices)
        embed_splits = self._split_embeddings(model_output)

        # Build outputs for active requests
        outputs = self._build_active_outputs(
            model_output,
            scheduler_output,
            active_indices,
            modality_sizes,
            embed_splits,
        )

        # Add skip outputs for inactive requests
        for idx, request in enumerate(scheduler_output.requests):
            if idx not in active_indices:
                out = skip_results[idx] or {}
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=out,
                    finished=True,
                    finish_reason="stop",
                )

        return outputs

    def _extract_modality_sizes(
        self, input_dicts: list[dict[str, Any]], active_indices: list[int]
    ) -> dict[str, list[int]]:
        """Extract sizes for each modality from input dicts."""
        modality_sizes: dict[str, list[int]] = {
            name: [] for name in self.modality_configs
        }

        for req_idx in active_indices:
            inp = input_dicts[req_idx] if req_idx < len(input_dicts) else {}
            for name, config in self.modality_configs.items():
                size = config.size_extractor(inp)
                modality_sizes[name].append(size)

        return modality_sizes

    def _split_embeddings(
        self, model_output: dict[str, Any]
    ) -> dict[str, list[torch.Tensor]]:
        """Split embeddings by counts for each modality."""
        embed_splits: dict[str, list[torch.Tensor]] = {}

        for config in self.modality_configs.values():
            counts = model_output.get(config.count_key)
            embeds = model_output.get(config.embed_key)

            if isinstance(counts, torch.Tensor) and embeds is not None:
                embed_splits[config.embed_key] = self.splitter.split_by_counts(
                    embeds, counts
                )

        return embed_splits

    def _build_active_outputs(
        self,
        model_output: dict[str, Any],
        scheduler_output: SchedulerOutput,
        active_indices: list[int],
        modality_sizes: dict[str, list[int]],
        embed_splits: dict[str, list[torch.Tensor]],
    ) -> dict[str, RequestOutput]:
        outputs: dict[str, RequestOutput] = {}

        for out_idx, req_idx in enumerate(active_indices):
            request = scheduler_output.requests[req_idx]
            out: dict[str, Any] = {}

            for key, value in model_output.items():
                out[key] = self._extract_value_for_request(
                    key,
                    value,
                    out_idx,
                    active_indices,
                    modality_sizes,
                    embed_splits,
                )

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=out,
                finished=True,
                finish_reason="stop",
            )

        return outputs

    def _slice_for_request(
        self,
        items: list | torch.Tensor,
        sizes: list[int],
        out_idx: int,
    ) -> list | torch.Tensor:
        """Slice per-item ``items`` into the chunk belonging to request ``out_idx``."""
        offset = sum(sizes[:out_idx])
        count = sizes[out_idx]
        return items[offset : offset + count]

    def _extract_value_for_request(
        self,
        key: str,
        value: Any,
        out_idx: int,
        active_indices: list[int],
        modality_sizes: dict[str, list[int]],
        embed_splits: dict[str, list[torch.Tensor]],
    ) -> Any:
        """Extract the appropriate value for a specific request."""
        if not isinstance(value, torch.Tensor):
            return value

        # Handle special grid keys (image_grid_thw, video_grid_thw)
        if key in {"image_grid_thw", "video_grid_thw"}:
            modality = "image" if key == "image_grid_thw" else "video"
            sizes = modality_sizes[modality]
            if sizes and sum(sizes) == value.shape[0]:
                return self.splitter.split_by_sizes(value, sizes)[out_idx]
            return value if len(active_indices) == 1 else value[out_idx : out_idx + 1]

        # Handle audio_feature_lengths
        if key == "audio_feature_lengths":
            sizes = modality_sizes["audio"]
            if sizes and sum(sizes) == value.numel():
                return self.splitter.split_by_sizes(value, sizes)[out_idx]
            return value if len(active_indices) == 1 else value[out_idx : out_idx + 1]

        # Handle count keys
        if key in {"image_token_counts", "audio_output_lengths", "video_token_counts"}:
            for mod_name, config in self.modality_configs.items():
                if config.count_key == key:
                    sizes = modality_sizes[mod_name]
                    return self._slice_for_request(value, sizes, out_idx)
            return value

        # Note(Yifei):
        # Handle embeddings. ``embed_splits`` is built solely by
        # ``_split_embeddings`` from ``self.modality_configs``, so any key here
        # is guaranteed to match exactly one config.embed_key below.
        if key in embed_splits:
            splits = embed_splits[key]
            for mod_name, config in self.modality_configs.items():
                if config.embed_key == key:
                    sizes = modality_sizes[mod_name]
                    # Note(Yifei):
                    # Precondition: image/audio live in separate encoder stages
                    # and InputPreparer enforces homogeneous key sets across a
                    # batch, so whenever ``key`` is in ``embed_splits`` every
                    # active request must contribute at least one item.
                    assert sizes[out_idx] > 0, (
                        f"empty {mod_name} slice for active request {out_idx} "
                        f"despite {key} present in embed_splits"
                    )
                    chunk = self._slice_for_request(splits, sizes, out_idx)
                    return chunk[0] if len(chunk) == 1 else torch.cat(chunk)

        # Handle generic batched tensors
        if value.shape[0] == len(active_indices):
            return value[out_idx]

        return value

    def _process_text_embedding(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        """Process text encoder output (embedding extraction with pooling)."""
        batch_data: EncoderBatchData = scheduler_output.batch_data

        # Extract hidden states
        hidden_states = getattr(model_output, "last_hidden_state", None)
        if hidden_states is None:
            if not isinstance(model_output, torch.Tensor):
                raise ValueError(f"Unexpected model output type: {type(model_output)}")
            hidden_states = model_output

        # Apply pooling and build outputs
        outputs = {}
        for i, request in enumerate(scheduler_output.requests):
            seq_len = batch_data.seq_lens[i]
            emb = self._apply_pooling(hidden_states[i], seq_len)

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=emb,
                finished=True,
                finish_reason="stop",
            )

        return outputs

    def _apply_pooling(self, hidden_state: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply pooling strategy to extract embedding."""
        if self.pooling == "last":
            return hidden_state[seq_len - 1]
        elif self.pooling == "mean":
            return hidden_state[:seq_len].mean(dim=0)
        elif self.pooling == "cls":
            return hidden_state[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
