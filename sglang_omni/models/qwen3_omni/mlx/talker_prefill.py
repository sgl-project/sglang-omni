# SPDX-License-Identifier: Apache-2.0
"""Native MLX construction of Qwen3-Omni talker prefill rows."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    _resolve_embed_source,
    resolve_speaker_id,
)
from sglang_omni.models.qwen3_omni.mlx.talker import Qwen3OmniMlxTalker
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.proto import StagePayload

__all__ = ["Qwen3OmniMlxTalkerPrefillBuilder", "build_mlx_prefill_input"]


def _torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    tensor = tensor.detach().cpu()
    if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.float()
    return mx.array(tensor.numpy())


def _mlx_to_torch(array: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(array.astype(mx.float32))))


@dataclass(frozen=True)
class _SafetensorTensorMetadata:
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int


class _SafetensorRowReader:
    def __init__(self) -> None:
        self._metadata: dict[tuple[Path, str], _SafetensorTensorMetadata] = {}
        self._binary_handles: dict[Path, Any] = {}
        self._numpy_handles: dict[Path, Any] = {}

    def read_rows(
        self,
        shard: Path,
        tensor_name: str,
        row_ids: list[int],
    ) -> mx.array:
        metadata = self._tensor_metadata(shard, tensor_name)
        self._validate_row_ids(metadata, row_ids)
        if metadata.dtype == "BF16":
            return self._read_bfloat16_rows(shard, tensor_name, metadata, row_ids)

        handle = self._numpy_handles.get(shard)
        if handle is None:
            handle = safe_open(str(shard), framework="numpy")
            handle.__enter__()
            self._numpy_handles[shard] = handle
        tensor_slice = handle.get_slice(tensor_name)
        rows = np.stack([np.asarray(tensor_slice[row_id]) for row_id in row_ids])
        return mx.array(rows)

    def _tensor_metadata(
        self,
        shard: Path,
        tensor_name: str,
    ) -> _SafetensorTensorMetadata:
        key = (shard, tensor_name)
        cached = self._metadata.get(key)
        if cached is not None:
            return cached

        handle = self._binary_handles.get(shard)
        if handle is None:
            handle = shard.open("rb")
            self._binary_handles[shard] = handle
        handle.seek(0)
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"truncated safetensors header in {shard}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"truncated safetensors header in {shard}")
        header = json.loads(raw_header)
        data_base = 8 + header_size
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            offsets = tensor["data_offsets"]
            self._metadata[(shard, name)] = _SafetensorTensorMetadata(
                dtype=str(tensor["dtype"]),
                shape=tuple(int(size) for size in tensor["shape"]),
                data_start=data_base + int(offsets[0]),
                data_end=data_base + int(offsets[1]),
            )
        try:
            return self._metadata[key]
        except KeyError as exc:
            raise KeyError(f"tensor {tensor_name!r} is missing from {shard}") from exc

    @staticmethod
    def _validate_row_ids(
        metadata: _SafetensorTensorMetadata,
        row_ids: list[int],
    ) -> None:
        if not metadata.shape:
            raise ValueError("cannot select rows from a scalar safetensor")
        for row_id in row_ids:
            if not 0 <= row_id < metadata.shape[0]:
                raise IndexError(row_id)

    def _read_bfloat16_rows(
        self,
        shard: Path,
        tensor_name: str,
        metadata: _SafetensorTensorMetadata,
        row_ids: list[int],
    ) -> mx.array:
        handle = self._binary_handles[shard]
        shape = metadata.shape
        row_elements = math.prod(shape[1:])
        row_bytes = row_elements * 2
        if metadata.data_end - metadata.data_start != math.prod(shape) * 2:
            raise ValueError(f"invalid BF16 tensor size for {tensor_name!r} in {shard}")
        rows = []
        for row_id in row_ids:
            handle.seek(metadata.data_start + row_id * row_bytes)
            raw = handle.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError(f"truncated BF16 tensor {tensor_name!r} in {shard}")
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            rows.append((words << 16).view("<f4").reshape(shape[1:]))
        return mx.array(np.stack(rows), dtype=mx.bfloat16)

    def close(self) -> None:
        for handle in self._numpy_handles.values():
            handle.__exit__(None, None, None)
        self._numpy_handles.clear()
        for handle in self._binary_handles.values():
            handle.close()
        self._binary_handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _load_mlx_embedding_rows(
    model_path: str,
    row_ids: list[int],
    *,
    row_reader: _SafetensorRowReader | None = None,
) -> mx.array:
    owns_reader = row_reader is None
    row_reader = row_reader or _SafetensorRowReader()
    try:
        source = _resolve_embed_source(model_path)
        if source.quantization is None:
            return row_reader.read_rows(source.shard, source.tensor_name, row_ids)

        if source.scales_shard is None or source.biases_shard is None:
            raise KeyError(
                f"packed thinker embedding {source.tensor_name!r} is incomplete"
            )
        prefix = source.tensor_name[: -len(".weight")]
        weight = row_reader.read_rows(source.shard, source.tensor_name, row_ids)
        scales = row_reader.read_rows(source.scales_shard, f"{prefix}.scales", row_ids)
        biases = row_reader.read_rows(source.biases_shard, f"{prefix}.biases", row_ids)
        quantization = source.quantization
        return mx.dequantize(
            weight,
            scales,
            biases,
            group_size=int(quantization["group_size"]),
            bits=int(quantization["bits"]),
            mode=str(quantization.get("mode", "affine")),
        )
    finally:
        if owns_reader:
            row_reader.close()


def _coerce_feature_array(value: Any) -> mx.array | None:
    if value is None:
        return None
    if isinstance(value, mx.array):
        array = value
    elif isinstance(value, torch.Tensor):
        array = _torch_to_mlx(value)
    elif isinstance(value, (list, tuple)):
        arrays = [_coerce_feature_array(item) for item in value]
        arrays = [array for array in arrays if array is not None]
        if not arrays:
            return None
        array = mx.concatenate(arrays, axis=0)
    else:
        return None

    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim > 2:
        return array.reshape(-1, array.shape[-1])
    return array


def _segment_chat_template(
    input_ids: list[int],
    *,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
) -> list[dict[str, Any]]:
    role_map = {
        system_token_id: "system",
        user_token_id: "user",
        assistant_token_id: "assistant",
    }
    starts = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id == im_start_token_id
    ]
    segments = []
    for index, start in enumerate(starts):
        role_token = input_ids[start + 1] if start + 1 < len(input_ids) else None
        segments.append(
            {
                "role": role_map.get(role_token, "unknown"),
                "start": start,
                "end": starts[index + 1] if index + 1 < len(starts) else len(input_ids),
            }
        )
    return segments


def _build_mlx_user_part(
    *,
    thinker_embed: mx.array,
    thinker_hidden: mx.array,
    multimodal_mask: mx.array,
    text_projection: Any,
    hidden_projection: Any,
) -> mx.array:
    return mx.where(
        multimodal_mask[:, None],
        hidden_projection(thinker_hidden),
        text_projection(thinker_embed),
    )


def _build_mlx_assistant_part(
    *,
    assistant_embed: mx.array,
    text_projection: Any,
    codec_embedding: Any,
    tts_bos_embed: mx.array,
    tts_eos_embed: mx.array,
    tts_pad_embed: mx.array,
    speaker_id: int,
    special_token_ids: Mapping[str, int],
) -> dict[str, mx.array]:
    projected = text_projection(assistant_embed)
    fourth_token = (
        projected[3:4]
        if projected.shape[0] > 3
        else mx.zeros((1, projected.shape[-1]), dtype=projected.dtype)
    )
    text_hidden = mx.concatenate(
        [
            projected[:3],
            mx.broadcast_to(tts_pad_embed, (4, tts_pad_embed.shape[-1])),
            tts_bos_embed,
            fourth_token,
        ],
        axis=0,
    )
    codec_ids = mx.array(
        [
            special_token_ids["codec_nothink_id"],
            special_token_ids["codec_think_bos_id"],
            special_token_ids["codec_think_eos_id"],
            speaker_id,
            special_token_ids["codec_pad_id"],
            special_token_ids["codec_bos_id"],
        ],
        dtype=mx.int32,
    )
    codec_hidden = mx.concatenate(
        [
            mx.zeros((3, text_hidden.shape[-1]), dtype=text_hidden.dtype),
            codec_embedding(codec_ids),
        ],
        axis=0,
    )
    future_text_rows = (
        mx.concatenate([projected[4:], tts_eos_embed], axis=0)
        if projected.shape[0] > 4
        else mx.array(tts_eos_embed)
    )
    return {
        "input_embeds": text_hidden + codec_hidden,
        "input_ids": mx.full(
            (text_hidden.shape[0],),
            special_token_ids["tts_pad_token_id"],
            dtype=mx.int32,
        ),
        "future_text_rows": future_text_rows,
    }


def build_mlx_prefill_input(
    *,
    thinker_embed: mx.array,
    thinker_hidden: mx.array,
    thinker_input_ids: list[int],
    multimodal_mask: mx.array,
    text_projection: Any,
    hidden_projection: Any,
    codec_embedding: Any,
    tts_bos_embed: mx.array,
    tts_eos_embed: mx.array,
    tts_pad_embed: mx.array,
    special_token_ids: Mapping[str, int],
    speaker_id: int,
    include_assistant_eos: bool = True,
) -> dict[str, mx.array | None]:
    segments = _segment_chat_template(
        thinker_input_ids,
        im_start_token_id=special_token_ids["im_start_token_id"],
        system_token_id=special_token_ids["system_token_id"],
        user_token_id=special_token_ids["user_token_id"],
        assistant_token_id=special_token_ids["assistant_token_id"],
    )
    assistant_indices = [
        index
        for index, segment in enumerate(segments)
        if segment["role"] == "assistant"
    ]
    last_assistant = assistant_indices[-1] if assistant_indices else None
    all_embeds: list[mx.array] = []
    all_ids: list[mx.array] = []
    future_text_rows = None

    for segment_index, segment in enumerate(segments):
        if segment["role"] == "system":
            continue
        start, end = int(segment["start"]), int(segment["end"])
        if segment["role"] == "user":
            all_embeds.append(
                _build_mlx_user_part(
                    thinker_embed=thinker_embed[start:end],
                    thinker_hidden=thinker_hidden[start:end],
                    multimodal_mask=multimodal_mask[start:end],
                    text_projection=text_projection,
                    hidden_projection=hidden_projection,
                )
            )
            all_ids.append(mx.array(thinker_input_ids[start:end], dtype=mx.int32))
        elif segment["role"] == "assistant":
            if last_assistant is not None and segment_index != last_assistant:
                continue
            assistant_embed = thinker_embed[start:end]
            if (
                assistant_embed.shape[0] > 0
                and thinker_input_ids[end - 1] == special_token_ids["im_end_token_id"]
            ):
                assistant_embed = assistant_embed[:-1]
            assistant = _build_mlx_assistant_part(
                assistant_embed=assistant_embed,
                text_projection=text_projection,
                codec_embedding=codec_embedding,
                tts_bos_embed=tts_bos_embed,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                speaker_id=speaker_id,
                special_token_ids=special_token_ids,
            )
            all_embeds.append(assistant["input_embeds"])
            all_ids.append(assistant["input_ids"])
            future_text_rows = assistant["future_text_rows"]
            if (
                not include_assistant_eos
                and future_text_rows is not None
                and future_text_rows.shape[0] > 0
            ):
                future_text_rows = future_text_rows[:-1]

    return {
        "input_embeds": mx.concatenate(all_embeds, axis=0),
        "input_ids": mx.concatenate(all_ids, axis=0),
        "future_text_rows": future_text_rows,
    }


class Qwen3OmniMlxTalkerPrefillBuilder:
    def __init__(
        self,
        *,
        talker: Qwen3OmniMlxTalker,
        model_path: str,
        special_token_ids: Mapping[str, int],
        speaker_map: Mapping[str, int],
    ) -> None:
        model_dir = Path(model_path)
        self._model_path = str(
            model_dir
            if model_dir.exists()
            else resolve_model_path(model_path, local_files_only=False)
        )
        self._talker = talker
        self._special_token_ids = {
            name: int(token_id) for name, token_id in special_token_ids.items()
        }
        self._speaker_map = {
            str(name).lower(): int(speaker_id)
            for name, speaker_id in speaker_map.items()
        }
        self._embedding_row_reader = _SafetensorRowReader()
        self._thinker_embed_cache: dict[int, mx.array] = {}
        self._tts_special_cache: tuple[mx.array, mx.array, mx.array] | None = None

    @classmethod
    def from_talker(
        cls,
        talker: Qwen3OmniMlxTalker,
        *,
        model_path: str,
        special_token_ids: Mapping[str, int],
        speaker_map: Mapping[str, int],
    ) -> "Qwen3OmniMlxTalkerPrefillBuilder":
        return cls(
            talker=talker,
            model_path=model_path,
            special_token_ids=special_token_ids,
            speaker_map=speaker_map,
        )

    def build_prompt_prefill(
        self,
        payload: StagePayload,
        thinker_chunks: list[Any],
        *,
        thinker_done: bool,
    ) -> dict[str, Any]:
        if not thinker_chunks:
            raise ValueError("prompt prefill requires thinker chunks")

        state = Qwen3OmniPipelineState.from_dict(payload.data)
        prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs = (
            self._reconstruct_prompt_states(state)
        )
        assistant_ids = self._extract_chunk_token_ids(thinker_chunks)
        assistant_embed = self._load_prompt_token_embeddings(assistant_ids)
        thinker_ids = [*prompt_ids, *assistant_ids]
        thinker_embed = mx.concatenate([prompt_embed, assistant_embed], axis=0)
        thinker_hidden = mx.concatenate([prompt_hidden, assistant_embed], axis=0)
        multimodal_mask = self._build_multimodal_mask(thinker_ids)
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_tts_special_embeds()
        prefill = build_mlx_prefill_input(
            thinker_embed=thinker_embed,
            thinker_hidden=thinker_hidden,
            thinker_input_ids=thinker_ids,
            multimodal_mask=multimodal_mask,
            text_projection=self._talker.text_projection,
            hidden_projection=self._talker.hidden_projection,
            codec_embedding=self._talker.model.codec_embedding,
            tts_bos_embed=tts_bos_embed,
            tts_eos_embed=tts_eos_embed,
            tts_pad_embed=tts_pad_embed,
            special_token_ids=self._special_token_ids,
            speaker_id=resolve_speaker_id(payload.request.params, self._speaker_map),
            include_assistant_eos=thinker_done,
        )
        input_embeds = prefill["input_embeds"]
        input_ids = prefill["input_ids"]
        future_text_rows = prefill["future_text_rows"]
        assert input_embeds is not None
        assert input_ids is not None
        mx.eval(input_embeds, input_ids, tts_pad_embed, tts_eos_embed)
        if future_text_rows is not None:
            mx.eval(future_text_rows)
            pending_text_queue = PendingTextTensorQueue.from_tensor(
                _mlx_to_torch(future_text_rows)
            )
        else:
            pending_text_queue = PendingTextTensorQueue()
        return {
            "input_embeds": _mlx_to_torch(input_embeds),
            "input_ids": _mlx_to_torch(input_ids).to(torch.long),
            "pending_text_queue": pending_text_queue,
            "tts_pad_embed": _mlx_to_torch(tts_pad_embed[0]),
            "tts_eos_embed": _mlx_to_torch(tts_eos_embed[0]),
            "prompt_model_inputs": prompt_model_inputs,
        }

    def append_text_chunk(self, req_data: Any, chunk: Any) -> None:
        if req_data.thinker_chunks_done:
            return
        metadata = chunk.metadata or {}
        token_id = metadata.get("token_id")
        if (
            token_id is not None
            and int(token_id) == self._special_token_ids["im_end_token_id"]
        ):
            return
        pending = getattr(req_data, "pending_text_queue", None)
        if not isinstance(pending, PendingTextTensorQueue):
            pending = coerce_pending_text_queue(pending)
            req_data.pending_text_queue = pending
        pending.append(self._project_assistant_chunk(chunk))

    def mark_thinker_done(self, req_data: Any) -> None:
        if req_data.thinker_chunks_done:
            return
        req_data.thinker_chunks_done = True
        pending = getattr(req_data, "pending_text_queue", None)
        if not isinstance(pending, PendingTextTensorQueue):
            pending = coerce_pending_text_queue(pending)
            req_data.pending_text_queue = pending
        if isinstance(req_data.tts_eos_embed, torch.Tensor):
            pending.append(req_data.tts_eos_embed)

    def _extract_chunk_token_ids(self, thinker_chunks: list[Any]) -> list[int]:
        return [int((chunk.metadata or {})["token_id"]) for chunk in thinker_chunks]

    def _project_assistant_chunk(self, chunk: Any) -> torch.Tensor:
        metadata = chunk.metadata or {}
        token_id = metadata.get("token_id")
        if token_id is not None:
            row = self._load_prompt_token_embeddings([int(token_id)])
        else:
            row = _torch_to_mlx(chunk.data).reshape(1, -1)
        projected = self._talker.text_projection(row)
        mx.eval(projected)
        return _mlx_to_torch(projected[0])

    def _build_multimodal_mask(self, token_ids: list[int]) -> mx.array:
        ids = mx.array(token_ids, dtype=mx.int32)
        mask = mx.zeros((len(token_ids),), dtype=mx.bool_)
        for name in ("audio_token_id", "image_token_id", "video_token_id"):
            token_id = self._special_token_ids.get(name)
            if token_id is not None:
                mask = mx.logical_or(mask, ids == int(token_id))
        return mask

    def _get_tts_special_embeds(
        self,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if self._tts_special_cache is None:
            rows = self._load_prompt_token_embeddings(
                [
                    self._special_token_ids["tts_bos_token_id"],
                    self._special_token_ids["tts_eos_token_id"],
                    self._special_token_ids["tts_pad_token_id"],
                ]
            )
            projected = self._talker.text_projection(rows)
            mx.eval(projected)
            self._tts_special_cache = (
                projected[0:1],
                projected[1:2],
                projected[2:3],
            )
        return self._tts_special_cache

    def _reconstruct_prompt_states(
        self,
        state: Qwen3OmniPipelineState,
    ) -> tuple[list[int], mx.array, mx.array, dict[str, Any]]:
        prompt = state.prompt or {}
        prompt_input_ids = prompt["input_ids"]
        if prompt_input_ids.dim() == 2:
            prompt_input_ids = prompt_input_ids[0]
        prompt_ids = [int(token_id) for token_id in prompt_input_ids.tolist()]
        prompt_embed = self._load_prompt_token_embeddings(prompt_ids)
        prompt_hidden = mx.array(prompt_embed)
        prompt_model_inputs = self._prompt_model_inputs(state)

        for name in ("audio", "image", "video"):
            token_id = self._special_token_ids.get(f"{name}_token_id")
            features = _coerce_feature_array(prompt_model_inputs.get(f"{name}_embeds"))
            if token_id is None or features is None:
                continue
            positions = [
                index
                for index, prompt_id in enumerate(prompt_ids)
                if prompt_id == int(token_id)
            ]
            count = len(positions)
            if count == 0:
                continue
            if int(features.shape[0]) != count:
                raise ValueError(
                    f"{name} feature rows ({features.shape[0]}) do not match "
                    f"prompt placeholders ({count})"
                )
            indices = mx.array(positions, dtype=mx.int32)
            prompt_embed[indices] = features.astype(prompt_embed.dtype)
            prompt_hidden[indices] = mx.zeros(
                (count, prompt_hidden.shape[-1]),
                dtype=prompt_hidden.dtype,
            )
        return prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs

    def _load_prompt_token_embeddings(self, token_ids: list[int]) -> mx.array:
        missing = [
            token_id
            for token_id in dict.fromkeys(token_ids)
            if token_id not in self._thinker_embed_cache
        ]
        if missing:
            rows = _load_mlx_embedding_rows(
                self._model_path,
                missing,
                row_reader=self._embedding_row_reader,
            )
            mx.eval(rows)
            for token_id, row in zip(missing, rows):
                self._thinker_embed_cache[token_id] = mx.array(row)
        return mx.stack([self._thinker_embed_cache[token_id] for token_id in token_ids])

    @staticmethod
    def _prompt_model_inputs(state: Qwen3OmniPipelineState) -> dict[str, Any]:
        thinker_inputs = state.thinker_inputs or {}
        model_inputs = thinker_inputs.get("model_inputs")
        if isinstance(model_inputs, dict):
            return dict(model_inputs)
        prompt_model_inputs = dict(thinker_inputs)
        prompt_model_inputs.pop("capture_model_output_keys", None)
        return prompt_model_inputs
