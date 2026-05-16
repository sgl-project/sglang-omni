# SPDX-License-Identifier: Apache-2.0
"""Prompt-aware talker prefill helpers.

This module mirrors HF's talker prefill layout, then converts HF's
``trailing_text_hidden`` tensor into a host-side FIFO queue of future text rows.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from sglang_omni.models.qwen3_omni.components.talker_input import build_prefill_input
from sglang_omni.models.qwen3_omni.payload_types import PipelineState
from sglang_omni.models.weight_loader import resolve_model_path

_THINKER_EMBED_CANDIDATE_KEYS = (
    "thinker.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


def _read_rows_from_safetensor(
    file_path: Path, tensor_name: str, row_ids: list[int]
) -> torch.Tensor:
    with safe_open(str(file_path), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_name)
        try:
            rows = [tensor_slice[row_id] for row_id in row_ids]
            return torch.stack(rows, dim=0)
        except (IndexError, RuntimeError, TypeError, ValueError):
            tensor = handle.get_tensor(tensor_name)
            return tensor[row_ids]


def load_thinker_embedding_rows(model_path: str, row_ids: list[int]) -> torch.Tensor:
    model_dir = Path(model_path)

    for index_path in model_dir.glob("*.safetensors.index.json"):
        index_data = json.loads(index_path.read_text())
        weight_map = index_data["weight_map"]
        for tensor_name in _THINKER_EMBED_CANDIDATE_KEYS:
            shard_name = weight_map.get(tensor_name)
            if shard_name is not None:
                return _read_rows_from_safetensor(
                    model_dir / shard_name, tensor_name, row_ids
                )

    for shard_path in model_dir.glob("*.safetensors"):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for tensor_name in _THINKER_EMBED_CANDIDATE_KEYS:
                if tensor_name in handle.keys():
                    return _read_rows_from_safetensor(shard_path, tensor_name, row_ids)

    raise KeyError(f"Unable to locate thinker embedding weights in {model_path}")


def coerce_feature_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, (list, tuple)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if not tensors:
            return None
        tensor = torch.cat(tensors, dim=0)
    else:
        return None

    if tensor.dim() == 3 and tensor.shape[0] == 1:
        return tensor[0]
    if tensor.dim() > 2:
        return tensor.reshape(-1, tensor.shape[-1])
    return tensor


def merge_prompt_modality(
    prompt_ids: torch.Tensor,
    prompt_embed: torch.Tensor,
    prompt_hidden: torch.Tensor,
    *,
    token_id: int | None,
    features: Any,
) -> None:
    if token_id is None:
        return
    feature_tensor = coerce_feature_tensor(features)
    if feature_tensor is None:
        return

    mask = prompt_ids == int(token_id)
    if not mask.any():
        return

    prompt_embed[mask] = feature_tensor.to(
        device=prompt_embed.device,
        dtype=prompt_embed.dtype,
    )
    prompt_hidden[mask] = 0.0


def resolve_speaker_id(params: dict[str, Any], speaker_map: dict[str, int]) -> int:
    speaker_name = str(params.get("speaker", "Ethan")).lower()
    if speaker_name in speaker_map:
        return speaker_map[speaker_name]
    if speaker_map:
        return next(iter(speaker_map.values()))
    return int(params.get("speaker_id", 0))


class TalkerPrefillBuilder:
    def __init__(
        self,
        *,
        model: Any,
        model_path: str,
        audio_token_id: int | None,
        image_token_id: int | None,
        video_token_id: int | None,
        tts_bos_token_id: int,
        tts_eos_token_id: int,
        tts_pad_token_id: int,
        im_start_token_id: int,
        im_end_token_id: int,
        system_token_id: int,
        user_token_id: int,
        assistant_token_id: int,
        codec_bos_id: int,
        codec_nothink_id: int,
        codec_think_bos_id: int,
        codec_think_eos_id: int,
        codec_pad_id: int,
        speaker_map: dict[str, int] | None = None,
    ) -> None:
        self._model = model
        model_dir = Path(model_path)
        if model_dir.exists():
            self._model_path = str(model_dir)
        else:
            self._model_path = str(
                resolve_model_path(model_path, local_files_only=False)
            )

        self._audio_token_id = audio_token_id
        self._image_token_id = image_token_id
        self._video_token_id = video_token_id
        self._tts_bos_token_id = tts_bos_token_id
        self._tts_eos_token_id = tts_eos_token_id
        self._tts_pad_token_id = tts_pad_token_id
        self._im_start_token_id = im_start_token_id
        self._im_end_token_id = im_end_token_id
        self._system_token_id = system_token_id
        self._user_token_id = user_token_id
        self._assistant_token_id = assistant_token_id
        self._codec_bos_id = codec_bos_id
        self._codec_nothink_id = codec_nothink_id
        self._codec_think_bos_id = codec_think_bos_id
        self._codec_think_eos_id = codec_think_eos_id
        self._codec_pad_id = codec_pad_id
        self._speaker_map = {
            str(name).lower(): int(speaker_id)
            for name, speaker_id in (speaker_map or {}).items()
        }

        parameter = next(model.parameters())
        self._device = parameter.device
        self._dtype = parameter.dtype
        self._thinker_embed_cache: dict[int, torch.Tensor] = {}
        self._tts_special_cache: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

    def build_prompt_prefill(
        self,
        payload,
        thinker_chunks: list[Any],
        *,
        thinker_done: bool,
    ) -> dict[str, Any]:
        if not thinker_chunks:
            raise ValueError("prompt prefill requires thinker chunks")

        state = PipelineState.from_dict(payload.data)
        prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs = (
            self._reconstruct_prompt_states(state)
        )

        assistant_token_ids = self.extract_chunk_token_ids(thinker_chunks)
        assistant_embed = self._load_prompt_token_embeddings(assistant_token_ids)
        assistant_hidden = torch.stack(
            [self.chunk_layer_hidden_or_embed(chunk) for chunk in thinker_chunks],
            dim=0,
        ).to(device=self._device, dtype=self._dtype)

        thinker_input_ids = torch.cat([prompt_ids, assistant_token_ids], dim=0)
        thinker_embed = torch.cat([prompt_embed, assistant_embed], dim=0)
        thinker_hidden = torch.cat([prompt_hidden, assistant_hidden], dim=0)
        multimodal_mask = self.build_multimodal_mask(thinker_input_ids)

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.get_tts_special_embeds()
        speaker_id = resolve_speaker_id(payload.request.params, self._speaker_map)

        prefill = build_prefill_input(
            thinker_embed=thinker_embed,
            thinker_hidden=thinker_hidden,
            thinker_input_ids=thinker_input_ids,
            multimodal_mask=multimodal_mask,
            text_projection=self._model.text_projection,
            hidden_projection=self._model.hidden_projection,
            codec_embed_fn=self._model.get_input_embeddings(),
            tts_bos_embed=tts_bos_embed,
            tts_eos_embed=tts_eos_embed,
            tts_pad_embed=tts_pad_embed,
            im_start_token_id=self._im_start_token_id,
            system_token_id=self._system_token_id,
            user_token_id=self._user_token_id,
            assistant_token_id=self._assistant_token_id,
            speaker_id=speaker_id,
            codec_nothink_id=self._codec_nothink_id,
            codec_think_bos_id=self._codec_think_bos_id,
            codec_think_eos_id=self._codec_think_eos_id,
            codec_pad_id=self._codec_pad_id,
            codec_bos_id=self._codec_bos_id,
            tts_pad_token_id=self._tts_pad_token_id,
            include_assistant_eos=thinker_done,
            im_end_token_id=self._im_end_token_id,
        )

        return {
            "input_embeds": prefill["input_embeds"],
            "input_ids": prefill["input_ids"],
            "pending_text_queue": self.tensor_rows_to_queue(
                prefill["future_text_rows"]
            ),
            "tts_pad_embed": tts_pad_embed[0].detach().cpu(),
            "tts_eos_embed": tts_eos_embed[0].detach().cpu(),
            "prompt_model_inputs": prompt_model_inputs,
        }

    def append_text_chunk(self, req_data: Any, chunk: Any) -> None:
        if getattr(req_data, "thinker_chunks_done", False):
            return

        metadata = chunk.metadata or {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == self._im_end_token_id:
            return

        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = collections.deque()
            req_data.pending_text_queue = pending_text_queue
        pending_text_queue.append(self.project_assistant_chunk(chunk).cpu())

    def mark_thinker_done(self, req_data: Any) -> None:
        if req_data.thinker_chunks_done:
            return

        req_data.thinker_chunks_done = True
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = collections.deque()
            req_data.pending_text_queue = pending_text_queue
        if isinstance(
            pending_text_queue,
            collections.deque,
        ) and isinstance(req_data.tts_eos_embed, torch.Tensor):
            pending_text_queue.append(req_data.tts_eos_embed.detach().cpu())

    def extract_chunk_token_ids(self, thinker_chunks: list[Any]) -> torch.Tensor:
        token_ids = []
        for chunk in thinker_chunks:
            metadata = chunk.metadata or {}
            token_ids.append(int(metadata["token_id"]))
        return torch.tensor(token_ids, dtype=torch.long)

    def chunk_layer_hidden_or_embed(self, chunk: Any) -> torch.Tensor:
        metadata = chunk.metadata or {}
        layer_hidden = metadata.get("layer_hidden")
        if isinstance(layer_hidden, torch.Tensor):
            return layer_hidden
        return chunk.data

    def project_assistant_chunk(self, chunk: Any) -> torch.Tensor:
        metadata = chunk.metadata or {}
        token_id = metadata.get("token_id")
        if token_id is not None:
            chunk_tensor = self._load_prompt_token_embeddings(
                torch.tensor([int(token_id)], dtype=torch.long)
            )
        else:
            chunk_tensor = chunk.data.to(
                device=self._device, dtype=self._dtype
            ).unsqueeze(0)
        projected = self._model.text_projection(chunk_tensor)
        return projected[0].detach()

    def build_multimodal_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=self._device)
        token_ids = token_ids.to(device=self._device)
        for token_id in (
            self._audio_token_id,
            self._image_token_id,
            self._video_token_id,
        ):
            if token_id is not None:
                mask |= token_ids == int(token_id)
        return mask

    def get_tts_special_embeds(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._tts_special_cache is None:
            special_rows = load_thinker_embedding_rows(
                self._model_path,
                [
                    self._tts_bos_token_id,
                    self._tts_eos_token_id,
                    self._tts_pad_token_id,
                ],
            ).to(device=self._device, dtype=self._dtype)
            projected = self._model.text_projection(special_rows)
            self._tts_special_cache = projected.chunk(3, dim=0)
        return self._tts_special_cache

    def tensor_rows_to_queue(
        self, tensor: torch.Tensor | None
    ) -> collections.deque[torch.Tensor]:
        queue: collections.deque[torch.Tensor] = collections.deque()
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return queue
        for row in tensor:
            queue.append(row.detach().cpu())
        return queue

    def _reconstruct_prompt_states(
        self, state: PipelineState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        prompt = state.prompt or {}
        prompt_input_ids = prompt["input_ids"]
        if prompt_input_ids.dim() == 2:
            prompt_input_ids = prompt_input_ids[0]
        prompt_ids = prompt_input_ids.to(dtype=torch.long).cpu()

        prompt_embed = self._load_prompt_token_embeddings(prompt_ids)
        prompt_hidden = prompt_embed.clone()
        prompt_model_inputs = self._prompt_model_inputs(state)

        merge_prompt_modality(
            prompt_ids,
            prompt_embed,
            prompt_hidden,
            token_id=self._audio_token_id,
            features=prompt_model_inputs.get("audio_embeds"),
        )
        merge_prompt_modality(
            prompt_ids,
            prompt_embed,
            prompt_hidden,
            token_id=self._image_token_id,
            features=prompt_model_inputs.get("image_embeds"),
        )
        merge_prompt_modality(
            prompt_ids,
            prompt_embed,
            prompt_hidden,
            token_id=self._video_token_id,
            features=prompt_model_inputs.get("video_embeds"),
        )

        return prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs

    def _load_prompt_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(dtype=torch.long).view(-1).cpu()
        unique_ids, inverse = torch.unique(token_ids, sorted=False, return_inverse=True)
        missing_ids = [
            int(token_id)
            for token_id in unique_ids.tolist()
            if int(token_id) not in self._thinker_embed_cache
        ]
        if missing_ids:
            loaded_rows = load_thinker_embedding_rows(self._model_path, missing_ids).to(
                device=self._device,
                dtype=self._dtype,
            )
            for token_id, row in zip(missing_ids, loaded_rows):
                self._thinker_embed_cache[int(token_id)] = row.detach().clone()

        unique_rows = torch.stack(
            [
                self._thinker_embed_cache[int(token_id)]
                for token_id in unique_ids.tolist()
            ],
            dim=0,
        )
        gathered = unique_rows.index_select(0, inverse.to(device=unique_rows.device))
        return gathered.view(token_ids.shape[0], unique_rows.shape[-1])

    def _prompt_model_inputs(self, state: PipelineState) -> dict[str, Any]:
        thinker_inputs = state.thinker_inputs or {}
        model_inputs = thinker_inputs.get("model_inputs")
        if isinstance(model_inputs, dict):
            return dict(model_inputs)

        prompt_model_inputs = dict(thinker_inputs)
        prompt_model_inputs.pop("capture_model_output_keys", None)
        return prompt_model_inputs
