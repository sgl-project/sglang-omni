# SPDX-License-Identifier: Apache-2.0
"""Prompt-aware talker prefill helpers.

This module mirrors HF's talker prefill layout, then keeps HF's
``trailing_text_hidden`` tensor as a device-backed FIFO of future text rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from sglang_omni.models.qwen3_omni.components.talker_input import build_prefill_input
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
)
from sglang_omni.models.weight_loader import resolve_model_path

_THINKER_EMBED_OFFICIAL_KEY = "thinker.model.embed_tokens.weight"
_THINKER_EMBED_MLX_VLM_KEY = "thinker.language_model.model.embed_tokens.weight"
_THINKER_EMBED_LOCAL_KEY = "model.embed_tokens.weight"
_THINKER_EMBED_CANDIDATE_KEYS = (
    _THINKER_EMBED_OFFICIAL_KEY,
    _THINKER_EMBED_MLX_VLM_KEY,
    _THINKER_EMBED_LOCAL_KEY,
)
# Subdirectory names a converted MLX export uses, matching the ownership rules
# in ``sglang_omni.models.qwen3_omni.mlx.runner``.
_COMPONENT_DIRECTORIES = ("thinker", "talker", "code2wav")


@dataclass(frozen=True)
class _EmbedSource:
    """Where the thinker embedding table lives, and how it is stored."""

    shard: Path
    tensor_name: str
    scales_shard: Path | None = None
    biases_shard: Path | None = None
    #: ``{"bits", "group_size", "mode"}`` when the table is packed 4/8-bit.
    quantization: dict[str, Any] | None = None


_EMBED_SOURCE_CACHE: dict[str, _EmbedSource] = {}
_EMBED_HANDLE_CACHE: dict[Path, Any] = {}


def _component_directory(shard: Path, root: Path) -> str | None:
    try:
        parts = shard.relative_to(root).parts
    except ValueError:  # pragma: no cover - rglob results are always under root
        return None
    if len(parts) > 1 and parts[0] in _COMPONENT_DIRECTORIES:
        return parts[0]
    return None


def _checkpoint_quantization(model_dir: Path) -> dict[str, Any] | None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    raw = json.loads(config_path.read_text())
    quantization = raw.get("quantization") or raw.get("quantization_config")
    return quantization if isinstance(quantization, dict) else None


def _embed_source_from_shard(
    shard: Path,
    *,
    root: Path,
    has_component_dirs: bool,
) -> _EmbedSource | None:
    """Claim the thinker embedding from ``shard`` only when it owns it.

    An official checkpoint namespaces the key (``thinker.model.embed_tokens``),
    so it is unambiguous wherever it lives. A converted export strips that
    namespace, and the *talker* owns an identically named
    ``model.embed_tokens.weight``; claiming it would splice the talker's table
    into the talker prompt with no error at all. So an unprefixed key is taken
    only from the ``thinker/`` component shard, or from the root when the
    export has no component directories.
    """

    component = _component_directory(shard, root)
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    for tensor_name in (
        _THINKER_EMBED_OFFICIAL_KEY,
        _THINKER_EMBED_MLX_VLM_KEY,
    ):
        if tensor_name not in keys:
            continue
        prefix = tensor_name[: -len(".weight")]
        quantization = None
        has_scales = f"{prefix}.scales" in keys
        has_biases = f"{prefix}.biases" in keys
        if has_scales != has_biases:
            raise KeyError(f"{shard} carries an incomplete packed thinker embedding")
        if has_scales:
            quantization = _checkpoint_quantization(root)
            if quantization is None:
                raise KeyError(
                    f"{shard} carries a packed thinker embedding but "
                    f"{root / 'config.json'} declares no quantization block"
                )
        return _EmbedSource(
            shard=shard,
            tensor_name=tensor_name,
            scales_shard=shard if has_scales else None,
            biases_shard=shard if has_biases else None,
            quantization=quantization,
        )
    if _THINKER_EMBED_LOCAL_KEY not in keys:
        return None
    if component == "thinker" or (component is None and not has_component_dirs):
        prefix = _THINKER_EMBED_LOCAL_KEY[: -len(".weight")]
        packed = f"{prefix}.scales" in keys
        has_biases = f"{prefix}.biases" in keys
        if packed != has_biases:
            raise KeyError(f"{shard} carries an incomplete packed thinker embedding")
        quantization = None
        if packed:
            quantization = _checkpoint_quantization(root)
            if quantization is None:
                raise KeyError(
                    f"{shard} carries a packed thinker embedding but "
                    f"{root / 'config.json'} declares no quantization block"
                )
        return _EmbedSource(
            shard=shard,
            tensor_name=_THINKER_EMBED_LOCAL_KEY,
            scales_shard=shard if packed else None,
            biases_shard=shard if has_biases else None,
            quantization=quantization,
        )
    return None


def _resolve_embed_source(model_path: str) -> _EmbedSource:
    cached = _EMBED_SOURCE_CACHE.get(model_path)
    if cached is not None:
        return cached

    model_dir = Path(model_path)
    index_paths = sorted(model_dir.rglob("*.safetensors.index.json"))
    has_component_indexes = any(
        _component_directory(index_path, model_dir) is not None
        for index_path in index_paths
    )
    for index_path in index_paths:
        index_data = json.loads(index_path.read_text())
        weight_map = index_data["weight_map"]
        component = _component_directory(index_path, model_dir)
        for tensor_name in _THINKER_EMBED_CANDIDATE_KEYS:
            if tensor_name == _THINKER_EMBED_LOCAL_KEY and not (
                component == "thinker"
                or (component is None and not has_component_indexes)
            ):
                continue
            shard_name = weight_map.get(tensor_name)
            if shard_name is not None:
                prefix = tensor_name[: -len(".weight")]
                quantization = None
                scales_name = weight_map.get(f"{prefix}.scales")
                biases_name = weight_map.get(f"{prefix}.biases")
                if (scales_name is None) != (biases_name is None):
                    raise KeyError(
                        f"{model_dir / shard_name} carries an incomplete packed "
                        "thinker embedding"
                    )
                if scales_name is not None:
                    quantization = _checkpoint_quantization(model_dir)
                    if quantization is None:
                        raise KeyError(
                            f"{model_dir / shard_name} carries a packed thinker "
                            f"embedding but {model_dir / 'config.json'} declares "
                            "no quantization block"
                        )
                source = _EmbedSource(
                    shard=index_path.parent / shard_name,
                    tensor_name=tensor_name,
                    scales_shard=(
                        index_path.parent / str(scales_name)
                        if scales_name is not None
                        else None
                    ),
                    biases_shard=(
                        index_path.parent / str(biases_name)
                        if biases_name is not None
                        else None
                    ),
                    quantization=quantization,
                )
                _EMBED_SOURCE_CACHE[model_path] = source
                return source

    shards = sorted(path for path in model_dir.rglob("*.safetensors") if path.is_file())
    has_component_dirs = any(
        _component_directory(shard, model_dir) is not None for shard in shards
    )
    for shard in shards:
        source = _embed_source_from_shard(
            shard, root=model_dir, has_component_dirs=has_component_dirs
        )
        if source is not None:
            _EMBED_SOURCE_CACHE[model_path] = source
            return source

    raise KeyError(f"Unable to locate thinker embedding weights in {model_path}")


def _safetensor_rows(
    shard: Path,
    tensor_name: str,
    row_ids: list[int],
) -> torch.Tensor:
    handle = _EMBED_HANDLE_CACHE.get(shard)
    if handle is None:
        handle = safe_open(str(shard), framework="pt", device="cpu")
        _EMBED_HANDLE_CACHE[shard] = handle
    tensor_slice = handle.get_slice(tensor_name)
    try:
        rows = [tensor_slice[row_id] for row_id in row_ids]
    except (IndexError, RuntimeError, TypeError, ValueError):
        tensor = handle.get_tensor(tensor_name)
        rows = [tensor[row_id].clone() for row_id in row_ids]
    return torch.stack(rows, dim=0)


def _packed_embedding_rows(
    model_path: str,
    source: _EmbedSource,
    row_ids: list[int],
) -> torch.Tensor:
    """Dequantize just the requested rows of a packed embedding table.

    Affine quantization groups along the last axis, so selecting rows first is
    exact and keeps a production-size table from being materialised in full.
    """

    from sglang_omni.models.qwen3_omni.apple_runtime import (
        get_qwen3_omni_mps_quantization,
    )

    if get_qwen3_omni_mps_quantization() is not None:
        raise ValueError(
            "Torch MPS HF INT4 requires dense thinker embeddings; "
            "MLX affine embedding conversion is not supported"
        )

    import mlx.core as mx
    import numpy as np

    dequantized = _packed_embedding_rows_mlx(model_path, source, row_ids)
    return torch.from_numpy(
        np.ascontiguousarray(np.asarray(dequantized.astype(mx.float32)))
    )


def _packed_embedding_tensors(
    source: _EmbedSource, row_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if source.scales_shard is None or source.biases_shard is None:
        raise KeyError(f"packed thinker embedding {source.tensor_name!r} is incomplete")
    prefix = source.tensor_name[: -len(".weight")]
    return (
        _safetensor_rows(source.shard, source.tensor_name, row_ids),
        _safetensor_rows(source.scales_shard, f"{prefix}.scales", row_ids),
        _safetensor_rows(source.biases_shard, f"{prefix}.biases", row_ids),
    )


def _packed_embedding_rows_mlx(
    model_path: str,
    source: _EmbedSource,
    row_ids: list[int],
):
    """Return selectively loaded packed rows with checkpoint dtypes preserved."""

    del model_path
    import mlx.core as mx

    from sglang_omni.models.qwen3_omni.mlx.tensor_utils import torch_to_mlx

    def to_mlx(tensor: torch.Tensor):
        array = torch_to_mlx(tensor)
        if tensor.dtype == torch.bfloat16:
            return array.astype(mx.bfloat16)
        return array

    weight, scales, biases = _packed_embedding_tensors(source, row_ids)
    quantization = source.quantization or {}
    return mx.dequantize(
        to_mlx(weight),
        to_mlx(scales),
        to_mlx(biases),
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=str(quantization.get("mode", "affine")),
    )


def load_thinker_embedding_rows(model_path: str, row_ids: list[int]) -> torch.Tensor:
    source = _resolve_embed_source(model_path)
    if source.quantization is not None:
        return _packed_embedding_rows(model_path, source, row_ids)
    return _safetensor_rows(source.shard, source.tensor_name, row_ids)


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

        self._device = model.model.codec_embedding.weight.device
        self._dtype = model.activation_dtype
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

        state = Qwen3OmniPipelineState.from_dict(payload.data)
        prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs = (
            self._reconstruct_prompt_states(state)
        )

        assistant_token_ids = self.extract_chunk_token_ids(thinker_chunks)
        assistant_embed = self._load_prompt_token_embeddings(assistant_token_ids)

        thinker_input_ids = torch.cat([prompt_ids, assistant_token_ids], dim=0)
        thinker_embed = torch.cat([prompt_embed, assistant_embed], dim=0)
        thinker_hidden = torch.cat([prompt_hidden, assistant_embed], dim=0)
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
            "tts_pad_embed": tts_pad_embed[0].detach(),
            "tts_eos_embed": tts_eos_embed[0].detach(),
            "prompt_model_inputs": prompt_model_inputs,
        }

    def append_text_chunk(self, req_data: Any, chunk: Any) -> None:
        if req_data.thinker_chunks_done:
            return

        metadata = chunk.metadata or {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == self._im_end_token_id:
            return

        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if not isinstance(pending_text_queue, PendingTextTensorQueue):
            pending_text_queue = coerce_pending_text_queue(pending_text_queue)
            req_data.pending_text_queue = pending_text_queue
        pending_text_queue.append(self.project_assistant_chunk(chunk))

    def mark_thinker_done(self, req_data: Any) -> None:
        if req_data.thinker_chunks_done:
            return

        req_data.thinker_chunks_done = True
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if not isinstance(pending_text_queue, PendingTextTensorQueue):
            pending_text_queue = coerce_pending_text_queue(pending_text_queue)
            req_data.pending_text_queue = pending_text_queue
        if isinstance(req_data.tts_eos_embed, torch.Tensor):
            pending_text_queue.append(req_data.tts_eos_embed)

    def extract_chunk_token_ids(self, thinker_chunks: list[Any]) -> torch.Tensor:
        token_ids = []
        for chunk in thinker_chunks:
            metadata = chunk.metadata or {}
            token_ids.append(int(metadata["token_id"]))
        return torch.tensor(token_ids, dtype=torch.long)

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
    ) -> PendingTextTensorQueue:
        if tensor is None:
            return PendingTextTensorQueue()
        return PendingTextTensorQueue.from_tensor(tensor)

    def _reconstruct_prompt_states(
        self, state: Qwen3OmniPipelineState
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

    def _prompt_model_inputs(self, state: Qwen3OmniPipelineState) -> dict[str, Any]:
        thinker_inputs = state.thinker_inputs or {}
        model_inputs = thinker_inputs.get("model_inputs")
        if isinstance(model_inputs, dict):
            return dict(model_inputs)

        prompt_model_inputs = dict(thinker_inputs)
        prompt_model_inputs.pop("capture_model_output_keys", None)
        return prompt_model_inputs
