"""Prefix cache."""

import hashlib
import math
import os
import time
from pathlib import Path

import torch.utils.checkpoint
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache, DynamicLayer

from sglang_omni.models.sensenova_u1.neo_unify.thinking_types import (
    PrefixMetadata,
    PrefixTimings,
)

# Modified for SGLang; see this directory's README.md for upstream source.


class BatchedSRTThinkingFallbackRequired(RuntimeError):
    """Retained for optional SRT paths; the first Omni pipeline uses native T2I."""


KV_DIAGNOSTIC_DIR: str = "SGLANG_SENSENOVA_KV_DIAGNOSTIC_DIR"


KV_TRANSFER_DIR: str = "SGLANG_SENSENOVA_KV_TRANSFER_DIR"


USE_SRT_KV_TRANSFER: str = "SGLANG_SENSENOVA_USE_SRT_KV_TRANSFER"


KV_IMPORT_PINNED_CHUNK_BYTES: int = 64 * 1024**2


def token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()


def dump_native_prefix_kv(
    cache: DynamicCache, token_ids: list[int], dump_id: str
) -> None:
    output_dir = os.environ.get(KV_DIAGNOSTIC_DIR)
    if not output_dir:
        return
    else:
        pass
    layer = cache.layers[0]
    payload = {
        "source": "native",
        "layer_id": 0,
        "token_ids": token_ids,
        "token_sha256": token_ids_sha256(token_ids),
        "keys": layer.keys[0, :, : len(token_ids), :].contiguous().cpu(),
        "values": layer.values[0, :, : len(token_ids), :].contiguous().cpu(),
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path / f"{dump_id}-native.pt")


def load_srt_prefix_kv(
    metadata: PrefixMetadata,
    token_ids: list[int],
    config: Qwen3Config,
    device: torch.device,
) -> tuple[DynamicCache, PrefixTimings]:
    output_dir = Path(os.environ[KV_TRANSFER_DIR])
    metadata_path = output_dir / f"{metadata.get('dump_id')}.json"
    data_path = output_dir / Path(str(metadata.get("data_file"))).name
    lock_file = metadata.get("lock_file")
    cleanup = [metadata_path]
    if metadata.get("reuse_buffer"):
        if lock_file:
            cleanup.append(output_dir / Path(str(lock_file)).name)
        else:
            pass
    else:
        cleanup.append(data_path)
    try:
        expected_shape = (
            int(config.num_hidden_layers),
            2,
            int(config.num_key_value_heads),
            len(token_ids),
            int(config.head_dim),
        )
        shape = tuple(metadata.get("shape") or ())
        if shape != expected_shape:
            raise RuntimeError(
                f"SenseNova SRT KV shape {shape} does not match {expected_shape}"
            )
        else:
            pass
        if metadata.get("layer_ids") != list(range(expected_shape[0])):
            raise RuntimeError("SenseNova SRT KV layer order does not match")
        else:
            pass
        if metadata.get("dtype") != "torch.bfloat16":
            raise RuntimeError(
                f"SenseNova SRT KV dtype is {metadata.get('dtype')}, expected BF16"
            )
        else:
            pass
        if metadata.get("token_sha256") != token_ids_sha256(token_ids):
            raise RuntimeError("SenseNova SRT KV token hash does not match")
        else:
            pass
        element_size = torch.empty((), dtype=torch.bfloat16).element_size()
        expected_bytes = math.prod(shape) * element_size
        if metadata.get("data_bytes") != expected_bytes:
            raise RuntimeError("SenseNova SRT KV metadata has an invalid byte count")
        else:
            pass
        if data_path.stat().st_size < expected_bytes:
            raise RuntimeError("SenseNova SRT KV shared buffer is incomplete")
        else:
            pass

        mapped = torch.from_file(
            str(data_path),
            shared=False,
            size=math.prod(shape),
            dtype=torch.bfloat16,
        ).view(shape)
        pin_memory = device.type == "cuda"
        layer_bytes = math.prod(shape[1:]) * element_size
        layers_per_chunk = max(
            1,
            min(shape[0], KV_IMPORT_PINNED_CHUNK_BYTES // layer_bytes),
        )
        pinned = torch.empty(
            (layers_per_chunk, *shape[1:]),
            dtype=torch.bfloat16,
            pin_memory=pin_memory,
        )
        cache = DynamicCache(config=config)
        shared_to_pinned_ms = 0.0
        h2d_ms = 0.0
        wall_started = time.perf_counter()
        for chunk_start in range(0, shape[0], layers_per_chunk):
            chunk_end = min(chunk_start + layers_per_chunk, shape[0])
            chunk_layers = chunk_end - chunk_start
            stage_started = time.perf_counter()
            pinned[:chunk_layers].copy_(mapped[chunk_start:chunk_end])
            shared_to_pinned_ms += (time.perf_counter() - stage_started) * 1000
            stage_started = time.perf_counter()
            for chunk_layer in range(chunk_layers):
                layer = chunk_start + chunk_layer
                keys = pinned[chunk_layer, 0].unsqueeze(0)
                values = pinned[chunk_layer, 1].unsqueeze(0)
                if pin_memory:
                    keys = keys.to(device, non_blocking=True)
                    values = values.to(device, non_blocking=True)
                else:
                    keys = keys.clone()
                    values = values.clone()
                cache.update(keys, values, layer)
            if pin_memory:
                torch.cuda.synchronize(device)
            else:
                pass
            h2d_ms += (time.perf_counter() - stage_started) * 1000
        timings = {
            "shared_to_pinned": shared_to_pinned_ms,
            "h2d": h2d_ms,
            "import_wall": (time.perf_counter() - wall_started) * 1000,
            "export": metadata.get("timings_ms") or {},
            "session_reused_tokens": metadata.get("session_reused_tokens"),
            "srt_cached_tokens": (metadata.get("request_meta_info") or {}).get(
                "cached_tokens"
            ),
        }
        return cache, timings
    finally:
        for path in cleanup:
            path.unlink(missing_ok=True)


def merge_srt_prefix_caches(
    caches: list[DynamicCache], lengths: list[int], config: Qwen3Config
) -> DynamicCache:
    if len(caches) != len(lengths) or not caches:
        raise ValueError("SenseNova SRT cache batch does not match prefix lengths")
    else:
        pass
    if len(caches) == 1:
        return caches[0]
    else:
        pass

    max_length = max(lengths)
    merged = DynamicCache(config=config)
    layer_count = len(caches[0].layers)
    if any(len(cache.layers) != layer_count for cache in caches):
        raise ValueError("SenseNova SRT cache batches have different layer counts")
    else:
        pass
    for layer_index in range(layer_count):
        sample = caches[0].layers[layer_index]
        keys = sample.keys.new_zeros(
            (len(caches), sample.keys.shape[1], max_length, sample.keys.shape[3])
        )
        values = sample.values.new_zeros(
            (len(caches), sample.values.shape[1], max_length, sample.values.shape[3])
        )
        for batch_index, (cache, length) in enumerate(zip(caches, lengths)):
            layer = cache.layers[layer_index]
            if layer.keys.shape[0] != 1 or layer.keys.shape[2] != length:
                raise ValueError("SenseNova SRT cache has an invalid prefix shape")
            else:
                pass
            keys[batch_index, :, :length].copy_(layer.keys[0])
            values[batch_index, :, :length].copy_(layer.values[0])
        merged.update(keys, values, layer_index)
    return merged


class SenseNovaPreallocatedLayer(DynamicLayer):
    def __init__(self, layer: DynamicLayer, capacity: int) -> None:
        super().__init__()
        self.dtype, self.device = layer.keys.dtype, layer.keys.device
        self.is_initialized: bool = True
        length = layer.keys.shape[-2]
        key_shape = (*layer.keys.shape[:-2], capacity, layer.keys.shape[-1])
        value_shape = (*layer.values.shape[:-2], capacity, layer.values.shape[-1])
        self.key_buffer: torch.Tensor = torch.empty(
            key_shape, dtype=self.dtype, device=self.device
        )
        self.value_buffer: torch.Tensor = torch.empty(
            value_shape, dtype=self.dtype, device=self.device
        )
        self.key_buffer[..., :length, :].copy_(layer.keys)
        self.value_buffer[..., :length, :].copy_(layer.values)
        self.keys: torch.Tensor = self.key_buffer[..., :length, :]
        self.values: torch.Tensor = self.value_buffer[..., :length, :]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: dict[str, torch.Tensor] | None,
        **kwargs: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.keys.shape[-2]
        end = start + key_states.shape[-2]
        if end > self.key_buffer.shape[-2]:
            return super().update(key_states, value_states, *args, **kwargs)
        else:
            pass
        self.key_buffer[..., start:end, :].copy_(key_states)
        self.value_buffer[..., start:end, :].copy_(value_states)
        self.keys = self.key_buffer[..., :end, :]
        self.values = self.value_buffer[..., :end, :]
        return self.keys, self.values


def preallocate_think_cache(cache: DynamicCache, additional_tokens: int) -> None:
    if not cache.layers or not all(
        type(layer) is DynamicLayer and layer.keys is not None for layer in cache.layers
    ):
        return
    else:
        pass
    for index, layer in enumerate(cache.layers):
        cache.layers[index] = SenseNovaPreallocatedLayer(
            layer, layer.keys.shape[-2] + additional_tokens
        )
