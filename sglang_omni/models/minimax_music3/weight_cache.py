# SPDX-License-Identifier: Apache-2.0
"""Bounded runtime-weight snapshots and stable-storage restoration."""

from __future__ import annotations

import fcntl
import hashlib
import json
import mmap
import os
import shutil
import struct
import tempfile
import threading
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel, ConfigDict, Field
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 256 * 1024**2
CHUNK_BYTES = SHARD_BYTES - 4096
HASH_BLOCK_BYTES = 1024**2
CACHE_FORMAT_VERSION = 1


class TensorLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    storage: int = Field(ge=0)
    dtype: str
    shape: list[int]
    stride: list[int]
    offset: int = Field(ge=0)


class WeightShard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    storage: int = Field(ge=0)
    offset_bytes: int = Field(ge=0)
    length_bytes: int = Field(gt=0, le=CHUNK_BYTES)
    sha256: str


class WeightManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: str
    layouts: list[TensorLayout]
    storage_bytes: list[int]
    shards: list[WeightShard]


@dataclass(kw_only=True)
class StorageChunk:
    storage: int
    offset_bytes: int
    values: torch.Tensor
    mapping: mmap.mmap | None = None


class StagingRing:
    """One bounded host ring and copy stream shared by both stages."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.lock = threading.Lock()
        self.buffers = [
            torch.empty(SHARD_BYTES, dtype=torch.uint8, pin_memory=True)
            for _ in range(2)
        ]
        self.stream = torch.cuda.Stream(device=device)
        self.events = [torch.cuda.Event() for _ in self.buffers]
        self.next_buffer = 0

    def restore(self, storages: list[torch.Tensor], chunks: list[StorageChunk]) -> None:
        compute_stream = torch.cuda.current_stream(self.device)
        with self.lock, torch.cuda.stream(self.stream):
            self.stream.wait_stream(compute_stream)
            for chunk in chunks:
                index = self.next_buffer
                self.events[index].synchronize()
                staging = self.buffers[index][: chunk.values.numel()]
                staging.copy_(chunk.values)
                if chunk.mapping is not None:
                    chunk.mapping.madvise(mmap.MADV_DONTNEED)
                else:
                    pass
                target = storages[chunk.storage].narrow(
                    0, chunk.offset_bytes, staging.numel()
                )
                target.copy_(staging, non_blocking=True)
                self.events[index].record(self.stream)
                self.next_buffer = (index + 1) % len(self.buffers)
            self.stream.synchronize()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(paths: list[Path]) -> dict[str, str]:
    return {str(index): file_digest(path) for index, path in enumerate(sorted(paths))}


def storage_layout(
    modules: dict[str, torch.nn.Module],
) -> tuple[list[torch.Tensor], list[TensorLayout]]:
    storages: list[torch.Tensor] = []
    layouts: list[TensorLayout] = []
    storage_indices: dict[tuple[str, int, int], int] = {}
    for module_name, module in modules.items():
        for tensor_name, tensor in (
            *module.named_parameters(remove_duplicate=False),
            *module.named_buffers(remove_duplicate=False),
        ):
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if identity not in storage_indices:
                storage_indices[identity] = len(storages)
                storages.append(
                    torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                        storage, 0, (storage.nbytes(),), (1,)
                    )
                )
            else:
                pass
            layouts.append(
                TensorLayout(
                    name=f"{module_name}.{tensor_name}",
                    storage=storage_indices[identity],
                    dtype=str(tensor.dtype),
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    offset=tensor.storage_offset(),
                )
            )
    return storages, layouts


def writable_cache_directory(cache_dir: str | None) -> Path:
    directory = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path(HF_HUB_CACHE).parent / "music3-runtime-weights"
    )
    directory.mkdir(parents=True, exist_ok=True)
    mount_path = directory.resolve()
    mount_type = ""
    longest_mount = 0
    for mount_line in Path("/proc/self/mountinfo").read_text().splitlines():
        mount_fields, filesystem_fields = mount_line.split(" - ", 1)
        mountpoint = Path(mount_fields.split()[4].replace("\\040", " "))
        if (
            mount_path.is_relative_to(mountpoint)
            and len(str(mountpoint)) > longest_mount
        ):
            longest_mount = len(str(mountpoint))
            mount_type = filesystem_fields.split()[0]
        else:
            pass
    if mount_type in {"tmpfs", "ramfs"}:
        raise ValueError(
            f"Music3 runtime cache requires non-tmpfs storage: {directory}"
        )
    else:
        pass
    with tempfile.TemporaryFile(dir=directory) as probe:
        probe.write(b"music3")
        probe.flush()
        os.fsync(probe.fileno())
    return directory


class RuntimeWeights:
    """Snapshot finalized storage once, retaining aliases and arbitrary views."""

    def __init__(
        self,
        modules: dict[str, torch.nn.Module],
        *,
        source: Literal["mmap", "ram"],
        cache_dir: str | None,
        checkpoint_contents: dict[str, str],
        folding_backend: str,
    ) -> None:
        self.storages, layouts = storage_layout(modules)
        self.source = source
        self.mappings: list[mmap.mmap] = []
        self.chunks: list[StorageChunk] = []
        storage_bytes = [storage.numel() for storage in self.storages]
        if source == "ram":
            for storage_index, storage in enumerate(self.storages):
                for offset_bytes in range(0, storage.numel(), CHUNK_BYTES):
                    self.chunks.append(
                        StorageChunk(
                            storage=storage_index,
                            offset_bytes=offset_bytes,
                            values=storage[
                                offset_bytes : offset_bytes + CHUNK_BYTES
                            ].to("cpu", copy=True),
                        )
                    )
            self.cache_path: Path | None = None
        else:
            directory = writable_cache_directory(cache_dir)
            metadata = {
                "format": CACHE_FORMAT_VERSION,
                "checkpoint": checkpoint_contents,
                "versions": {
                    package: version(package)
                    for package in (
                        "torch",
                        "sglang",
                        "sglang-omni",
                        "torch_memory_saver",
                        "safetensors",
                    )
                },
                "folding_backend": folding_backend,
                "layouts": [layout.model_dump() for layout in layouts],
                "storage_bytes": storage_bytes,
            }
            identity = hashlib.sha256(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest()
            self.cache_path = directory / identity
            with (directory / f"{identity}.lock").open("a+b") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                for stale_directory in directory.glob(f".{identity}-*"):
                    shutil.rmtree(stale_directory)
                if not self.cache_path.exists():
                    required_bytes = sum(storage_bytes) + SHARD_BYTES
                    if shutil.disk_usage(directory).free < required_bytes:
                        raise OSError(
                            f"Insufficient disk space for Music3 runtime cache: "
                            f"requires {required_bytes} bytes in {directory}"
                        )
                    else:
                        pass
                    temporary_path = Path(
                        tempfile.mkdtemp(prefix=f".{identity}-", dir=directory)
                    )
                    try:
                        shards: list[WeightShard] = []
                        for storage_index, storage in enumerate(self.storages):
                            for offset_bytes in range(0, storage.numel(), CHUNK_BYTES):
                                values = (
                                    storage[offset_bytes : offset_bytes + CHUNK_BYTES]
                                    .cpu()
                                    .contiguous()
                                )
                                filename = f"weights-{len(shards):05d}.safetensors"
                                shard_path = temporary_path / filename
                                save_file({"bytes": values}, shard_path)
                                with shard_path.open("rb") as shard_file:
                                    os.fsync(shard_file.fileno())
                                shards.append(
                                    WeightShard(
                                        filename=filename,
                                        storage=storage_index,
                                        offset_bytes=offset_bytes,
                                        length_bytes=values.numel(),
                                        sha256=file_digest(shard_path),
                                    )
                                )
                                del values
                        manifest = WeightManifest(
                            identity=identity,
                            layouts=layouts,
                            storage_bytes=storage_bytes,
                            shards=shards,
                        )
                        manifest_path = temporary_path / "manifest.json"
                        manifest_path.write_text(manifest.model_dump_json())
                        with manifest_path.open("rb") as manifest_file:
                            os.fsync(manifest_file.fileno())
                        temporary_descriptor = os.open(
                            temporary_path, os.O_RDONLY | os.O_DIRECTORY
                        )
                        try:
                            os.fsync(temporary_descriptor)
                        finally:
                            os.close(temporary_descriptor)
                        temporary_path.rename(self.cache_path)
                        directory_descriptor = os.open(
                            directory, os.O_RDONLY | os.O_DIRECTORY
                        )
                        try:
                            os.fsync(directory_descriptor)
                        finally:
                            os.close(directory_descriptor)
                    finally:
                        if temporary_path.exists():
                            shutil.rmtree(temporary_path)
                        else:
                            pass
                else:
                    pass
                manifest = WeightManifest.model_validate_json(
                    (self.cache_path / "manifest.json").read_text()
                )
                if (
                    manifest.identity != identity
                    or manifest.layouts != layouts
                    or manifest.storage_bytes != storage_bytes
                ):
                    raise ValueError(f"Invalid Music3 runtime cache: {self.cache_path}")
                else:
                    pass
                restored_bytes = [0 for _ in storage_bytes]
                for shard in manifest.shards:
                    if (
                        Path(shard.filename).name != shard.filename
                        or shard.storage >= len(storage_bytes)
                        or shard.offset_bytes != restored_bytes[shard.storage]
                    ):
                        raise ValueError("Invalid Music3 cache shard layout")
                    else:
                        pass
                    shard_path = self.cache_path / shard.filename
                    if (
                        shard_path.stat().st_size > SHARD_BYTES
                        or file_digest(shard_path) != shard.sha256
                    ):
                        raise ValueError(f"Invalid Music3 cache shard: {shard_path}")
                    else:
                        pass
                    with safe_open(shard_path, framework="pt", device="cpu") as tensors:
                        values = tensors.get_tensor("bytes")
                        if values.dtype != torch.uint8 or values.shape != (
                            shard.length_bytes,
                        ):
                            raise ValueError(
                                f"Invalid Music3 cache tensor: {shard_path}"
                            )
                        else:
                            pass
                        del values
                    restored_bytes[shard.storage] += shard.length_bytes
                if restored_bytes != storage_bytes:
                    raise ValueError("Incomplete Music3 runtime cache")
                else:
                    pass
                for shard in manifest.shards:
                    shard_path = self.cache_path / shard.filename
                    with shard_path.open("rb") as shard_file:
                        mapping = mmap.mmap(
                            shard_file.fileno(), 0, access=mmap.ACCESS_READ
                        )
                    header_bytes = struct.unpack("<Q", mapping[:8])[0]
                    header = json.loads(mapping[8 : 8 + header_bytes])
                    if header["bytes"]["data_offsets"] != [0, shard.length_bytes]:
                        mapping.close()
                        raise ValueError(
                            f"Invalid Music3 cache tensor offsets: {shard_path}"
                        )
                    else:
                        pass
                    self.mappings.append(mapping)
                    self.chunks.append(
                        StorageChunk(
                            storage=shard.storage,
                            offset_bytes=shard.offset_bytes,
                            mapping=mapping,
                            values=torch.frombuffer(
                                mapping,
                                dtype=torch.uint8,
                                count=shard.length_bytes,
                                offset=8 + header_bytes,
                            ),
                        )
                    )

    def restore(self, staging_ring: StagingRing | None) -> None:
        if staging_ring is None:
            for chunk in self.chunks:
                self.storages[chunk.storage].narrow(
                    0, chunk.offset_bytes, chunk.values.numel()
                ).copy_(chunk.values)
        else:
            staging_ring.restore(self.storages, self.chunks)
        for mapping in self.mappings:
            mapping.madvise(mmap.MADV_DONTNEED)

    def close(self) -> None:
        self.chunks.clear()
        for mapping in self.mappings:
            mapping.close()
        self.mappings.clear()
