# SPDX-License-Identifier: Apache-2.0
"""Prepare Torch-compatible code2wav weights from an MLX Qwen3-Omni export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang_omni.models.qwen3_omni.mlx.config import QuantizationConfig

_COMPONENT_PREFIX = "code2wav."
_INDEX_NAME = "model.safetensors.index.json"
_MANIFEST_NAME = "conversion.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read checkpoint metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {path}")
    return value


def _load_quantization(root: Path) -> QuantizationConfig:
    config = _read_json(root / "config.json")
    raw = config.get("quantization")
    if not isinstance(raw, dict):
        raw = config.get("quantization_config")
    if not isinstance(raw, dict):
        raise ValueError(
            f"Checkpoint {root} has no MLX quantization metadata in config.json"
        )
    quantization = QuantizationConfig.from_dict(raw)
    if quantization.mode != "affine":
        raise ValueError(
            "Code2wav preparation supports MLX affine quantization only; "
            f"got {quantization.mode!r}"
        )
    return quantization


def _load_weight_map(root: Path) -> tuple[Path, dict[str, str]]:
    index_path = root / _INDEX_NAME
    index = _read_json(index_path)
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise ValueError(f"Missing weight_map in checkpoint index {index_path}")
    weight_map = {str(key): str(value) for key, value in raw_weight_map.items()}
    if not any(key.startswith(_COMPONENT_PREFIX) for key in weight_map):
        raise ValueError(f"Checkpoint {root} contains no code2wav weights")
    return index_path, weight_map


def _validate_quantized_groups(weight_map: dict[str, str]) -> None:
    component_keys = {key for key in weight_map if key.startswith(_COMPONENT_PREFIX)}
    for key in sorted(component_keys):
        if key.endswith(".scales"):
            stem = key[: -len(".scales")]
            weight_key = f"{stem}.weight"
            biases_key = f"{stem}.biases"
            if weight_key not in component_keys:
                raise ValueError(f"Quantized code2wav tensor {key!r} has no weight")
            if biases_key not in component_keys:
                raise ValueError(f"Affine code2wav tensor {weight_key!r} has no biases")
        elif key.endswith(".biases"):
            stem = key[: -len(".biases")]
            weight_key = f"{stem}.weight"
            scales_key = f"{stem}.scales"
            if weight_key not in component_keys or scales_key not in component_keys:
                raise ValueError(
                    f"Affine code2wav tensor {key!r} has an incomplete quantized group"
                )


def _torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return mx.array(tensor.numpy())


def _mlx_to_bfloat16_tensor(array: mx.array) -> torch.Tensor:
    mx.eval(array)
    values = np.ascontiguousarray(np.asarray(array.astype(mx.float32)))
    return torch.from_numpy(values).to(torch.bfloat16).contiguous()


def _expected_code2wav_shapes(root: Path) -> dict[str, tuple[int, ...]]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    config = AutoConfig.from_pretrained(root, trust_remote_code=True)
    with init_empty_weights():
        model = Qwen3OmniMoeCode2Wav._from_config(config.code2wav_config)
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _restore_torch_tensor_layout(
    key: str,
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...] | None,
) -> torch.Tensor:
    if expected_shape is None or tuple(tensor.shape) == expected_shape:
        return tensor
    if tensor.ndim not in (3, 4, 5):
        raise ValueError(
            f"Prepared code2wav tensor {key!r} has shape {tuple(tensor.shape)}, "
            f"expected {expected_shape}"
        )

    permutations = (
        (0, tensor.ndim - 1, *range(1, tensor.ndim - 1)),
        (tensor.ndim - 1, 0, *range(1, tensor.ndim - 1)),
    )
    for permutation in permutations:
        candidate = tensor.permute(permutation)
        if tuple(candidate.shape) == expected_shape:
            return candidate.contiguous()
    raise ValueError(
        f"Prepared code2wav tensor {key!r} has MLX shape {tuple(tensor.shape)} "
        f"which cannot be restored to Torch shape {expected_shape}"
    )


def _read_selected_tensors(
    root: Path,
    weight_map: dict[str, str],
    keys: set[str],
) -> dict[str, torch.Tensor]:
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key in sorted(keys):
        shard_name = weight_map.get(key)
        if shard_name is None:
            raise ValueError(f"Checkpoint index has no shard for tensor {key!r}")
        keys_by_shard[shard_name].append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_keys in sorted(keys_by_shard.items()):
        shard_path = root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Checkpoint shard does not exist: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in shard_keys:
                if key not in available:
                    raise ValueError(
                        f"Checkpoint shard {shard_path} does not contain {key!r}"
                    )
                tensors[key] = handle.get_tensor(key)
    return tensors


def _convert_output_group(
    root: Path,
    weight_map: dict[str, str],
    source_keys: list[str],
    quantization: QuantizationConfig,
    expected_shapes: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    required_keys = set(source_keys)
    for weight_key in source_keys:
        if not weight_key.endswith(".weight"):
            continue
        stem = weight_key[: -len(".weight")]
        scales_key = f"{stem}.scales"
        if scales_key in weight_map:
            required_keys.add(scales_key)
            required_keys.add(f"{stem}.biases")

    source_tensors = _read_selected_tensors(root, weight_map, required_keys)
    converted: dict[str, torch.Tensor] = {}
    for source_key in source_keys:
        output_key = source_key[len(_COMPONENT_PREFIX) :]
        source_tensor = source_tensors[source_key]
        if source_key.endswith(".weight"):
            stem = source_key[: -len(".weight")]
            scales_key = f"{stem}.scales"
            biases_key = f"{stem}.biases"
            if scales_key in source_tensors:
                dense = mx.dequantize(
                    _torch_to_mlx(source_tensor),
                    _torch_to_mlx(source_tensors[scales_key]),
                    _torch_to_mlx(source_tensors[biases_key]),
                    group_size=quantization.group_size,
                    bits=quantization.bits,
                    mode=quantization.mode,
                    dtype=mx.bfloat16,
                )
                converted[output_key] = _restore_torch_tensor_layout(
                    output_key,
                    _mlx_to_bfloat16_tensor(dense),
                    expected_shapes.get(output_key),
                )
                continue
            if not source_tensor.is_floating_point():
                raise ValueError(
                    f"Code2wav weight {source_key!r} is packed but has no scales"
                )
        if source_tensor.is_floating_point():
            source_tensor = source_tensor.to(torch.bfloat16)
        converted[output_key] = _restore_torch_tensor_layout(
            output_key,
            source_tensor,
            expected_shapes.get(output_key),
        )
    return converted


def _output_groups(weight_map: dict[str, str]) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        if not key.startswith(_COMPONENT_PREFIX):
            continue
        if key.endswith((".scales", ".biases")):
            continue
        grouped[shard_name].append(key)
    if not grouped:
        raise ValueError("Checkpoint contains no materializable code2wav tensors")
    return [(shard_name, sorted(keys)) for shard_name, keys in sorted(grouped.items())]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _install_sidecar(temp_dir: Path, output_dir: Path, *, force: bool) -> None:
    if not output_dir.exists():
        os.replace(temp_dir, output_dir)
        return
    if not force:
        raise FileExistsError(
            f"Code2wav sidecar already exists at {output_dir}; pass --force to replace it"
        )

    backup = output_dir.with_name(f".code2wav-backup-{uuid.uuid4().hex}")
    os.replace(output_dir, backup)
    try:
        os.replace(temp_dir, output_dir)
    except Exception:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def prepare_code2wav_sidecar(
    model_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Create dense BF16 code2wav weights beside an MLX Qwen3-Omni checkpoint."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    output_dir = root / "code2wav"
    if output_dir.exists() and not force:
        raise FileExistsError(
            f"Code2wav sidecar already exists at {output_dir}; pass --force to replace it"
        )

    quantization = _load_quantization(root)
    index_path, weight_map = _load_weight_map(root)
    _validate_quantized_groups(weight_map)
    groups = _output_groups(weight_map)
    source_shards = sorted(
        {weight_map[key] for key in weight_map if key.startswith(_COMPONENT_PREFIX)}
    )
    expected_shapes = _expected_code2wav_shapes(root)
    temp_dir = Path(tempfile.mkdtemp(prefix=".code2wav-", dir=root))
    temp_dir.chmod(0o755)

    output_weight_map: dict[str, str] = {}
    total_size = 0
    try:
        shard_count = len(groups)
        for shard_number, (_, source_keys) in enumerate(groups, start=1):
            converted = _convert_output_group(
                root,
                weight_map,
                source_keys,
                quantization,
                expected_shapes,
            )
            shard_name = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            save_file(converted, temp_dir / shard_name)
            for key, tensor in converted.items():
                output_weight_map[key] = shard_name
                total_size += _tensor_bytes(tensor)
            del converted

        (temp_dir / _INDEX_NAME).write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": output_weight_map,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temp_dir / _MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "source_path": str(root),
                    "source_index_sha256": _sha256_file(index_path),
                    "source_shards_sha256": {
                        shard_name: _sha256_file(root / shard_name)
                        for shard_name in source_shards
                    },
                    "bits": quantization.bits,
                    "group_size": quantization.group_size,
                    "mode": quantization.mode,
                    "output_dtype": "bfloat16",
                    "tensor_count": len(output_weight_map),
                    "total_size": total_size,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _install_sidecar(temp_dir, output_dir, force=force)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Dequantize MLX affine code2wav weights into a Torch BF16 sidecar")
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing code2wav sidecar",
    )
    args = parser.parse_args()
    output_dir = prepare_code2wav_sidecar(args.model_path, force=args.force)
    manifest = _read_json(output_dir / _MANIFEST_NAME)
    print(
        f"Prepared {output_dir} "
        f"({manifest['tensor_count']} tensors, {manifest['total_size']} bytes)"
    )


if __name__ == "__main__":
    main()
