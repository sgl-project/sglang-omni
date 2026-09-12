# SPDX-License-Identifier: Apache-2.0
"""Selective HF INT4 loading; repack calibrated codes, never requantize weights."""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack

import torch
from safetensors import safe_open
from torch import nn

from sglang_omni.models.weight_loader import resolve_model_path

from .torch_mps_quantization import (
    GROUP_SIZES,
    MpsQuantizedExperts,
    MpsQuantizedLinear,
    unpack_int4,
)

logger = logging.getLogger(__name__)


def _json_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint metadata key {key!r}")
        result[key] = value
    return result


def read_mps_quantization_config(config):
    """Accept the HF symmetric W4A16 layouts, not arbitrary AWQ/GPTQ formats."""
    quant = config.get("quantization_config")
    error = (
        "Torch MPS INT4 requires a symmetric compressed-tensors pack-quantized "
        "or AutoRound auto_round:auto_gptq checkpoint; MLX affine and other "
        "quantization formats are not supported"
    )
    if not isinstance(quant, dict) or config.get("quantization") is not None:
        raise ValueError(error)
    method = quant.get("quant_method")
    if method == "compressed-tensors":
        groups = quant.get("config_groups", {})
        if not isinstance(groups, dict) or len(groups) != 1:
            raise ValueError(error)
        scheme = next(iter(groups.values()))
        if not isinstance(scheme, dict) or not isinstance(scheme.get("weights"), dict):
            raise ValueError(error)
        weights = scheme.get("weights") or {}
        if (
            quant.get("format") != "pack-quantized"
            or scheme.get("input_activations") is not None
            or scheme.get("output_activations") is not None
            or weights.get("num_bits") != 4
            or weights.get("symmetric") is not True
            or weights.get("strategy") != "group"
            or weights.get("type") != "int"
            or weights.get("dynamic", False)
            or weights.get("actorder") is not None
        ):
            raise ValueError(error)
        group_size = weights.get("group_size")
    elif method == "auto-round":
        overrides = quant.get("extra_config", {})
        if not isinstance(overrides, dict) or any(
            not isinstance(value, dict) or value.get("bits") != 16
            for value in overrides.values()
        ):
            raise ValueError(error)
        if (
            quant.get("packing_format") != "auto_round:auto_gptq"
            or quant.get("bits") != 4
            or quant.get("sym") is not True
            or quant.get("data_type", "int") != "int"
            or quant.get("act_bits", 16) != 16
            or quant.get("desc_act", False)
        ):
            raise ValueError(error)
        group_size = quant.get("group_size")
    else:
        raise ValueError(error)
    if group_size not in GROUP_SIZES:
        raise ValueError(error)
    return method, group_size


class _Source:
    def __init__(self, directory, prefixes, stack):
        config = json.loads(
            (directory / "config.json").read_text(), object_pairs_hook=_json_unique
        )
        self.format, self.group_size = read_mps_quantization_config(config)
        self.entries, self.used = {}, set()
        handles, shard_keys = {}, {}
        index = directory / "model.safetensors.index.json"
        weight_map = (
            json.loads(index.read_text(), object_pairs_hook=_json_unique).get(
                "weight_map"
            )
            if index.exists()
            else None
        )
        if index.exists() and (not isinstance(weight_map, dict) or not weight_map):
            raise ValueError("Checkpoint index must contain a nonempty weight_map")

        def open_shard(filename):
            if not isinstance(filename, str):
                raise ValueError("Checkpoint shard names must be strings")
            path = (directory / filename).resolve()
            if path.parent != directory.resolve() or not path.is_file():
                raise ValueError(f"Invalid or missing checkpoint shard {filename!r}")
            if filename not in handles:
                handles[filename] = stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                shard_keys[filename] = set(handles[filename].keys())
            return handles[filename]

        if weight_map is None:
            entries = (
                (key, path.name)
                for path in sorted(directory.glob("*.safetensors"))
                for key in open_shard(path.name).keys()
            )
        else:
            entries = weight_map.items()
        for key, filename in entries:
            if not isinstance(key, str):
                raise ValueError("Checkpoint tensor names must be strings")
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                local = key[len(prefix) :]
                if prefix == "thinker." and local.startswith(
                    ("visual.", "audio_tower.", "vision_tower.")
                ):
                    break
                if local in self.entries:
                    raise ValueError(f"Duplicate checkpoint weight {local!r}")
                handle = open_shard(filename)
                if key not in shard_keys[filename]:
                    raise ValueError(f"Missing indexed checkpoint weight {key!r}")
                self.entries[local] = (handle, key)
                break
        if not self.entries:
            raise ValueError(f"Missing checkpoint weights for prefixes {prefixes}")
        self.quantized_linears = 0

    def read(self, name, shape=None):
        if name not in self.entries:
            raise ValueError(f"Missing checkpoint weight {name!r}")
        handle, key = self.entries[name]
        value = handle.get_tensor(key)
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"Shape mismatch for {name}: {tuple(value.shape)}, expected {tuple(shape)}"
            )
        self.used.add(name)
        return value

    def linear(self, name, rows, columns, *, bias, dtype, device):
        stem = f"{name}." if name else ""
        if f"{stem}weight" in self.entries:
            with torch.device("meta"):
                result = nn.Linear(columns, rows, bias=bias is not None)
            result.weight = nn.Parameter(
                self.read(f"{stem}weight", (rows, columns)).to(
                    device=device, dtype=dtype
                ),
                requires_grad=False,
            )
            if bias is not None:
                result.bias = nn.Parameter(
                    bias.to(device=device, dtype=dtype), requires_grad=False
                )
            return result
        groups = columns // self.group_size
        if columns % self.group_size:
            raise ValueError(f"Input width is not divisible by group size for {name}")
        if self.format == "compressed-tensors":
            shape = self.read(f"{stem}weight_shape", (2,)).tolist()
            if shape != [rows, columns]:
                raise ValueError(f"Shape mismatch for {name}: {shape}")
            codes = unpack_int4(
                self.read(f"{stem}weight_packed", (rows, (columns + 7) // 8))
            )[:, :columns]
            scales = self.read(f"{stem}weight_scale", (rows, groups))
        else:
            codes = unpack_int4(self.read(f"{stem}qweight", (columns // 8, rows)).T)
            scales = self.read(f"{stem}scales", (groups, rows)).T
            zeros = unpack_int4(self.read(f"{stem}qzeros", (groups, (rows + 7) // 8)))[
                :, :rows
            ]
            # AutoRound's GPTQ format stores zero_point - 1. Symmetric INT4 is 8.
            if not torch.all(zeros == 7):
                raise ValueError(f"Unsupported AutoRound zero points for {name}")
        self.quantized_linears += 1
        return MpsQuantizedLinear(
            codes,
            scales,
            torch.zeros_like(scales),
            self.group_size,
            bias=bias,
            dtype=dtype,
            device=device,
        )


def load_quantized_mps_module(module, model_path, *, prefix, bits, dtype, device):
    """Load one meta stage; preserve dense exclusions and read one expert at a time."""
    MpsQuantizedLinear._validate_target(bits, dtype, device)
    directory = resolve_model_path(model_path)
    prefixes = (prefix,) if isinstance(prefix, str) else prefix
    if not prefixes or any(not isinstance(value, str) for value in prefixes):
        raise ValueError("Checkpoint prefixes must be strings")
    assigned, aliases = {}, {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    with ExitStack() as stack:
        source = _Source(directory, prefixes, stack)

        def visit(current, path):
            stem = f"{path}." if path else ""
            gate_up = current._parameters.get("gate_up_proj")
            down = current._parameters.get("down_proj")
            if gate_up is not None and down is not None and gate_up.ndim == 3:
                count, doubled, hidden = gate_up.shape
                if doubled % 2 or tuple(down.shape) != (count, hidden, doubled // 2):
                    raise ValueError(f"Invalid expert shell at {path}")
                projections = ([], [], [])
                for expert in range(count):
                    for target, projection in zip(
                        projections, ("gate_proj", "up_proj", "down_proj")
                    ):
                        rows, columns = (
                            (hidden, doubled // 2)
                            if projection == "down_proj"
                            else (doubled // 2, hidden)
                        )
                        target.append(
                            source.linear(
                                f"{path}.{expert}.{projection}",
                                rows,
                                columns,
                                bias=None,
                                dtype=dtype,
                                device=device,
                            )
                        )
                return MpsQuantizedExperts(*projections, current.act_fn)
            if (
                isinstance(current, nn.Linear)
                and id(current.weight) not in assigned
                and any(
                    f"{stem}{suffix}" in source.entries
                    for suffix in ("weight", "weight_packed", "qweight")
                )
            ):
                bias = (
                    None
                    if current.bias is None
                    else source.read(f"{stem}bias", current.bias.shape)
                )
                result = source.linear(
                    path,
                    current.out_features,
                    current.in_features,
                    bias=bias,
                    dtype=dtype,
                    device=device,
                )
                if isinstance(result, nn.Linear):
                    assigned[id(current.weight)] = result.weight
                return result
            for name, original in list(current._parameters.items()):
                if original is None:
                    continue
                full = f"{stem}{name}"
                if id(original) in assigned:
                    value = assigned[id(original)]
                    if full in source.entries and full not in source.used:
                        duplicate = source.read(full, original.shape).to(dtype=dtype)
                        if not torch.equal(duplicate, value.cpu()):
                            raise ValueError(
                                f"Conflicting tied checkpoint weight {full}"
                            )
                else:
                    serialized = next(
                        (
                            alias
                            for alias in (full, *aliases[id(original)])
                            if alias in source.entries
                        ),
                        full,
                    )
                    value = nn.Parameter(
                        source.read(serialized, original.shape).to(
                            device=device, dtype=dtype
                        ),
                        requires_grad=False,
                    )
                    assigned[id(original)] = value
                setattr(current, name, value)
            for name, original in list(current._buffers.items()):
                if original is None:
                    continue
                full = f"{stem}{name}"
                if full in source.entries:
                    original = source.read(full, original.shape)
                elif original.is_meta:
                    raise ValueError(f"Missing initialization buffer {full}")
                setattr(
                    current,
                    name,
                    original.to(
                        device=device,
                        dtype=dtype if original.is_floating_point() else original.dtype,
                    ),
                )
            for name, child in list(current.named_children()):
                setattr(current, name, visit(child, f"{stem}{name}"))
            return current

        module = visit(module, "")
        unexpected = source.entries.keys() - source.used
        if unexpected:
            raise ValueError(
                f"Unexpected checkpoint weights: {sorted(unexpected)[:12]}"
            )
        logger.info(
            "Qwen3-Omni Torch MPS quantization bits=4 prefix=%s "
            "quantized_linears=%d source=%s (dense exclusions preserved)",
            prefix,
            source.quantized_linears,
            source.format,
        )
    return module.eval()
