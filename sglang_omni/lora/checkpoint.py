# SPDX-License-Identifier: Apache-2.0
"""PEFT / safetensors LoRA checkpoint loader with optional step-distillation solver metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file

from sglang_omni.lora.merged_manager import normalize_target_modules
from sglang_omni.lora.solver_override import ODESolverOverride
from sglang_omni.lora.weight_transfer import (
    LoRAAdapterWeightSource,
    parse_lora_tensor_name,
)

ADAPTER_CONFIG_FILENAME = "adapter_config.json"
SOLVER_CONFIG_FILENAMES = ("ode_solver_config.json", "lora_solver_config.json")
EMBEDDED_SOLVER_KEYS = ("ode_solver", "solver_config", "step_distillation")
SAFETENSORS_WEIGHT_CANDIDATES = (
    "adapter_model.safetensors",
    "lora_weights.safetensors",
)
PT_WEIGHT_CANDIDATES = ("adapter_model.bin", "adapter_model.pt")


@dataclass(frozen=True)
class LoadedLoRACheckpoint:
    """Validated LoRA tensors and optional step-distillation ODE solver override from disk."""

    source: LoRAAdapterWeightSource
    solver_override: ODESolverOverride | None
    checkpoint_path: str
    config: dict[str, object]


def read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON file and require a top-level JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object in {path}, got {type(parsed).__name__}")
    else:
        return parsed


def load_weights_from_file(
    weight_file: Path, *, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Load LoRA factor tensors from a single .safetensors or .bin/.pt file."""
    suffix = weight_file.suffix.lower()
    if suffix == ".safetensors":
        raw_state = load_safetensors_file(str(weight_file), device=device)
        return {str(name): tensor for name, tensor in raw_state.items()}
    elif suffix in (".bin", ".pt", ".pth"):
        raw_state = torch.load(str(weight_file), map_location=device, weights_only=True)
        if not isinstance(raw_state, Mapping):
            raise ValueError(
                f"expected tensor mapping in {weight_file}, got {type(raw_state).__name__}"
            )
        else:
            pass
        nested_state = raw_state.get("state_dict")
        if isinstance(nested_state, Mapping):
            raw_state = nested_state
        else:
            pass
        return {
            str(name): tensor
            for name, tensor in raw_state.items()
            if isinstance(tensor, torch.Tensor)
        }
    else:
        raise ValueError(
            f"unsupported LoRA weight file extension {weight_file.suffix!r}; "
            "expected .safetensors, .bin, .pt, or .pth"
        )


def resolve_weight_files(root: Path) -> list[Path]:
    """Resolve one or more weight files from a LoRA checkpoint directory or file path."""
    if root.is_file():
        return [root]
    else:
        pass
    if not root.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint path does not exist: {root}")
    else:
        pass

    for filename in SAFETENSORS_WEIGHT_CANDIDATES:
        candidate = root / filename
        if candidate.is_file():
            return [candidate]
        else:
            pass

    sharded_safetensors = sorted(root.glob("*.safetensors"))
    if sharded_safetensors:
        return sharded_safetensors
    else:
        pass

    for filename in PT_WEIGHT_CANDIDATES:
        candidate = root / filename
        if candidate.is_file():
            return [candidate]
        else:
            pass

    raise FileNotFoundError(
        f"no LoRA weight files (.safetensors/.bin/.pt) found under {root}"
    )


def infer_target_modules_from_tensors(
    named_tensors: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Infer leaf target module suffixes when adapter_config.json does not list them."""
    inferred: list[str] = []
    for param_name in sorted(named_tensors):
        parsed = parse_lora_tensor_name(param_name)
        if parsed is None:
            continue
        else:
            pass
        base_name, _side = parsed
        leaf = base_name.rsplit(".", 1)[-1]
        if leaf and leaf not in inferred:
            inferred.append(leaf)
        else:
            pass
    if not inferred:
        raise ValueError(
            "could not infer target_modules: no PEFT lora_A / lora_B tensors found"
        )
    else:
        pass
    return tuple(inferred)


def discover_solver_override(
    root: Path,
    adapter_config: Mapping[str, object] | None = None,
    explicit_override: ODESolverOverride | Mapping[str, object] | None = None,
) -> ODESolverOverride | None:
    """Resolve ODE solver override from explicit input, sidecar file, or adapter_config."""
    if explicit_override is not None:
        parsed = ODESolverOverride.from_dict(explicit_override)
        return None if parsed.is_empty() else parsed
    else:
        pass

    if root.is_dir():
        for filename in SOLVER_CONFIG_FILENAMES:
            sidecar = root / filename
            if sidecar.is_file():
                parsed = ODESolverOverride.from_dict(read_json_object(sidecar))
                return None if parsed.is_empty() else parsed
            else:
                pass
    else:
        pass

    if adapter_config is not None:
        for embedded_key in EMBEDDED_SOLVER_KEYS:
            raw_embedded = adapter_config.get(embedded_key)
            if isinstance(raw_embedded, Mapping):
                parsed = ODESolverOverride.from_dict(raw_embedded)
                return None if parsed.is_empty() else parsed
            else:
                pass
    else:
        pass

    return None


def load_lora_checkpoint(
    lora_path: str | Path,
    *,
    adapter_id: str | None = None,
    scale: float | None = None,
    alpha: float | None = None,
    target_modules: object = None,
    solver_override: ODESolverOverride | Mapping[str, object] | None = None,
    device: str = "cpu",
) -> LoadedLoRACheckpoint:
    """Load a PEFT LoRA checkpoint directory or single weight file from disk.

    Reads ``adapter_config.json`` when present alongside the weights, discovers any
    step-distillation ODE solver override from explicit arguments, sidecar JSON
    (``ode_solver_config.json`` / ``lora_solver_config.json``), or embedded keys in
    ``adapter_config.json``, and returns a validated ``LoRAAdapterWeightSource``.
    """
    path = Path(lora_path).expanduser()
    weight_files = resolve_weight_files(path)

    adapter_config: dict[str, object] = {}
    config_dir = path if path.is_dir() else path.parent
    config_file = config_dir / ADAPTER_CONFIG_FILENAME
    if config_file.is_file():
        adapter_config = read_json_object(config_file)
    else:
        pass

    tensors: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        loaded = load_weights_from_file(weight_file, device=device)
        for name, tensor in loaded.items():
            if parse_lora_tensor_name(name) is not None:
                tensors[name] = tensor
            else:
                pass

    if not tensors:
        raise ValueError(
            f"no lora_A / lora_B factor tensors found in checkpoint {path}"
        )
    else:
        pass

    resolved_adapter_id = (
        adapter_id.strip()
        if isinstance(adapter_id, str) and adapter_id.strip()
        else path.stem if path.is_file() else path.name
    )
    if not resolved_adapter_id:
        raise ValueError("adapter_id could not be determined from lora_path")
    else:
        pass

    raw_targets = (
        target_modules
        if target_modules is not None
        else adapter_config.get("target_modules")
    )
    if raw_targets is None:
        resolved_targets: object = infer_target_modules_from_tensors(tensors)
    else:
        resolved_targets = normalize_target_modules(raw_targets)

    resolved_alpha = alpha
    if resolved_alpha is None and adapter_config.get("lora_alpha") is not None:
        resolved_alpha = float(adapter_config["lora_alpha"])  # type: ignore[arg-type]
    else:
        pass

    resolved_scale = scale
    if resolved_scale is None:
        raw_scale = adapter_config.get("lora_scale", 1.0)
        resolved_scale = float(raw_scale)  # type: ignore[arg-type]
    else:
        resolved_scale = float(resolved_scale)

    resolved_solver = discover_solver_override(path, adapter_config, solver_override)
    source = LoRAAdapterWeightSource(
        adapter_id=resolved_adapter_id,
        target_modules=resolved_targets,
        named_tensors=tensors,
        alpha=resolved_alpha,
        scale=resolved_scale,
    )
    return LoadedLoRACheckpoint(
        source=source,
        solver_override=resolved_solver,
        checkpoint_path=str(path),
        config=adapter_config,
    )


__all__ = [
    "ADAPTER_CONFIG_FILENAME",
    "EMBEDDED_SOLVER_KEYS",
    "LoadedLoRACheckpoint",
    "SOLVER_CONFIG_FILENAMES",
    "discover_solver_override",
    "infer_target_modules_from_tensors",
    "load_lora_checkpoint",
    "load_weights_from_file",
    "resolve_weight_files",
]
