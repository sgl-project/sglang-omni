# SPDX-License-Identifier: Apache-2.0
"""Target-module normalization, packed/aliased binding, and merged LoRA manager."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from sglang_omni.lora.capability import (
    MergedLoRACapabilityError,
    MergedLoRASpec,
    is_sharded_linear_module,
    resolve_merged_lora_spec,
    resolve_submodule,
)
from sglang_omni.lora.merged_weights import MergedLinearWeights, ParkingPolicy


def normalize_target_modules(target_modules: object) -> tuple[str, ...] | str:
    """Normalize target_modules once at ingest, rejecting None, empty, or unknown types."""
    if isinstance(target_modules, str):
        stripped = target_modules.strip()
        if not stripped:
            raise ValueError("target_modules string must be non-empty")
        else:
            pass
        return stripped
    elif isinstance(target_modules, (list, tuple, set, frozenset)):
        if len(target_modules) == 0:
            raise ValueError("target_modules collection must be non-empty")
        else:
            pass
        cleaned: list[str] = []
        for item in target_modules:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "target_modules elements must be non-empty str, "
                    f"got {type(item).__name__}"
                )
            else:
                pass
            cleaned.append(item.strip())
        if isinstance(target_modules, (set, frozenset)):
            return tuple(sorted(set(cleaned)))
        else:
            deduped: list[str] = []
            seen: set[str] = set()
            for item in cleaned:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
                else:
                    pass
            return tuple(deduped)
    else:
        raise ValueError(
            "target_modules must be str, list, tuple, set, or frozenset, "
            f"got {type(target_modules).__name__}"
        )


def strip_known_prefixes(name: str, strip_prefixes: Sequence[str]) -> str:
    """Strip any matching configured prefix once from an incoming adapter name."""
    for prefix in strip_prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
        else:
            pass
    return name


def expand_name_aliases(name: str, name_aliases: Mapping[str, str]) -> tuple[str, ...]:
    """Return canonical candidate names by expanding exact and suffix aliases."""
    candidates: list[str] = [name]
    seen: set[str] = {name}
    for alias_key, alias_target in name_aliases.items():
        if name == alias_key and alias_target not in seen:
            seen.add(alias_target)
            candidates.append(alias_target)
        elif name.endswith("." + alias_key):
            replaced = name[: -len(alias_key)] + alias_target
            if replaced not in seen:
                seen.add(replaced)
                candidates.append(replaced)
            else:
                pass
        else:
            pass
        if name == alias_target and alias_key not in seen:
            seen.add(alias_key)
            candidates.append(alias_key)
        elif name.endswith("." + alias_target):
            replaced = name[: -len(alias_target)] + alias_key
            if replaced not in seen:
                seen.add(replaced)
                candidates.append(replaced)
            else:
                pass
        else:
            pass
    return tuple(candidates)


class TargetModuleMatcher:
    """Matcher shared by LoRA request validation and engine-side module binding."""

    def __init__(
        self,
        normalized: tuple[str, ...] | str,
        *,
        name_aliases: Mapping[str, str] = (),
        strip_prefixes: Sequence[str] = (),
    ) -> None:
        self.normalized = normalized
        self.name_aliases = dict(name_aliases)
        self.strip_prefixes = tuple(strip_prefixes)
        if isinstance(normalized, str):
            self.pattern: re.Pattern[str] | None = re.compile(normalized)
            self.exact_targets: frozenset[str] = frozenset()
        else:
            self.pattern = None
            self.exact_targets = frozenset(normalized)

    def matches(self, module_name: str) -> bool:
        """Return whether module_name matches after prefix stripping and alias expansion."""
        stripped = strip_known_prefixes(module_name, self.strip_prefixes)
        candidates = expand_name_aliases(stripped, self.name_aliases)
        if self.pattern is not None:
            for candidate in candidates:
                if self.pattern.search(candidate) is not None:
                    return True
                else:
                    pass
            return False
        else:
            for candidate in candidates:
                if candidate in self.exact_targets:
                    return True
                else:
                    pass
                leaf = candidate.rsplit(".", 1)[-1]
                if leaf in self.exact_targets:
                    return True
                else:
                    pass
            return False


def build_target_module_matcher(
    target_modules: object,
    *,
    name_aliases: Mapping[str, str] = (),
    strip_prefixes: Sequence[str] = (),
) -> TargetModuleMatcher:
    """Normalize target_modules and return a shared TargetModuleMatcher."""
    normalized = normalize_target_modules(target_modules)
    return TargetModuleMatcher(
        normalized,
        name_aliases=name_aliases,
        strip_prefixes=strip_prefixes,
    )


@dataclass(frozen=True)
class LoRAAdapterTensor:
    """One logical module's LoRA factor pair (delta = scaling * (lora_b @ lora_a))."""

    module_name: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scaling: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.module_name, str) or not self.module_name.strip():
            raise ValueError("LoRAAdapterTensor.module_name must be a non-empty str")
        else:
            pass
        if self.lora_a.ndim != 2 or self.lora_b.ndim != 2:
            raise ValueError(
                f"LoRA tensors for {self.module_name!r} must be 2-D, got "
                f"lora_a={tuple(self.lora_a.shape)}, lora_b={tuple(self.lora_b.shape)}"
            )
        else:
            pass
        if self.lora_a.shape[0] != self.lora_b.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {self.module_name!r}: "
                f"lora_a={tuple(self.lora_a.shape)}, lora_b={tuple(self.lora_b.shape)}"
            )
        else:
            pass


@dataclass(frozen=True)
class LoRARequestPayload:
    """Validated request carrying LoRA adapter tensors and target_modules."""

    adapter_id: str
    target_modules: tuple[str, ...] | str
    tensors: tuple[LoRAAdapterTensor, ...]
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_id, str) or not self.adapter_id.strip():
            raise ValueError("adapter_id must be a non-empty str")
        else:
            pass
        normalized = normalize_target_modules(self.target_modules)
        object.__setattr__(self, "target_modules", normalized)
        object.__setattr__(self, "tensors", tuple(self.tensors))


@dataclass(frozen=True)
class LinearBinding:
    """Resolved mapping from a logical adapter target to a physical weight slice."""

    logical_name: str
    param_name: str
    module: nn.Module
    row_offset: int
    row_size: int
    col_offset: int
    col_size: int
    full_out_features: int
    full_in_features: int


def infer_packed_output_sizes(
    module: nn.Module,
    sublayers: tuple[str, ...],
) -> tuple[int, ...]:
    """Infer per-sublayer output partition sizes on a packed physical linear."""
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        raise MergedLoRACapabilityError(
            f"packed module {type(module).__name__} lacks a 2-D weight parameter"
        )
    else:
        pass
    out_features = int(weight.shape[0])
    explicit_sizes = getattr(module, "output_sizes", None)
    if explicit_sizes is not None:
        sizes = tuple(int(s) for s in explicit_sizes)
        if len(sizes) != len(sublayers):
            raise MergedLoRACapabilityError(
                f"packed module {type(module).__name__} has {len(sizes)} output_sizes "
                f"but packed_modules_mapping declares {len(sublayers)} sublayers"
            )
        else:
            pass
        tp_size = int(getattr(module, "tp_size", 1))
        if sum(sizes) == out_features:
            return sizes
        elif tp_size > 1 and sum(sizes) == out_features * tp_size:
            return tuple(s // tp_size for s in sizes)
        else:
            raise MergedLoRACapabilityError(
                f"packed module {type(module).__name__} output_sizes {sizes} "
                f"do not match weight rows {out_features} (tp_size={tp_size})"
            )
    else:
        pass

    count = len(sublayers)
    if out_features % count != 0:
        raise MergedLoRACapabilityError(
            f"packed module {type(module).__name__} has {out_features} rows, "
            f"not evenly divisible by {count} packed sublayers {sublayers}"
        )
    else:
        pass
    part = out_features // count
    return tuple(part for _ in range(count))


def infer_sharding_offsets(
    module: nn.Module,
    local_rows: int,
    local_cols: int,
) -> tuple[int, int, int]:
    """Return (full_rows, col_offset, full_cols) for a linear module."""
    tp_size = int(getattr(module, "tp_size", 1))
    tp_rank = int(getattr(module, "tp_rank", 0))
    if tp_size <= 1:
        return local_rows, 0, local_cols
    else:
        pass
    explicit_out = getattr(module, "output_size", None)
    explicit_in = getattr(module, "input_size", None)
    if isinstance(explicit_out, int) and isinstance(explicit_in, int):
        col_offset = tp_rank * local_cols if explicit_in > local_cols else 0
        return explicit_out, col_offset, explicit_in
    else:
        pass
    if hasattr(module, "input_size_per_partition") and not hasattr(
        module, "output_size_per_partition"
    ):
        return local_rows, tp_rank * local_cols, local_cols * tp_size
    else:
        return local_rows * tp_size, 0, local_cols


class MergedLoRAManager:
    """Binds LoRA adapters to model linears and merges them into host-backed weights."""

    def __init__(
        self,
        model: nn.Module,
        *,
        architecture: str | None = None,
        parking_policy: ParkingPolicy | None = None,
        free_memory_fn: Callable[[torch.device], int | None] | None = None,
        prefer_pinned: bool = True,
        pin_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.architecture = architecture
        self.spec: MergedLoRASpec = resolve_merged_lora_spec(
            model, architecture=architecture
        )
        self.target_root = resolve_submodule(model, self.spec.target_module_root)
        self.bindings: dict[str, LinearBinding] = {}
        self.managed_parameters: dict[str, nn.Parameter] = {}
        self.build_bindings()
        self.weights = MergedLinearWeights(
            self.managed_parameters,
            parking_policy=parking_policy,
            free_memory_fn=free_memory_fn,
            prefer_pinned=prefer_pinned,
            pin_fn=pin_fn,
        )
        self.active_adapter_id: str | None = None

    def build_bindings(self) -> None:
        """Index standalone and packed linear modules under target_module_root."""
        root_prefix = (
            f"{self.spec.target_module_root}." if self.spec.target_module_root else ""
        )
        for rel_name, module in self.target_root.named_modules():
            if not rel_name:
                continue
            else:
                pass
            if not is_sharded_linear_module(module):
                if not (self.spec.allow_plain_linear and isinstance(module, nn.Linear)):
                    continue
                else:
                    pass
            else:
                pass

            weight = getattr(module, "weight", None)
            if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
                continue
            else:
                pass

            param_name = f"{root_prefix}{rel_name}.weight"
            self.managed_parameters[param_name] = weight
            local_rows = int(weight.shape[0])
            local_cols = int(weight.shape[1])
            full_rows, col_off, full_cols = infer_sharding_offsets(
                module, local_rows, local_cols
            )

            leaf = rel_name.rsplit(".", 1)[-1]
            parent_prefix = rel_name[: -len(leaf)]
            packed_sublayers = self.spec.packed_modules_mapping.get(leaf)
            if packed_sublayers is not None:
                sublayer_sizes = infer_packed_output_sizes(module, packed_sublayers)
                tp_size = int(getattr(module, "tp_size", 1))
                is_col_sharded = full_rows > local_rows
                offset = 0
                for sublayer, sub_size in zip(packed_sublayers, sublayer_sizes):
                    logical_name = f"{parent_prefix}{sublayer}"
                    full_sub_rows = sub_size * tp_size if is_col_sharded else sub_size
                    binding = LinearBinding(
                        logical_name=logical_name,
                        param_name=param_name,
                        module=module,
                        row_offset=offset,
                        row_size=sub_size,
                        col_offset=col_off,
                        col_size=local_cols,
                        full_out_features=full_sub_rows,
                        full_in_features=full_cols,
                    )
                    for alias_name in expand_name_aliases(
                        logical_name, self.spec.name_aliases
                    ):
                        self.bindings[alias_name] = binding
                    offset += sub_size
            else:
                binding = LinearBinding(
                    logical_name=rel_name,
                    param_name=param_name,
                    module=module,
                    row_offset=0,
                    row_size=local_rows,
                    col_offset=col_off,
                    col_size=local_cols,
                    full_out_features=full_rows,
                    full_in_features=full_cols,
                )
                for alias_name in expand_name_aliases(rel_name, self.spec.name_aliases):
                    self.bindings[alias_name] = binding

    def resolve_binding(self, adapter_module_name: str) -> LinearBinding | None:
        """Resolve an adapter-side logical module name to a physical LinearBinding."""
        stripped = strip_known_prefixes(adapter_module_name, self.spec.strip_prefixes)
        for candidate in expand_name_aliases(stripped, self.spec.name_aliases):
            binding = self.bindings.get(candidate)
            if binding is not None:
                return binding
            else:
                pass
        return None

    def compute_slice_delta(
        self,
        binding: LinearBinding,
        item: LoRAAdapterTensor,
        request_scale: float,
    ) -> torch.Tensor:
        """Compute the local-shard fp32 delta tensor for one logical LoRA binding."""
        lora_a = item.lora_a.detach().to(dtype=torch.float32)
        lora_b = item.lora_b.detach().to(dtype=torch.float32)
        tp_size = int(getattr(binding.module, "tp_size", 1))
        tp_rank = int(getattr(binding.module, "tp_rank", 0))

        # Slice full-width LoRA matrices to this rank's shard when TP > 1.
        if lora_b.shape[0] == binding.full_out_features and (
            binding.full_out_features != binding.row_size
        ):
            shard_start = tp_rank * binding.row_size
            lora_b = lora_b[shard_start : shard_start + binding.row_size, :]
        else:
            pass

        if lora_a.shape[1] == binding.full_in_features and (
            binding.full_in_features != binding.col_size
        ):
            lora_a = lora_a[
                :, binding.col_offset : binding.col_offset + binding.col_size
            ]
        else:
            pass

        if lora_b.shape[0] != binding.row_size or lora_a.shape[1] != binding.col_size:
            raise ValueError(
                f"LoRA shape mismatch for {item.module_name!r} -> {binding.param_name!r}: "
                f"expected local delta ({binding.row_size}, {binding.col_size}), "
                f"got lora_b={tuple(lora_b.shape)}, lora_a={tuple(lora_a.shape)} "
                f"(tp_size={tp_size})"
            )
        else:
            pass

        factor = float(item.scaling) * float(request_scale)
        return (lora_b @ lora_a) * factor

    def prepare_adapter(
        self,
        request: LoRARequestPayload,
        *,
        restore_omitted: bool = True,
    ) -> tuple[str, int]:
        """Bind request tensors, compute merged deltas, and stage them off live weights."""
        matcher = build_target_module_matcher(
            request.target_modules,
            name_aliases=self.spec.name_aliases,
            strip_prefixes=self.spec.strip_prefixes,
        )
        accumulated: dict[str, torch.Tensor] = {}
        matched_count = 0

        for item in request.tensors:
            if not matcher.matches(item.module_name):
                continue
            else:
                pass
            binding = self.resolve_binding(item.module_name)
            if binding is None:
                raise KeyError(
                    f"matched LoRA module {item.module_name!r} does not bind to any "
                    f"linear parameter on {type(self.model).__name__}"
                )
            else:
                pass

            param = self.managed_parameters[binding.param_name]
            slice_delta = self.compute_slice_delta(binding, item, request.scale)
            physical_delta = accumulated.get(binding.param_name)
            if physical_delta is None:
                physical_delta = torch.zeros(
                    tuple(param.shape),
                    dtype=torch.float32,
                    device=slice_delta.device,
                )
                accumulated[binding.param_name] = physical_delta
            else:
                pass

            if binding.param_name in accumulated:
                target_slice = physical_delta[
                    binding.row_offset : binding.row_offset + binding.row_size,
                    :,
                ]
                target_slice.add_(slice_delta.to(device=physical_delta.device))
            else:
                pass
            matched_count += 1

        if matched_count == 0:
            raise ValueError(
                f"LoRA adapter {request.adapter_id!r} matched 0 target modules "
                f"(target_modules={request.target_modules!r})"
            )
        else:
            pass

        park_location = self.weights.prepare(
            accumulated, restore_omitted=restore_omitted
        )
        return park_location, matched_count

    def commit_adapter(self, adapter_id: str) -> int:
        """Atomically commit staged weights in-place and record active_adapter_id."""
        updated = self.weights.commit()
        self.active_adapter_id = adapter_id
        return updated

    def rollback_adapter(self) -> None:
        """Drop staged weights without mutating live parameters."""
        self.weights.rollback()

    def restore_base(self) -> int:
        """Restore all modified parameters back to pristine host base weights."""
        restored = self.weights.restore_base()
        self.active_adapter_id = None
        return restored
