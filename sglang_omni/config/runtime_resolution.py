# SPDX-License-Identifier: Apache-2.0
"""Observing values that only become concrete at launch.

Several engine fields carry ``None`` through the whole config pipeline on
purpose: ``None`` means "auto", and SGLang's ``ServerArgs.__post_init__``
resolves it from the GPU it lands on (``chunked_prefill_size`` becomes 2048
on a small card, 8192 on an H100, and so on). Until that moment the value
does not exist anywhere in this repo — no config source can know it, and
deriving anything from it earlier reads a value that is not there yet.

This module is the channel for those observations:

* :func:`capture_runtime_resolutions` diffs the overrides a builder passed
  into ``ServerArgs`` against the constructed object and reports every field
  the runtime filled in or rewrote.
* :class:`RuntimeResolutionRecord` keeps the observations queryable inside
  the stage worker process that made them.
* :func:`require_resolved` is the guardrail for derivation code: it refuses
  an unresolved "auto" with a message naming the safe place to run instead.

Deliberately not patches: a runtime resolution is a fact about what the
launch did, not a configuration write, so it never enters patch precedence
(:mod:`sglang_omni.config.patch`) and never mutates the config model
(:mod:`sglang_omni.config.resolver`). Provenance renders it as a separate
line via :meth:`~sglang_omni.config.provenance.ProvenanceMap.record_runtime`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PENDING",
    "RUNTIME_RESOLVED_FIELDS",
    "RuntimeResolution",
    "RuntimeResolutionRecord",
    "RUNTIME_RESOLUTION_RECORD",
    "capture_runtime_resolutions",
    "require_resolved",
]


class _Pending:
    """Sentinel for "will be resolved at launch, on hardware we cannot see"."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "auto (resolved from GPU memory at launch)"


PENDING = _Pending()
"""Resolved value used by GPU-less previews (``sgl-omni config resolve``)."""


RUNTIME_RESOLVED_FIELDS: dict[str, str] = {
    "chunked_prefill_size": "chunked_prefill_size",
    "mem_fraction_static": "mem_fraction_static",
    # The omni override name; ServerArgs has no attribute of that name — the
    # builder aliases it to cuda_graph_max_bs_decode and the resolved value
    # lands on the nested cuda_graph_config.
    "cuda_graph_max_bs": "cuda_graph_config.decode.max_bs",
}
"""Override name -> dotted ``ServerArgs`` attribute path for "auto" fields.

The set mirrors what SGLang's ``ServerArgs._handle_gpu_memory_settings``
fills in from device memory (``max_total_tokens`` is not among them: it is
resolved later, from the allocated KV pool). Extend it when upstream grows
a new hardware-resolved field.
"""

SERVER_ARGS_HARDWARE_RESOLUTION = "sglang ServerArgs hardware resolution"
"""Origin string for values SGLang filled in during construction."""

_UNSET = object()


@dataclass(frozen=True)
class RuntimeResolution:
    """One field the runtime settled after configuration was finished."""

    path: str
    """Dotted config path or bare ServerArgs field name."""

    configured: Any
    """What the configuration supplied (``None`` == "auto")."""

    resolved: Any
    """What the constructed ``ServerArgs`` holds (or :data:`PENDING`)."""

    origin: str
    """Who settled it, e.g. :data:`SERVER_ARGS_HARDWARE_RESOLUTION`."""

    def describe(self) -> str:
        configured = "auto" if self.configured is None else repr(self.configured)
        return f"{self.path}: {configured} -> {self.resolved!r} ({self.origin})"


def _read_attr_path(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = getattr(obj, part, _UNSET)
        if obj is _UNSET:
            return _UNSET
    return obj


def capture_runtime_resolutions(
    configured: Mapping[str, Any],
    server_args: Any,
    *,
    fields: Mapping[str, str] = RUNTIME_RESOLVED_FIELDS,
) -> list[RuntimeResolution]:
    """Report every hardware-resolved field the constructed args changed.

    ``configured`` is the overrides mapping handed to ``ServerArgs``; a field
    absent from it counts as unset (``None``), because ``ServerArgs`` defaults
    it to ``None`` and resolves from there. Fields the runtime left exactly as
    configured are not reported — a resolution is a change of value, and an
    explicitly-set field passing through unchanged is not one.
    """
    out: list[RuntimeResolution] = []
    for name, attr_path in fields.items():
        resolved = _read_attr_path(server_args, attr_path)
        if resolved is _UNSET:
            continue
        supplied = configured.get(name)
        if resolved == supplied:
            continue
        out.append(
            RuntimeResolution(
                path=name,
                configured=supplied,
                resolved=resolved,
                origin=SERVER_ARGS_HARDWARE_RESOLUTION,
            )
        )
    return out


def require_resolved(value: Any, *, field_name: str, hint: str = "") -> Any:
    """Refuse to derive from a value that is still "auto".

    Derivation code that consumes a hardware-resolved field calls this at the
    top; before ``ServerArgs`` construction the field is ``None`` and the
    derived result would be built on a value that does not exist yet (the
    prefill CUDA graph ladder bug class). Returns ``value`` unchanged when it
    is concrete.
    """
    if value is None:
        message = (
            f"{field_name} is still unresolved ('auto'); it only becomes "
            "concrete inside ServerArgs construction. Derive from it in "
            "finalize_runtime_derived(), which runs after resolution — "
            "not in adjust_overrides()."
        )
        if hint:
            message += f" {hint}"
        raise ValueError(message)
    return value


@dataclass
class RuntimeResolutionRecord:
    """Process-local registry of what each stage's launch resolved.

    Lives in the stage worker process that constructed the ``ServerArgs``;
    the parent's provenance cannot be reached from here (the config crosses
    a spawn boundary as plain kwargs), so diagnostics in the same process
    query this instead.
    """

    _by_stage: dict[str, list[RuntimeResolution]] = field(default_factory=dict)

    def record(self, stage: str, resolutions: list[RuntimeResolution]) -> None:
        self._by_stage[stage] = list(resolutions)

    def get(self, stage: str) -> list[RuntimeResolution]:
        return list(self._by_stage.get(stage, []))

    def all(self) -> dict[str, list[RuntimeResolution]]:
        return {stage: list(items) for stage, items in self._by_stage.items()}


RUNTIME_RESOLUTION_RECORD = RuntimeResolutionRecord()
"""The worker process's registry; builders also keep their own list."""
