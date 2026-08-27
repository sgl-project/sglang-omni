# SPDX-License-Identifier: Apache-2.0
"""Where every configuration value came from.

The resolver records one :class:`ProvenanceEntry` per contributing patch, plus
the baseline the patches were applied on top of. That is what makes
``sgl-omni config explain <path>`` answerable::

    stages.thinker.engine.mem_fraction_static = 0.8
      0.87  <- model default (Qwen3OmniConfig)
      0.70  <- yaml file (configs/omni.yaml)            [superseded]
      0.80  <- cli flag (--thinker-mem-fraction-static) [winner]

Nothing here decides anything; precedence lives in
:mod:`sglang_omni.config.patch` and is executed by
:mod:`sglang_omni.config.resolver`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)

__all__ = ["ProvenanceEntry", "ProvenanceMap", "BASELINE_SOURCE"]

BASELINE_SOURCE = ConfigSource(
    SourceKind.MODEL_DEFAULT,
    detail="value before any patch was applied",
)

_UNSET = object()
"""Distinguishes "no resolved value recorded" from a recorded ``None``."""


@dataclass(frozen=True)
class ProvenanceEntry:
    """One source's contribution to one path: the patch, and whether it won."""

    patch: ConfigPatch
    winning: bool

    @property
    def path(self) -> str:
        return self.patch.path.raw

    @property
    def value(self) -> Any:
        return self.patch.value

    @property
    def source(self) -> ConfigSource:
        return self.patch.source

    def render(self, resolved: Any = _UNSET) -> str:
        marker = "[winner]" if self.winning else "[superseded]"
        line = f"{self.value!r}  <- {self.source.describe()}  {marker}"
        if self.winning and resolved is not _UNSET and resolved != self.value:
            # The model's own validation rewrote the value after this patch
            # won; showing only what the source said would claim it stuck.
            line += f" (resolved to {resolved!r})"
        return line


@dataclass
class ProvenanceMap:
    """Contribution history per canonical path."""

    entries: dict[str, list[ProvenanceEntry]] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    """Pre-patch value per path, recorded only for paths a patch touched."""

    resolved: dict[str, Any] = field(default_factory=dict)
    """Post-validation value per touched path, read back off the built config."""

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------

    def record(self, patch: ConfigPatch, *, winning: bool) -> None:
        self.entries.setdefault(patch.path.raw, []).append(
            ProvenanceEntry(patch, winning)
        )

    def record_baseline(self, path: str, value: Any) -> None:
        self.baseline.setdefault(path, value)

    def record_resolved(self, path: str, value: Any) -> None:
        self.resolved[path] = value

    @classmethod
    def from_patchset(cls, patchset: ConfigPatchSet) -> "ProvenanceMap":
        out = cls()
        winners = patchset.winners()
        for patch in patchset.ordered():
            out.record(patch, winning=winners[patch.key] is patch)
        return out

    # ------------------------------------------------------------------
    # querying
    # ------------------------------------------------------------------

    def paths(self) -> list[str]:
        return sorted(self.entries)

    def winner(self, path: str) -> ProvenanceEntry | None:
        for entry in reversed(self.entries.get(path, [])):
            if entry.winning:
                return entry
        return None

    def touched(self, path: str) -> bool:
        return path in self.entries

    def resolved_value(self, path: str, default: Any = None) -> Any:
        """The value the built config holds at a touched path."""
        return self.resolved.get(path, default)

    def explain(self, path: str) -> str:
        """Human-readable answer to 'why is this value what it is'.

        The headline reports the value the built config actually holds --
        validation may have rewritten what the winning source supplied. The
        winner's own line keeps the source's value, with the rewrite noted.

        Callers ask :meth:`touched` first; an untouched path raises ``KeyError``.
        """
        history = self.entries[path]
        winner = self.winner(path)
        resolved = self.resolved.get(path, _UNSET)
        if winner is None:
            headline = path
        elif resolved is _UNSET:
            headline = f"{path} = {winner.value!r}"
        else:
            headline = f"{path} = {resolved!r}"
        lines = [headline]
        if path in self.baseline:
            lines.append(f"  {self.baseline[path]!r}  <- {BASELINE_SOURCE.describe()}")
        lines.extend(f"  {entry.render(resolved)}" for entry in history)
        return "\n".join(lines)
