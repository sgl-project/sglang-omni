# SPDX-License-Identifier: Apache-2.0
"""The single place where configuration sources are merged.

``ConfigResolver.resolve`` takes a baseline config plus a
:class:`~sglang_omni.config.patch.ConfigPatchSet` and produces exactly two
things: the validated configuration, and the provenance that explains it.

What it deliberately does *not* do:

* it does not know about YAML, CLI flags or Router workers — sources normalize
  themselves into patches before they get here;
* it does not compute placement, process topology or SGLang server args —
  those are downstream consumers of the resolved value, and must not write
  back into it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sglang_omni.config.patch import ConfigPatch, ConfigPatchSet
from sglang_omni.config.path import ConfigPath, ConfigPathError
from sglang_omni.config.provenance import ProvenanceMap
from sglang_omni.config.schema import PipelineConfig

__all__ = ["ConfigResolver", "ResolvedConfig", "ConfigDifference", "diff_configs"]


@dataclass(frozen=True)
class ResolvedConfig:
    """A validated configuration together with the story of how it got there."""

    config: PipelineConfig
    provenance: ProvenanceMap
    patches: ConfigPatchSet

    def value(self, path: str) -> Any:
        return ConfigPath.parse(path, type(self.config)).read(self.config)


class ConfigResolver:
    """Applies a patch set to a baseline config."""

    def __init__(self, base: PipelineConfig) -> None:
        self._base = base

    @property
    def config_cls(self) -> type[PipelineConfig]:
        return type(self._base)

    def resolve(self, patchset: ConfigPatchSet) -> ResolvedConfig:
        patchset.require_no_conflicts()

        data = self._base.model_dump()
        provenance = ProvenanceMap.from_patchset(patchset)

        ordered = patchset.ordered()
        for patch in ordered:
            provenance.record_baseline(patch.key, _safe_read(patch.path, data))

        for patch in ordered:
            _apply(data, patch)

        # The baseline dump carries the name model_post_init derived from the
        # baseline's model_path. When a patch replaces model_path and nothing
        # sets name explicitly, clear the stale derivation so rebuilding
        # rederives it from the new model_path.
        touched = {patch.key for patch in ordered}
        if (
            "model_path" in touched
            and "name" not in touched
            and data.get("name") == self._base.model_path
        ):
            data["name"] = None

        config = self.config_cls(**data)

        # What the built config actually holds at each touched path. Not the
        # same thing as the winning patch's value: model validation may
        # rewrite a field after assignment, and provenance that ignored the
        # rewrite would explain a value the launch does not use.
        resolved_data = config.model_dump()
        for patch in ordered:
            provenance.record_resolved(patch.key, _safe_read(patch.path, resolved_data))

        return ResolvedConfig(config=config, provenance=provenance, patches=patchset)


# ----------------------------------------------------------------------
# application
# ----------------------------------------------------------------------


def _apply(data: dict[str, Any], patch: ConfigPatch) -> None:
    """Assign a leaf, or deep-merge a mapping written at a container path."""
    if patch.path.is_leaf or not isinstance(patch.value, dict):
        patch.path.write(data, deepcopy(patch.value))
        return

    existing = _safe_read(patch.path, data)
    if isinstance(existing, dict):
        patch.path.write(data, _deep_merge(existing, patch.value))
    else:
        patch.path.write(data, deepcopy(patch.value))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _safe_read(path: ConfigPath, data: dict[str, Any]) -> Any:
    """Read a path that may not exist yet (a new mapping key, for instance)."""
    try:
        return path.read(data)
    except ConfigPathError:
        return None


# ----------------------------------------------------------------------
# config comparison
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigDifference:
    path: str
    expected: Any
    actual: Any

    def render(self) -> str:
        return f"{self.path}: expected {self.expected!r}, got {self.actual!r}"


def diff_configs(
    expected: PipelineConfig | dict[str, Any],
    actual: PipelineConfig | dict[str, Any],
) -> list[ConfigDifference]:
    """Compare two configs field by field, addressing stages by name.

    Two readers: ``sgl-omni config resolve --show diff``, which reports what
    the sources changed about the model's own defaults, and the V1 parity gate
    in ``tests/unit_test/config/test_v1_parity.py``, which requires this to
    come back empty for every probe the frozen V1 oracle also accepted.
    """
    return _diff(_as_dump(expected), _as_dump(actual), "")


def _as_dump(value: PipelineConfig | dict[str, Any]) -> dict[str, Any]:
    return value.model_dump() if isinstance(value, PipelineConfig) else value


def _diff(expected: Any, actual: Any, prefix: str) -> list[ConfigDifference]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[ConfigDifference] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                out.append(ConfigDifference(child, expected.get(key), actual.get(key)))
                continue
            out.extend(_diff(expected[key], actual[key], child))
        return out

    if _is_named_list(expected) and _is_named_list(actual):
        out = []
        expected_by_name = {item["name"]: item for item in expected}
        actual_by_name = {item["name"]: item for item in actual}
        for name in sorted(set(expected_by_name) | set(actual_by_name)):
            child = f"{prefix}.{name}" if prefix else name
            if name not in expected_by_name or name not in actual_by_name:
                out.append(
                    ConfigDifference(
                        child, expected_by_name.get(name), actual_by_name.get(name)
                    )
                )
                continue
            out.extend(_diff(expected_by_name[name], actual_by_name[name], child))
        return out

    if expected != actual:
        return [ConfigDifference(prefix, expected, actual)]
    return []


def _is_named_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and "name" in item for item in value)
    )
