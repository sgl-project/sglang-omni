from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class CompilerError(Exception):
    pass


class UnsupportedFeatureError(CompilerError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise CompilerError(f"{path} must contain a mapping at the top level")
    return data


def source_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:16]


def workflow_stem(path: Path) -> str:
    return path.stem


def resolve_workflow_path(repo_root: Path, uses: str) -> Path:
    uses = uses.strip()
    if uses.startswith("./"):
        return (repo_root / uses.removeprefix("./")).resolve()
    if uses.startswith(".github/workflows/"):
        return (repo_root / uses).resolve()
    raise UnsupportedFeatureError(f"unsupported reusable workflow reference: {uses!r}")


def normalize_needs(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise CompilerError(f"invalid needs value: {value!r}")


def is_reusable_workflow_call(job: dict[str, Any]) -> bool:
    uses = job.get("uses")
    return isinstance(uses, str) and (
        uses.endswith(".yaml") or uses.endswith(".yml")
    )


DEFAULT_SCHEDULER_ROOTS: tuple[str, ...] = ("omni-ci.yaml",)


def root_workflow_paths(workflows_dir: Path, *, roots: tuple[str, ...] = DEFAULT_SCHEDULER_ROOTS) -> list[Path]:
    return [workflows_dir / name for name in roots]
